"""Multi-View Masked Autoencoder (MV-MAE), after Seo et al. 2023,
"Multi-View Masked World Models for Visual Robotic Manipulation" (arXiv:2302.02408).

Input: a short video from several cameras, shape (B, F, V, 3, H, W), uint8 or
float in [0, 255]. Here V = 2 (left / right eye of the stereo rig) and F = the
frame stack.

* Convolutional stem (shared by every view and frame) turns each image into an
  (H/p) x (W/p) grid of feature tokens -- masking features instead of raw pixel
  patches, as in the paper.
* Every token gets a fixed 2-D sin-cos position embedding plus learnable
  embeddings saying which camera and which frame it came from.
* View masking: for every frame one camera is hidden completely and random
  tokens are hidden from the remaining camera, so ``mask_ratio`` of all tokens
  are hidden. The decoder has to redraw the hidden camera from the other one
  and from the neighbouring frames (video autoencoding).
* ViT encoder / ViT decoder, linear head to pixels, plus a reward head.

For control the encoder is run with nothing masked (``encode``); the RL agent
uses its token outputs as the state representation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sincos_pos_embed_2d(dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Fixed 2-D sin-cos position embedding, shape (grid_h * grid_w, dim)."""
    assert dim % 4 == 0, "embedding dim must be divisible by 4 for 2-D sin-cos embeddings"

    def embed_1d(d: int, pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    gy, gx = np.meshgrid(np.arange(grid_h, dtype=np.float64), np.arange(grid_w, dtype=np.float64), indexing="ij")
    return np.concatenate([embed_1d(dim // 2, gy), embed_1d(dim // 2, gx)], axis=1).astype(np.float32)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ConvStem(nn.Module):
    """log2(patch) stride-2 convolutions + 1x1 projection: (B, 3, H, W) -> (B, D, H/p, W/p)."""

    def __init__(self, patch_size: int, dim: int):
        super().__init__()
        n_down = int(math.log2(patch_size))
        chans = [max(dim // 2 ** (n_down - i), 32) for i in range(1, n_down + 1)]
        layers, c_in = [], 3
        for c_out in chans:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1), nn.GELU()]
            c_in = c_out
        layers.append(nn.Conv2d(c_in, dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MVMAE(nn.Module):
    def __init__(
        self,
        img_hw: tuple[int, int],
        num_views: int,
        num_frames: int,
        patch_size: int = 16,
        embed_dim: int = 256,
        encoder_depth: int = 8,
        encoder_heads: int = 4,
        decoder_dim: int = 256,
        decoder_depth: int = 6,
        decoder_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.95,
        loss_on_masked_only: bool = True,
        reward_prediction: bool = True,
    ):
        super().__init__()
        h, w = img_hw
        self.img_hw = (h, w)
        self.V, self.F, self.p = num_views, num_frames, patch_size
        self.gh, self.gw = h // patch_size, w // patch_size
        self.N = self.gh * self.gw  # tokens per image
        self.L = self.F * self.V * self.N  # tokens per sample
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.loss_on_masked_only = loss_on_masked_only
        self.reward_prediction = reward_prediction

        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0, persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0, persistent=False)

        # ---- encoder
        self.stem = ConvStem(patch_size, embed_dim)
        self.register_buffer(
            "enc_pos", torch.from_numpy(sincos_pos_embed_2d(embed_dim, self.gh, self.gw)), persistent=False
        )
        self.enc_view = nn.Parameter(torch.zeros(self.V, embed_dim))
        self.enc_frame = nn.Parameter(torch.zeros(self.F, embed_dim))
        self.encoder = nn.ModuleList(Block(embed_dim, encoder_heads, mlp_ratio) for _ in range(encoder_depth))
        self.enc_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # ---- decoder
        self.dec_in = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer(
            "dec_pos", torch.from_numpy(sincos_pos_embed_2d(decoder_dim, self.gh, self.gw)), persistent=False
        )
        self.dec_view = nn.Parameter(torch.zeros(self.V, decoder_dim))
        self.dec_frame = nn.Parameter(torch.zeros(self.F, decoder_dim))
        self.decoder = nn.ModuleList(Block(decoder_dim, decoder_heads, mlp_ratio) for _ in range(decoder_depth))
        self.dec_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.pixel_head = nn.Linear(decoder_dim, patch_size * patch_size * 3)
        if reward_prediction:
            self.reward_head = nn.Sequential(nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, 1))

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        for p in (self.enc_view, self.enc_frame, self.dec_view, self.dec_frame, self.mask_token):
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------- helpers
    def _check(self, x: torch.Tensor) -> None:
        if x.dim() != 6 or x.shape[1:4] != (self.F, self.V, 3) or tuple(x.shape[-2:]) != self.img_hw:
            raise ValueError(f"expected (B, {self.F}, {self.V}, 3, {self.img_hw[0]}, {self.img_hw[1]}), got {tuple(x.shape)}")

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F, V, 3, H, W) in [0, 255] -> ImageNet-normalized float, same shape."""
        b = x.shape[0]
        flat = x.reshape(-1, 3, *self.img_hw).float()
        return ((flat - self.img_mean) / self.img_std).reshape(b, self.F, self.V, 3, *self.img_hw)

    def patchify(self, x_norm: torch.Tensor) -> torch.Tensor:
        """(B, F, V, 3, H, W) -> (B, L, p*p*3), token order (frame, view, row, col)."""
        b, p = x_norm.shape[0], self.p
        x = x_norm.reshape(b, self.F, self.V, 3, self.gh, p, self.gw, p)
        x = x.permute(0, 1, 2, 4, 6, 5, 7, 3)  # b f v gh gw p p c
        return x.reshape(b, self.L, p * p * 3)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse of patchify: (B, L, p*p*3) -> (B, F, V, 3, H, W)."""
        b, p = tokens.shape[0], self.p
        x = tokens.reshape(b, self.F, self.V, self.gh, self.gw, p, p, 3)
        x = x.permute(0, 1, 2, 7, 3, 5, 4, 6)  # b f v c gh p gw p
        return x.reshape(b, self.F, self.V, 3, *self.img_hw)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized (..., 3, H, W) -> uint8."""
        shape = x_norm.shape
        x = x_norm.reshape(-1, 3, *self.img_hw) * self.img_std + self.img_mean
        return x.clamp(0, 255).round().to(torch.uint8).reshape(shape)

    def _tokens(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Embedded tokens (B, L, D), order (frame, view, row, col)."""
        b = x_norm.shape[0]
        feat = self.stem(x_norm.reshape(-1, 3, *self.img_hw))  # (B*F*V, D, gh, gw)
        feat = feat.flatten(2).transpose(1, 2)  # (B*F*V, N, D)
        feat = feat.reshape(b, self.F, self.V, self.N, self.embed_dim)
        feat = (
            feat
            + self.enc_pos.to(feat.dtype)
            + self.enc_view.to(feat.dtype)[None, None, :, None, :]
            + self.enc_frame.to(feat.dtype)[None, :, None, None, :]
        )
        return feat.reshape(b, self.L, self.embed_dim)

    def _run_encoder(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.encoder:
            tokens = blk(tokens)
        return self.enc_norm(tokens)

    # ----------------------------------------------------------- masking
    def view_mask(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns keep indices (B, K) and a mask (B, L) with 1 = hidden, 0 = visible.

        Per frame one random view is hidden entirely; the K visible tokens are
        drawn at random from the other views' tokens across all frames, with
        K = round((1 - mask_ratio) * L).
        """
        hidden_view = torch.randint(self.V, (batch, self.F), device=device)  # (B, F)
        view_of_token = torch.arange(self.V, device=device).repeat_interleave(self.N)  # (V*N,)
        view_of_token = view_of_token.repeat(self.F)  # (L,) order (frame, view, token)
        frame_of_token = torch.arange(self.F, device=device).repeat_interleave(self.V * self.N)  # (L,)
        candidate = view_of_token[None, :] != hidden_view[:, frame_of_token]  # (B, L)
        num_candidates = self.F * (self.V - 1) * self.N
        keep = max(1, min(num_candidates, int(round((1.0 - self.mask_ratio) * self.L))))
        scores = torch.rand(batch, self.L, device=device)
        scores = scores.masked_fill(~candidate, 2.0)  # non-candidates sort last
        keep_idx = scores.argsort(dim=1)[:, :keep]
        keep_idx, _ = keep_idx.sort(dim=1)
        mask = torch.ones(batch, self.L, device=device)
        mask.scatter_(1, keep_idx, 0.0)
        return keep_idx, mask

    # ------------------------------------------------------------ public
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Unmasked representation for control: (B, F, V, 3, H, W) -> (B, L, D)."""
        self._check(x)
        return self._run_encoder(self._tokens(self.normalize(x)))

    def forward_mae(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Masked reconstruction pass. Returns predictions, targets and the mask."""
        self._check(x)
        b = x.shape[0]
        x_norm = self.normalize(x)
        tokens = self._tokens(x_norm)
        keep_idx, mask = self.view_mask(b, x.device)
        visible = torch.gather(tokens, 1, keep_idx[..., None].expand(-1, -1, tokens.shape[-1]))
        latent = self._run_encoder(visible)

        dec_visible = self.dec_in(latent)
        d = dec_visible.shape[-1]
        full = self.mask_token.to(dec_visible.dtype).expand(b, self.L, d).clone()
        full.scatter_(1, keep_idx[..., None].expand(-1, -1, d), dec_visible)
        full = full.reshape(b, self.F, self.V, self.N, d)
        full = (
            full
            + self.dec_pos.to(full.dtype)
            + self.dec_view.to(full.dtype)[None, None, :, None, :]
            + self.dec_frame.to(full.dtype)[None, :, None, None, :]
        ).reshape(b, self.L, d)
        for blk in self.decoder:
            full = blk(full)
        full = self.dec_norm(full)

        out = {"pred": self.pixel_head(full), "target": self.patchify(x_norm), "mask": mask}
        if self.reward_prediction:
            per_frame = full.reshape(b, self.F, self.V * self.N, d).mean(dim=2)  # (B, F, d)
            out["reward_pred"] = self.reward_head(per_frame).squeeze(-1)  # (B, F)
        return out

    def losses(
        self, x: torch.Tensor, reward_targets: torch.Tensor | None = None, reward_valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Reconstruction loss and (optional) reward-prediction loss.

        Args:
            x: (B, F, V, 3, H, W) images.
            reward_targets: (B, F) reward associated with each frame (the reward of
                the transition that led to it).
            reward_valid: (B, F) bool, False where a frame has no such reward.
        """
        out = self.forward_mae(x)
        per_token = (out["pred"].float() - out["target"].float()).pow(2).mean(dim=-1)  # (B, L)
        if self.loss_on_masked_only:
            recon = (per_token * out["mask"]).sum() / out["mask"].sum().clamp(min=1.0)
        else:
            recon = per_token.mean()
        reward_loss = torch.zeros((), device=x.device)
        if self.reward_prediction and reward_targets is not None and reward_valid is not None:
            valid = reward_valid.float()
            err = (out["reward_pred"].float() - reward_targets.float()).pow(2)
            reward_loss = (err * valid).sum() / valid.sum().clamp(min=1.0)
        return recon, reward_loss, out
