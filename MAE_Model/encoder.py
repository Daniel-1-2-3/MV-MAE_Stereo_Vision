import functools
from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from MAE_Model.sincos_pos_embeds import PosEmbed
from MAE_Model.vit import VitBlock

class ViTMaskedEncoder(nn.Module):
    nviews: int = 2
    patch_size: int = 6
    embed_dim: int = 768
    in_channels: int = 3
    img_h_size: int = 84
    img_w_fused_size: int = 168  # width of fused image (all views stitched horizontally)
    heads: int = 8
    depth: int = 8
    masking_ratio: float = 0.75

    def setup(self):
        # Conv stem that produces patch embeddings
        self.forward_conv = self._construct_conv_layers()

        # Transformer blocks
        self.vit_blocks = [
            VitBlock(
                embed_dim=self.embed_dim,
                heads=self.heads,
                mlp_ratio=4.0,
                attn_drop_rate=0.1,
                mlp_drop_rate=0.1,
                path_drop_rate=0.05,
            )
            for _ in range(self.depth)
        ]

        self.norm = nn.LayerNorm(epsilon=1e-6)

        # Positional embeddings (constant buffer)
        each_view_w = self.img_w_fused_size // self.nviews
        each_view_h = self.img_h_size

        pe_np = PosEmbed.get_2d_sincos_pos_embed(
            self.embed_dim,
            int(each_view_h // self.patch_size),
            int(each_view_w // self.patch_size),
        )  # (grid_h * grid_w, embed_dim)

        pe = jnp.array(pe_np).repeat(self.nviews, axis=0)  # (total_patches, embed_dim)
        self.pos_embed_all = self.param(
            "pos_embed_all",
            lambda rng, shape: pe,
            pe.shape,
        )

    def __call__(self, x: jnp.ndarray, mask_x: bool, *, deterministic: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Args:
            x: (batch, height, width_total, channels) where width_total = fused across views
            mask_x: whether to apply masking
            deterministic: True for eval/inference, False for training

        Returns:
            x_out:
              - if mask_x: (batch, unmasked_patches, embed_dim)
              - else:      (batch, total_patches, embed_dim)
            mask:
              - if mask_x: (batch, total_patches) with 0=keep, 1=mask
              - else:      None
        """
        x = self.forward_early_conv(x)          # (B, total_patches, D)
        x = self.add_pos_embeds(x)              # (B, total_patches, D)

        mask = None
        if mask_x:
            x, mask = self.random_view_masking(x)  # (B, kept, D), (B, total_patches)

        for block in self.vit_blocks:
            x = block(x, deterministic=deterministic)

        x = x = self.norm(x)
        return x, mask

    def random_view_masking(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Masks x such that either the left or right view is fully masked, while
        the other view is partially masked according to masking_ratio.

        x: (batch, patches_left + patches_right, embed_dim)
        """
        batch, num_patches, _ = x.shape
        half = num_patches // 2

        # portion to mask on the partially-masked side
        mask_ratio = (self.masking_ratio - 0.5) / 0.5
        num_mask = int(mask_ratio * half)
        keep_per_half = half - num_mask

        rng = self.make_rng("mask")
        rng_side, rng_scores = jax.random.split(rng, 2)

        # choose which side to fully mask (True => fully mask right)
        mask_right = jax.random.uniform(rng_side, (batch,)) > 0.5

        # random scores for each patch in one half; keep smallest scores
        scores = jax.random.uniform(rng_scores, (batch, half))
        keep_left = jnp.argsort(scores, axis=1)[:, :keep_per_half]
        keep_right = keep_left + half

        keep_idx = jnp.where(mask_right[:, None], keep_left, keep_right)  # (B, keep_per_half)

        mask = jnp.ones((batch, num_patches), dtype=jnp.float32)
        row = jnp.arange(batch)[:, None]
        mask = mask.at[row, keep_idx].set(0.0)

        # gather kept patches
        x_masked = jax.vmap(lambda xb, idx: xb[idx])(x, keep_idx)  # (B, keep_per_half, D)
        return x_masked, mask

    # Helpers
    def add_pos_embeds(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.pos_embed_all[None, :, :].astype(x.dtype)

    def _construct_conv_layers(self) -> nn.Sequential:
        layers = []

        # compute stride factors that multiply to patch_size (keeps your original behavior)
        stride_factors = []
        n = self.patch_size
        while n > 1:
            if n % 2 == 0:
                stride_factors.append(2)
                n //= 2
            else:
                stride_factors.append(n)
                break

        for i, s in enumerate(stride_factors):
            out_channels = max(self.embed_dim // (2 ** (len(stride_factors) - i)), 32)
            layers.append(
                nn.Conv(
                    features=out_channels,
                    kernel_size=(max(3, s), max(3, s)),
                    strides=(s, s),
                    padding=((max(3, s) // 2, max(3, s) // 2), (max(3, s) // 2, max(3, s) // 2)),
                )
            )
            layers.append(nn.relu)

        # final projection to embed_dim
        layers.append(
            nn.Conv(
                features=self.embed_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
            )
        )

        return nn.Sequential(layers)

    def forward_early_conv(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (B, H, W_total, C)

        Returns:
            (B, total_patches, embed_dim)
        """
        batch, height, width_total, channels = x.shape
        width_per_view = width_total // self.nviews

        # (B, H, W_total, C) -> (B*nviews, H, W_view, C)
        x = x.reshape(batch, height, self.nviews, width_per_view, channels)
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(batch * self.nviews, height, width_per_view, channels)

        x = self.forward_conv(x)  # (B*nviews, H', W', D)

        # (B*nviews, H', W', D) -> (B, total_patches, D)
        x = x.reshape(x.shape[0], -1, x.shape[-1])                  # (B*nviews, patches, D)
        x = x.reshape(batch, self.nviews, x.shape[1], x.shape[2])   # (B, nviews, patches, D)
        x = jnp.transpose(x, (0, 2, 1, 3))                          # (B, patches, nviews, D)
        x = x.reshape(batch, -1, x.shape[-1])                       # (B, patches*nviews, D)
        return x
