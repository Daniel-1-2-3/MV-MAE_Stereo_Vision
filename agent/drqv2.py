"""MV-MAE + DrQ-v2 agent.

* Representation: the MV-MAE encoder's unmasked tokens for the whole stereo
  video clip, flattened (as DrQ-v2 flattens its conv features). Baselines, the
  same agent without MV-MAE: agent.encoder="cnn" uses DrQ-v2's conv encoder on the
  raw frames, trained by the critic only; agent.encoder="pixels" has no learned
  image encoder at all (shrunk raw pixels straight into the actor and critic).
* Actor-critic: DrQ-v2 -- twin Q critics with a slow target copy, an actor with
  scheduled Gaussian exploration noise, n-step returns, random-shift image
  augmentation.
* The encoder is trained by the critic loss (DrQ-v2) plus the MV-MAE
  reconstruction and reward-prediction losses. The actor does not update the
  encoder.
* Demonstrations (optional): demo samples in a batch add a behaviour-cloning
  term to the actor loss, with the Q term normalised as in TD3+BC.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvmae.model import MVMAE

from . import utils


@dataclass
class Batch:
    obs: torch.Tensor  # (B, F, V, 3, H, W) uint8
    action: torch.Tensor  # (B, A)
    reward: torch.Tensor  # (B,) n-step discounted return
    discount: torch.Tensor  # (B,) gamma^m, 0 if the episode terminated inside the window
    next_obs: torch.Tensor  # (B, F, V, 3, H, W) uint8
    is_demo: torch.Tensor  # (B,) bool
    frame_reward: torch.Tensor  # (B, F) reward of the transition into each frame
    frame_reward_valid: torch.Tensor  # (B, F) bool
    proprio: torch.Tensor  # (B, P) robot state at obs's last frame; P = 0 when proprioception is off
    next_proprio: torch.Tensor  # (B, P) robot state at next_obs's last frame


class ConvEncoder(nn.Module):
    """DrQ-v2's image encoder (Yarats et al. 2021), for the baseline without MV-MAE.

    All stacked frames of both views go in as channels (F * V * 3), so the left/right pair is
    fused from the first layer: 4 conv layers (32 channels, 3x3, first with stride 2), ReLU,
    flattened. Pixels are scaled to [-0.5, 0.5] as in DrQ-v2. 96 x 96 input -> 32 x 41 x 41.
    """

    def __init__(self, in_channels: int, img_hw: tuple[int, int]):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        self.apply(utils.weight_init)
        with torch.no_grad():
            self.repr_dim = int(self.convnet(torch.zeros(1, in_channels, *img_hw)).numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F, V, 3, H, W) uint8 or float in [0, 255] -> (B, repr_dim)."""
        b, f, v, c, h, w = x.shape
        x = x.float().reshape(b, f * v * c, h, w) / 255.0 - 0.5
        return self.convnet(x).flatten(1)


class PixelEncoder(nn.Module):
    """Raw-pixel baseline: no learned image encoder.

    The stacked frames of both views, scaled to [-0.5, 0.5] and shrunk by averaging
    `downsample` x `downsample` pixel blocks, are flattened and passed on as they are. The
    first learned layer is the actor's / critic's own trunk (Linear -> LayerNorm -> Tanh),
    exactly as for the other encoders. No parameters.
    """

    def __init__(self, in_channels: int, img_hw: tuple[int, int], downsample: int):
        super().__init__()
        self.downsample = int(downsample)
        h, w = img_hw
        self.repr_dim = in_channels * (h // self.downsample) * (w // self.downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F, V, 3, H, W) uint8 or float in [0, 255] -> (B, repr_dim)."""
        b, f, v, c, h, w = x.shape
        x = x.float().reshape(b, f * v * c, h, w) / 255.0 - 0.5
        if self.downsample > 1:
            x = F.avg_pool2d(x, self.downsample)
        return x.flatten(1)


def _proprio_trunk(proprio_dim: int, feature_dim: int) -> nn.Module | None:
    """Robot-state features (same shape as the image trunk's output), or None without proprioception."""
    if proprio_dim <= 0:
        return None
    return nn.Sequential(nn.Linear(proprio_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())


def _features(trunk: nn.Module, proprio_trunk: nn.Module | None, z: torch.Tensor, proprio: torch.Tensor | None):
    h = trunk(z)
    if proprio_trunk is not None:
        h = torch.cat([h, proprio_trunk(proprio)], dim=-1)
    return h


class Actor(nn.Module):
    def __init__(self, repr_dim: int, action_dim: int, feature_dim: int, hidden_dim: int, proprio_dim: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.proprio_trunk = _proprio_trunk(proprio_dim, feature_dim)
        in_dim = feature_dim * (2 if self.proprio_trunk is not None else 1)
        self.policy = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, z: torch.Tensor, proprio: torch.Tensor | None, std: float) -> utils.TruncatedNormal:
        mu = torch.tanh(self.policy(_features(self.trunk, self.proprio_trunk, z, proprio)))
        return utils.TruncatedNormal(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    def __init__(self, repr_dim: int, action_dim: int, feature_dim: int, hidden_dim: int, proprio_dim: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.proprio_trunk = _proprio_trunk(proprio_dim, feature_dim)
        in_dim = feature_dim * (2 if self.proprio_trunk is not None else 1)

        def q_net():
            return nn.Sequential(
                nn.Linear(in_dim + action_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        self.Q1, self.Q2 = q_net(), q_net()
        self.apply(utils.weight_init)

    def forward(self, z: torch.Tensor, proprio: torch.Tensor | None, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([_features(self.trunk, self.proprio_trunk, z, proprio), action], dim=-1)
        return self.Q1(h).squeeze(-1), self.Q2(h).squeeze(-1)


class MVMAEDrQV2Agent:
    def __init__(self, agent_cfg, mvmae_cfg, img_hw: tuple[int, int], num_views: int, action_dim: int, device,
                 proprio_dim: int = 0):
        self.cfg = agent_cfg
        self.proprio_dim = proprio_dim  # > 0: the actor and critic also get the robot's own state
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.frame_stack = mvmae_cfg.frame_stack
        self.encoder_kind = getattr(agent_cfg, "encoder", "mvmae")
        self.uses_mvmae = self.encoder_kind == "mvmae"
        channels = mvmae_cfg.frame_stack * num_views * 3
        if self.encoder_kind == "cnn":  # baseline: plain conv encoder on the images
            self.mvmae = None
            self.encoder = ConvEncoder(channels, img_hw).to(self.device)
            self.repr_dim = self.encoder.repr_dim
        elif self.encoder_kind == "pixels":  # baseline: raw (shrunk) pixels, no image encoder
            self.mvmae = None
            self.encoder = PixelEncoder(channels, img_hw, getattr(agent_cfg, "pixel_downsample", 2)).to(self.device)
            self.repr_dim = self.encoder.repr_dim
        else:
            self._build_mvmae(mvmae_cfg, img_hw, num_views)
        c = agent_cfg
        self.actor = Actor(self.repr_dim, action_dim, c.feature_dim, c.hidden_dim, proprio_dim).to(self.device)
        self.critic = Critic(self.repr_dim, action_dim, c.feature_dim, c.hidden_dim, proprio_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

        # (the raw-pixel baseline has no encoder parameters: its optimiser holds an empty placeholder)
        enc_params = list(self.encoder.parameters()) or [nn.Parameter(torch.zeros(0, device=self.device))]
        self.encoder_opt = torch.optim.Adam(enc_params, lr=c.encoder_lr)
        # The transformer gets a learning-rate warm-up; DrQ-v2's conv encoder has none.
        warmup = max(1, c.encoder_warmup_updates) if self.uses_mvmae else 1
        self.encoder_sched = torch.optim.lr_scheduler.LambdaLR(
            self.encoder_opt, lambda u: min(1.0, (u + 1) / warmup)
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.aug = utils.RandomShiftsAug(c.aug_pad)
        self.use_amp = bool(c.amp) and self.device.type == "cuda"
        self.num_updates = 0
        self.encoder_frozen = False  # see freeze_encoder()

    def _build_mvmae(self, mvmae_cfg, img_hw, num_views) -> None:
        self.mvmae = MVMAE(
            img_hw=img_hw,
            num_views=num_views,
            num_frames=mvmae_cfg.frame_stack,
            patch_size=mvmae_cfg.patch_size,
            embed_dim=mvmae_cfg.embed_dim,
            encoder_depth=mvmae_cfg.encoder_depth,
            encoder_heads=mvmae_cfg.encoder_heads,
            decoder_dim=mvmae_cfg.decoder_dim,
            decoder_depth=mvmae_cfg.decoder_depth,
            decoder_heads=mvmae_cfg.decoder_heads,
            mlp_ratio=mvmae_cfg.mlp_ratio,
            mask_ratio=mvmae_cfg.mask_ratio,
            loss_on_masked_only=mvmae_cfg.loss_on_masked_only,
            reward_prediction=mvmae_cfg.reward_prediction,
        ).to(self.device)
        self.encoder = self.mvmae
        self.repr_dim = self.mvmae.L * mvmae_cfg.embed_dim

    # ----------------------------------------------------------------- utils
    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 (B, F, V, 3, H, W) -> float [0, 255] with one random shift per sample."""
        b, f, v, c, h, w = x.shape
        return self.aug(x.float().reshape(b, f * v * c, h, w)).reshape(b, f, v, c, h, w)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.uses_mvmae:
            return self.encoder(x)  # float32, like DrQ-v2
        with self._autocast():
            z = self.mvmae.encode(x)
        return z.float().flatten(1)

    def stddev(self, env_step: int) -> float:
        return utils.schedule(self.cfg.stddev_schedule, env_step)

    def train(self, training: bool = True) -> None:
        for m in (self.encoder, self.actor, self.critic):
            m.train(training)

    # ------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: torch.Tensor, env_step: int, eval_mode: bool, proprio: torch.Tensor | None = None) -> torch.Tensor:
        """obs: (N, F, V, 3, H, W) uint8 (+ proprio (N, P) when proprio_dim > 0) -> actions (N, A) in [-1, 1]."""
        z = self._encode(obs.to(self.device))
        p = self._proprio(proprio)
        dist = self.actor(z, p, self.stddev(env_step))
        return dist.mean if eval_mode else dist.sample(clip=None)

    def _proprio(self, proprio: torch.Tensor | None) -> torch.Tensor | None:
        if self.proprio_dim <= 0:
            return None
        if proprio is None or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"this agent needs the robot state (proprio, {self.proprio_dim} values) with every observation")
        return proprio.to(self.device).float()

    def freeze_encoder(self) -> None:
        """Stop all further encoder training (MV-MAE and critic gradients alike; for the conv baseline, the critic's).

        The actor and critic keep learning on the now-fixed features, like policies
        trained on a frozen pretrained encoder.
        """
        self.encoder_frozen = True
        self.encoder.requires_grad_(False)

    # ---------------------------------------------------------------- update
    def update(self, batch: Batch, env_step: int) -> dict[str, torch.Tensor]:
        c = self.cfg
        metrics: dict[str, torch.Tensor] = {}
        std = self.stddev(env_step)
        do_mae = (self.uses_mvmae and c.mae_coef > 0 and not self.encoder_frozen
                  and self.num_updates % max(1, c.mae_every) == 0)

        obs = self._augment(batch.obs)
        next_obs = self._augment(batch.next_obs)
        p, p_next = self._proprio(batch.proprio), self._proprio(batch.next_proprio)
        s = utils.schedule(str(c.critic_encoder_grad_scale), env_step) if c.critic_grad_to_encoder else 0.0
        if self.encoder_frozen:
            s = 0.0
        metrics["train/critic_encoder_grad_scale"] = torch.tensor(s, device=self.device)
        metrics["train/encoder_frozen"] = torch.tensor(float(self.encoder_frozen), device=self.device)
        if s > 0.0:
            z = self._encode(obs)
            if s != 1.0:
                z = z * s + z.detach() * (1.0 - s)  # same value, critic gradient into the encoder scaled by s
        else:
            with torch.no_grad():
                z = self._encode(obs)
        with torch.no_grad():
            z_next = self._encode(next_obs)
            next_action = self.actor(z_next, p_next, std).sample(clip=c.stddev_clip)
            target_q1, target_q2 = self.critic_target(z_next, p_next, next_action)
            target_q = batch.reward + batch.discount * torch.min(target_q1, target_q2)

        # ---- critic (+ encoder)
        q1, q2 = self.critic(z, p, batch.action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        total = critic_loss
        if do_mae:
            with self._autocast():
                recon, reward_loss, _ = self.mvmae.losses(obs, batch.frame_reward, batch.frame_reward_valid)
            total = total + c.mae_coef * (recon + c.reward_pred_coef * reward_loss)
            metrics["mvmae/recon_loss"] = recon.detach()
            metrics["mvmae/reward_loss"] = reward_loss.detach()
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        total.backward()
        if not self.encoder_frozen:
            metrics["grad/encoder"] = nn.utils.clip_grad_norm_(self.encoder.parameters(), c.max_grad_norm)
        metrics["grad/critic"] = nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
        if not self.encoder_frozen:
            self.encoder_opt.step()
            self.encoder_sched.step()
        self.critic_opt.step()

        # ---- actor (encoder frozen for this step, as in DrQ-v2)
        z_actor = z.detach()
        dist = self.actor(z_actor, p, std)
        action = dist.sample(clip=c.stddev_clip)
        q = torch.min(*self.critic(z_actor, p, action))
        use_bc = c.bc_coef > 0 and bool(batch.is_demo.any())
        if use_bc:
            lam = 1.0 / q.abs().mean().detach().clamp(min=1e-6)
            demo = batch.is_demo
            bc_loss = F.mse_loss(dist.mean[demo], batch.action[demo])
            actor_loss = -(lam * q).mean() + c.bc_coef * bc_loss
            metrics["actor/bc_loss"] = bc_loss.detach()
        else:
            actor_loss = -q.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        metrics["grad/actor"] = nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        self.actor_opt.step()

        utils.soft_update_params(self.critic, self.critic_target, c.critic_tau)
        self.num_updates += 1

        metrics.update(
            {
                "critic/loss": critic_loss.detach(),
                "critic/q1": q1.detach().mean(),
                "critic/q2": q2.detach().mean(),
                "critic/target_q": target_q.mean(),
                "actor/loss": actor_loss.detach(),
                "actor/q": q.detach().mean(),
                "train/batch_reward": batch.reward.mean(),
                "train/batch_demo_frac": batch.is_demo.float().mean(),
            }
        )
        return metrics

    def update_mae_only(self, batch: Batch) -> dict[str, torch.Tensor]:
        """MV-MAE pre-training step (no RL losses)."""
        if not self.uses_mvmae:
            raise RuntimeError("the baselines have no MV-MAE to pre-train")
        c = self.cfg
        obs = self._augment(batch.obs)
        with self._autocast():
            recon, reward_loss, _ = self.mvmae.losses(obs, batch.frame_reward, batch.frame_reward_valid)
        loss = recon + c.reward_pred_coef * reward_loss
        self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(self.mvmae.parameters(), c.max_grad_norm)
        self.encoder_opt.step()
        self.encoder_sched.step()
        return {"mvmae/recon_loss": recon.detach(), "mvmae/reward_loss": reward_loss.detach(), "grad/encoder": grad}

    # -------------------------------------------------------- visualisation
    @torch.no_grad()
    def reconstruction_image(self, obs: torch.Tensor) -> torch.Tensor:
        """One sample -> uint8 (H*F, W*6, 3) grid.

        One row per frame: truth L | truth R | visible L | visible R | recon L | recon R.
        Hidden regions are grey in the "visible" panels; the reconstruction panels
        show the model's prediction for hidden regions and the truth elsewhere.
        MV-MAE only.
        """
        m = self.mvmae
        x = obs[:1].to(self.device)
        with self._autocast():
            out = m.forward_mae(x)
        x_norm = m.normalize(x)
        pix_mask = m.unpatchify(out["mask"][..., None].expand(-1, -1, m.p * m.p * 3).float()).bool()
        pred = m.unpatchify(out["pred"].float())
        truth = m.denormalize(x_norm)
        visible = truth.clone()
        visible[pix_mask] = 128
        recon = m.denormalize(torch.where(pix_mask, pred, x_norm))
        rows = []
        for f in range(m.F):
            panels = [truth[0, f, 0], truth[0, f, 1], visible[0, f, 0], visible[0, f, 1], recon[0, f, 0], recon[0, f, 1]]
            rows.append(torch.cat(panels, dim=-1))  # (3, H, 6W)
        return torch.cat(rows, dim=-2).permute(1, 2, 0).cpu()

    # ----------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        return {
            ("mvmae" if self.uses_mvmae else "encoder"): self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "encoder_sched": self.encoder_sched.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "num_updates": self.num_updates,
            "encoder_frozen": self.encoder_frozen,
        }

    def load_state_dict(self, state: dict, load_optimizers: bool = True) -> None:
        key = "mvmae" if self.uses_mvmae else "encoder"
        if key not in state:
            raise ValueError(f"checkpoint has no '{key}' weights: it was trained with the other agent.encoder setting")
        self.encoder.load_state_dict(state[key])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        if load_optimizers:
            self.encoder_opt.load_state_dict(state["encoder_opt"])
            self.encoder_sched.load_state_dict(state["encoder_sched"])
            self.critic_opt.load_state_dict(state["critic_opt"])
            self.actor_opt.load_state_dict(state["actor_opt"])
        self.num_updates = state.get("num_updates", 0)
        if state.get("encoder_frozen", False):
            self.freeze_encoder()
