import os
from typing import Optional, Tuple

import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    """GPU->GPU via DLPack (stream-aware)."""
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        # Older stacks only accept a capsule.
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    """GPU->GPU via DLPack capsule path (avoids Torch calling __dlpack__ directly)."""
    return tpack.from_dlpack(jax.dlpack.to_dlpack(x))


class RSLRLBraxWrapper:
    """Batched MJX env wrapper for a Torch RL loop (DrQv2).

    Design constraints:
    - Physics is *single-world* inside the env and is batched here via vmap + jit.
    - Madrona renderer init/render must NEVER run under vmap/jit.
    - Transfers are GPU->GPU using DLPack.

    NOTE ABOUT YOUR CURRENT FAILURE MODE
    ------------------------------------
    If Madrona's custom call leaves the CUDA *context* on XLA's worker thread in a bad state,
    then *any* subsequent JAX GPU kernel (even `rgb + 0`) can fail with errors like:
        "cuModuleGetFunction ... cudaErrorInvalidValue"
    That cannot be fully repaired from Python by calling cuCtxSetCurrent on the Python thread.
    This wrapper therefore keeps rendering isolated (no extra JAX kernels after render),
    and provides an explicit "render health check" you can toggle.
    """

    def __init__(
        self,
        env,
        batch_size: int,
        seed: int,
        episode_length: int,
        action_repeat: int = 1,
        render_callback=None,
        randomization_fn=None,
        device_rank: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self._raw_env = env
        self.batch_size = int(batch_size)
        self.num_envs = int(batch_size)
        self.episode_length = int(episode_length)
        self.action_repeat = int(action_repeat)

        # Torch device for learner tensors
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # Safety: renderer num_worlds must match batch size
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"StereoPickCube(render_batch_size=...) must equal num_envs. "
                    f"Got render_batch_size={rbs}, num_envs={self.batch_size}."
                )

        # RNG keys per env
        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        # Batched physics: reset_physics / step_physics
        self._reset_physics_batched = jax.vmap(self._raw_env.reset_physics)
        self._step_physics_batched = jax.vmap(self._raw_env.step_physics, in_axes=(0, 0), out_axes=0)

        self.reset_fn = jax.jit(self._reset_physics_batched)

        if self.action_repeat == 1:
            self.step_fn = jax.jit(self._step_physics_batched)
        else:
            def _step_repeat(state, action):
                def body(_, st):
                    return self._step_physics_batched(st, action)
                return jax.lax.fori_loop(0, self.action_repeat, body, state)

            self.step_fn = jax.jit(_step_repeat)

        self.env_state = None  # set in reset()

        # Optional debugging: after a render, try a tiny JAX kernel to detect poisoning early.
        # If this fails, the only real fixes are (a) patch Madrona custom call save/restore ctx,
        # or (b) switch away from PJRT CUDA plugin to legacy cuda-enabled jaxlib.
        self._render_healthcheck = os.environ.get("MADRONA_RENDER_HEALTHCHECK", "0") == "1"

    def _render_rgba_batched(self) -> jax.Array:
        """Non-jit render, returns jax.Array uint8 [B,2,H,W,4]."""
        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        self._raw_env._ensure_render_token(self.env_state.data, debug)
        rgba = self._raw_env.render_rgba(self.env_state.data)

        # Ensure the render custom call has executed before we hand off via DLPack.
        jax.block_until_ready(rgba)

        if self._render_healthcheck:
            # This is deliberately tiny; if this fails, it's strong evidence the renderer
            # poisoned the CUDA context on XLA's worker thread.
            _ = jp.asarray(1, dtype=jp.int32) + jp.asarray(1, dtype=jp.int32)
            jax.block_until_ready(_)

        return rgba

    @staticmethod
    def _stereo_rgba_to_torch_obs(rgb_t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Convert [B,2,H,W,4] RGBA to [B,H,2W,3] RGB stitched."""
        left = rgb_t[:, 0, :, :, :3]   # [B,H,W,3]
        right = rgb_t[:, 1, :, :, :3]  # [B,H,W,3]
        B, H, W, C = left.shape
        obs_t = torch.empty((B, H, 2 * W, C), device=device, dtype=left.dtype)
        obs_t[:, :, :W, :].copy_(left)
        obs_t[:, :, W:, :].copy_(right)
        return obs_t

    def reset(self) -> TensorDict:
        # 1) Jitted, batched physics reset
        self.env_state = self.reset_fn(self.key_reset)

        # 2) Render (non-jit)
        rgba = self._render_rgba_batched()

        # 3) Export to Torch (GPU->GPU)
        rgb_t = _jax_to_torch(rgba)
        if rgb_t.device != self.device:
            rgb_t = rgb_t.to(self.device, non_blocking=True)

        obs_t = self._stereo_rgba_to_torch_obs(rgb_t, self.device)
        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        # Torch -> JAX
        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        # Jitted physics step
        self.env_state = self.step_fn(self.env_state, action_jax)

        # Render (non-jit)
        rgba = self._render_rgba_batched()

        # Export to Torch (GPU->GPU)
        rgb_t = _jax_to_torch(rgba)
        if rgb_t.device != self.device:
            rgb_t = rgb_t.to(self.device, non_blocking=True)

        obs_t = self._stereo_rgba_to_torch_obs(rgb_t, self.device)

        reward_t = _jax_to_torch(self.env_state.reward).to(self.device)
        done_t = _jax_to_torch(self.env_state.done).to(self.device)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}
        return obs_td, reward_t, done_t, info
