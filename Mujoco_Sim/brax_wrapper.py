import os
from typing import Optional, Tuple

import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    # GPU->GPU zero-copy via DLPack when possible.
    return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    # GPU->GPU via DLPack. Using jax.dlpack.to_dlpack avoids relying on __dlpack__ dispatch.
    return tpack.from_dlpack(jax.dlpack.to_dlpack(x))


class RSLRLBraxWrapper:
    """Batched MJX env wrapper for a Torch RL loop (DrQv2).

    Key rules for stability + speed:
    - Physics is jitted + vmapped (parallel worlds).
    - Madrona renderer init/render is **never** called under vmap/jit.
      Rendering happens from Python on batched `mjx.Data`.
    - All obs/action transfer uses DLPack to avoid CPU copies.
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

        # Torch device for the RL learner tensors
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # Safety: Madrona renderer must be created with num_worlds == batch_size
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"StereoPickCube(render_batch_size=...) must equal num_envs. Got render_batch_size={rbs}, num_envs={self.batch_size}."
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
                def body(i, st):
                    return self._step_physics_batched(st, action)
                return jax.lax.fori_loop(0, self.action_repeat, body, state)

            self.step_fn = jax.jit(_step_repeat)

        self.env_state = None  # set in reset()

    def reset(self) -> TensorDict:
        # Jitted, batched physics reset
        self.env_state = self.reset_fn(self.key_reset)

        # Non-jit renderer init + render
        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        self._raw_env._ensure_render_token(self.env_state.data, debug)        # Render raw RGBA (uint8)
        rgb = self._raw_env.render_rgba(self.env_state.data)

        # Important: surface any renderer-side CUDA errors here (before Torch touches it).
        jax.block_until_ready(rgb)

        # Convert to Torch (GPU->GPU) and make a Torch-owned contiguous copy to avoid
        # any DLPack/stride quirks with subsequent CUDA kernels.
        rgb_t = _jax_to_torch(rgb)
        if rgb_t.device != self.device:
            rgb_t = rgb_t.to(self.device, non_blocking=True)
        rgb_t = rgb_t.contiguous()

        # Fuse stereo views without torch.cat (which can be picky on some uint8/stride cases)
        left = rgb_t[:, 0, :, :, :3].contiguous()   # [B,H,W,3]
        right = rgb_t[:, 1, :, :, :3].contiguous()  # [B,H,W,3]
        B, H, W, C = left.shape
        obs_t = torch.empty((B, H, 2 * W, C), device=self.device, dtype=left.dtype)
        obs_t[:, :, :W, :].copy_(left)
        obs_t[:, :, W:, :].copy_(right)

        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        # Torch -> JAX (GPU->GPU), physics step (jitted), then render (non-jit)
        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        self._raw_env._ensure_render_token(self.env_state.data, debug)        # Render raw RGBA (uint8)
        rgb = self._raw_env.render_rgba(self.env_state.data)

        # Surface any renderer-side CUDA errors here.
        jax.block_until_ready(rgb)

        rgb_t = _jax_to_torch(rgb)
        if rgb_t.device != self.device:
            rgb_t = rgb_t.to(self.device, non_blocking=True)
        rgb_t = rgb_t.contiguous()

        left = rgb_t[:, 0, :, :, :3].contiguous()
        right = rgb_t[:, 1, :, :, :3].contiguous()
        B, H, W, C = left.shape
        obs_t = torch.empty((B, H, 2 * W, C), device=self.device, dtype=left.dtype)
        obs_t[:, :, :W, :].copy_(left)
        obs_t[:, :, W:, :].copy_(right)
        reward_t = _jax_to_torch(self.env_state.reward).to(self.device)
        done_t = _jax_to_torch(self.env_state.done).to(self.device)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

        # Keep info minimal to avoid host sync every step
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}
        return obs_td, reward_t, done_t, info