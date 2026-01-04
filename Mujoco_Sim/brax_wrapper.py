# Mujoco_Sim/brax_wrapper.py
import os
from typing import Optional, Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack

from madrona_mjx.wrapper import _identity_randomization_fn


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    return tpack.from_dlpack(jax.dlpack.to_dlpack(x))


class RSLRLBraxWrapper:
    """Torch-facing vector-env wrapper using debug.py's SAFE renderer calling pattern."""

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
        self.render_callback = render_callback

        # Torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # Sanity: renderer num_worlds must match
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"StereoPickCube(render_batch_size) must equal num_envs. "
                    f"Got render_batch_size={rbs}, num_envs={self.batch_size}."
                )

        # RNG keys per env
        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        # Batched physics
        self._reset_batched = jax.vmap(self._raw_env.reset)
        self._step_batched = jax.vmap(self._raw_env.step, in_axes=(0, 0), out_axes=0)

        self.reset_fn = jax.jit(self._reset_batched)

        if self.action_repeat == 1:
            self.step_fn = jax.jit(self._step_batched)
        else:
            def _step_repeat(state, action):
                def body(_, st):
                    return self._step_batched(st, action)
                return jax.lax.fori_loop(0, self.action_repeat, body, state)
            self.step_fn = jax.jit(_step_repeat)

        # Renderer state (debug-style)
        self.env_state = None
        self._render_token = None
        self._render_compiled = None  # compiled function that renders given batched mjx.Data

        # Optional: detect poisoning early (turn on only for debugging)
        self._render_healthcheck = os.environ.get("MADRONA_RENDER_HEALTHCHECK", "0") == "1"

    @staticmethod
    def _stereo_rgba_to_torch_obs(rgba_t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """[B,2,H,W,4] uint8 -> [B,H,2W,3] uint8 (Torch-only)."""
        left = rgba_t[:, 0, :, :, :3]
        right = rgba_t[:, 1, :, :, :3]
        B, H, W, C = left.shape
        out = torch.empty((B, H, 2 * W, C), device=device, dtype=left.dtype)
        out[:, :, :W, :].copy_(left)
        out[:, :, W:, :].copy_(right)
        return out

    def _build_renderer_like_debug(self):
        """Create token via jit+vmap init, then compile render step once (matches debug.py)."""
        assert self.env_state is not None, "Call reset() once before building renderer."

        renderer = self._raw_env.renderer
        mjx_model = self._raw_env.mjx_model
        B = self.batch_size

        # EXACTLY like debug.py: get v_mjx_model + v_in_axes from helper
        v_mjx_model, v_in_axes = _identity_randomization_fn(mjx_model, B)

        @jax.jit
        def init(keys, model):
            def init_(rng, model):
                # debug.py doesn't actually use rng; keep signature the same anyway
                data = mjx.make_data(model)
                data = mjx.forward(model, data)
                render_token, rgb, depth = renderer.init(data, model)
                return data, render_token, rgb, depth

            return jax.vmap(init_, in_axes=[0, v_in_axes])(keys, model)

        rng = jax.random.PRNGKey(2)
        rng, *keys = jax.random.split(rng, B + 1)
        v_mjx_data, render_token, _rgb0, _depth0 = init(jp.asarray(keys), v_mjx_model)

        # Save the token (debug passes the same token forever; does NOT update it)
        self._render_token = render_token

        # Now compile a render step ONCE, with token closed over (like debug)
        def render_step(data_batched):
            def step_(data_one):
                _tok2, rgb, depth = renderer.render(self._render_token, data_one)
                return data_one, rgb, depth
            return jax.vmap(step_)(data_batched)

        # Compile using a representative mjx.Data structure of correct shapes.
        # Use v_mjx_data from the same init path, just like debug.py.
        self._render_compiled = jax.jit(render_step).lower(v_mjx_data).compile()

        # Warm-up call (optional, but keeps behavior predictable)
        _data_out, rgb, _depth = self._render_compiled(v_mjx_data)
        jax.block_until_ready(rgb)

    def _render_rgba_batched(self) -> jax.Array:
        """Render using compiled debug-style path; returns JAX uint8 [B,2,H,W,4]."""
        assert self.env_state is not None, "env_state is None"

        if self._render_compiled is None:
            self._build_renderer_like_debug()

        # Render using current physics data
        _data_out, rgba, _depth = self._render_compiled(self.env_state.data)
        jax.block_until_ready(rgba)

        if self._render_healthcheck:
            # tiny kernel after render to detect poisoning early
            x = jp.asarray(1, dtype=jp.int32) + jp.asarray(1, dtype=jp.int32)
            jax.block_until_ready(x)

        return rgba

    def reset(self) -> TensorDict:
        self.env_state = self.reset_fn(self.key_reset)

        # Renderer init/compile happens lazily on first render
        rgba = self._render_rgba_batched()

        rgba_t = _jax_to_torch(rgba)
        if rgba_t.device != self.device:
            rgba_t = rgba_t.to(self.device, non_blocking=True)

        obs_t = self._stereo_rgba_to_torch_obs(rgba_t, self.device)
        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        rgba = self._render_rgba_batched()

        rgba_t = _jax_to_torch(rgba)
        if rgba_t.device != self.device:
            rgba_t = rgba_t.to(self.device, non_blocking=True)

        obs_t = self._stereo_rgba_to_torch_obs(rgba_t, self.device)

        reward_t = _jax_to_torch(self.env_state.reward).to(self.device)
        done_t = _jax_to_torch(self.env_state.done).to(self.device)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self.render_callback is not None:
            # preserve your hook shape: (unused, state)
            self.render_callback(None, self.env_state)

        return obs_td, reward_t, done_t, info
