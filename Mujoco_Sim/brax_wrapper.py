# Mujoco_Sim/brax_wrapper.py
import os
from typing import Optional, Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack

from madrona_mjx.wrapper import _identity_randomization_fn


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    # actions: zero-copy is fine (not a renderer output)
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch_hostcopy(x: jax.Array, device: torch.device) -> torch.Tensor:
    """Break any GPU pointer sharing: JAX -> host numpy -> Torch tensor -> (optional) GPU."""
    arr = np.asarray(jax.device_get(x))
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        t = t.to(device, non_blocking=True)
    return t


class RSLRLBraxWrapper:
    """Torch-facing vector-env wrapper using debug.py's EXACT renderer calling pattern.

    Matches debug.py in these ways:
      - wrapper (not env) constructs BatchRenderer
      - init path: @jit init(rng, model) containing vmap(init_)
      - init_ makes fresh data via mjx.make_data + mjx.forward, then renderer.init(data, model)
      - render path: compile `step = jax.jit(step).lower(v_mjx_data).compile()`
      - step uses `renderer.render(render_token, data)` and is vmapped over data
      - NO extra jax.block_until_ready / healthcheck kernels inside render path
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
        self.render_callback = render_callback

        # Torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # Sanity: renderer num_worlds must match (kept)
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"StereoPickCube(render_batch_size) must equal num_envs. "
                    f"Got render_batch_size={rbs}, num_envs={self.batch_size}."
                )

        # RNG keys per env (physics)
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

        # Renderer state (debug-exact)
        self.env_state = None
        self._renderer = None
        self._render_token = None
        self._render_step_compiled = None  # compiled `step` from debug.py
        self._v_mjx_data_for_compile = None

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
    
    def _build_renderer_exact_debug(self):
        """Renderer build that avoids tracing renderer.init into a big JIT compile,
        and compiles render() with the token as an explicit argument (more cacheable)."""
        assert self.env_state is not None, "Call reset() once before building renderer."

        B = self.batch_size
        mjx_model = self._raw_env.mjx_model

        print("[renderer] building renderer + compiling render step ONCE")
        self._renderer = self._raw_env.make_renderer_debug(B)

        # Matches debug intent: build batched model/in_axes
        v_mjx_model, v_in_axes = _identity_randomization_fn(mjx_model, B)
        print("[renderer] v_in_axes:", v_in_axes)

        renderer = self._renderer

        # --- 1) Pure MJX forward is what we JIT (fast + cacheable) ---
        @jax.jit
        def make_forward(model):
            data = mjx.make_data(model)
            return mjx.forward(model, data)

        v_make_forward = jax.vmap(make_forward, in_axes=v_in_axes)
        v_mjx_data = v_make_forward(v_mjx_model)

        # --- 2) Renderer init is NOT inside jitted code (reduces giant compile units) ---
        def init_render_one(d, m):
            return renderer.init(d, m)

        v_init_render = jax.vmap(init_render_one, in_axes=(0, v_in_axes))
        render_token, _rgb0, _depth0 = v_init_render(v_mjx_data, v_mjx_model)

        self._render_token = render_token
        self._v_mjx_data_for_compile = v_mjx_data

        # --- 3) Compile render step with token as an explicit argument (more stable caching) ---
        def step(token, data):
            def step_(d):
                _, rgb, depth = renderer.render(token, d)
                return d, rgb, depth
            return jax.vmap(step_)(data)

        # compile once
        self._render_step_compiled = jax.jit(step).lower(self._render_token, v_mjx_data).compile()
        print("[renderer] render step compiled")

    def _render_rgba_batched(self) -> jax.Array:
        """Render using debug.py compiled step; returns JAX uint8 [B,2,H,W,4]."""
        assert self.env_state is not None, "env_state is None"

        if self._render_step_compiled is None:
            self._build_renderer_exact_debug()

        # Render using current physics data (batched mjx.Data)
        _data_out, rgba, _depth = self._render_step_compiled(self._render_token, self.env_state.data)
        return rgba

    def reset(self) -> TensorDict:
        self.env_state = self.reset_fn(self.key_reset)

        rgba = self._render_rgba_batched()

        # To keep renderer/data semantics identical to debug.py (no Torch touching JAX buffers),
        # we break sharing via host copy here.
        rgba_t = _jax_to_torch_hostcopy(rgba, self.device)

        obs_t = self._stereo_rgba_to_torch_obs(rgba_t, self.device)
        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        rgba = self._render_rgba_batched()
        rgba_t = _jax_to_torch_hostcopy(rgba, self.device)
        obs_t = self._stereo_rgba_to_torch_obs(rgba_t, self.device)

        # Safe (no-sharing) reward/done transfers too
        reward_t = _jax_to_torch_hostcopy(self.env_state.reward, self.device).reshape(-1).to(torch.float32)
        done_t = _jax_to_torch_hostcopy(self.env_state.done, self.device).reshape(-1).to(torch.bool)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self.render_callback is not None:
            self.render_callback(None, self.env_state)

        return obs_td, reward_t, done_t, info
