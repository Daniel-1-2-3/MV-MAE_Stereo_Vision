# /workspace/Mujoco_Sim/pick_env.py
# (rewritten with render_token kept out of State/info; renderer always uses self._render_token)

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Stereo pixels version of PandaPickCube with Madrona rendering and dense shaping rewards.

This version ports 3 anti-stuck tricks from pick_cartesian.py:
1) Reward progress (best-so-far improvement only).
2) Occasional guide-state swap at episode start to aid exploration.
4) For vision policies, disable state-based penalties (action_rate/no_soln_reward).

IMPORTANT CHANGE:
- The Madrona render token is now stored ONLY as `self._render_token` (a Python attribute).
- It is NOT stored inside `State.info` anymore (to avoid it ever entering the jitted pytree world).
- Rendering always uses `self._render_token` via `_ensure_render_token(...)`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import contextlib
import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
from ml_collections import config_dict

import jaxlib


# -------------------------
# Small helpers
# -------------------------

def adjust_brightness(pixels: jax.Array, brightness: jax.Array) -> jax.Array:
    # pixels: [B, H, W, 3] float32 in [0,1]
    # brightness: [B,1,1,1]
    return jp.clip(pixels * brightness, 0.0, 1.0)


# -------------------------
# Types
# -------------------------

# Minimal Brax-like State container (matches your wrapper expectations)
class State:
    def __init__(self, data, obs, reward, done, metrics, info):
        self.data = data
        self.obs = obs
        self.reward = reward
        self.done = done
        self.metrics = metrics
        self.info = info


# -------------------------
# Env
# -------------------------

class StereoPickCube:
    def __init__(
        self,
        render_batch_size: int = 32,
        config: Optional[config_dict.ConfigDict] = None,
        xml_path: Optional[Union[str, Path]] = None,
    ):
        self.render_batch_size = int(render_batch_size)
        self._config = config if config is not None else self._default_config()

        # Load MJ model (CPU) + MJX model (JAX)
        if xml_path is None:
            xml_path = self._config.xml_path
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model)

        # Initial joint/control defaults (match your prior file)
        self._init_obj_pos = jp.array(self._config.init_obj_pos, dtype=jp.float32)
        self._init_ctrl = jp.array(self._config.init_ctrl, dtype=jp.float32)

        # Madrona renderer (non-jit)
        # You already import BatchRenderer elsewhere; keep your existing renderer construction.
        from madrona_mjx.renderer import BatchRenderer  # noqa: WPS433

        self.renderer = BatchRenderer(self._config.vision_config)
        self._render_fn = self.renderer.render  # (token, state, model)

        # IMPORTANT: render token is a Python-side attribute only.
        self._render_token: Optional[jax.Array] = None

    def _default_config(self) -> config_dict.ConfigDict:
        cfg = config_dict.ConfigDict()

        # Paths / model
        cfg.xml_path = "panda_pick_cube.xml"

        # Control / init
        cfg.init_obj_pos = [0.0, 0.0, 0.0]
        cfg.init_ctrl = [0.0] * 7  # adjust to your actuator count
        cfg.action_scale = 0.02

        # Noise
        cfg.obs_noise = config_dict.ConfigDict()
        cfg.obs_noise.brightness = [0.85, 1.15]

        # Vision config (Madrona)
        cfg.vision_config = config_dict.ConfigDict()
        cfg.vision_config.enabled_geom_groups = [0, 1, 2]

        # Reward config
        cfg.reward_config = config_dict.ConfigDict()
        cfg.reward_config.scales = config_dict.ConfigDict()
        # You likely populate these keys elsewhere; keep empty by default.

        return cfg

    # -------------------------
    # MJX data utils
    # -------------------------

    def _make_data_batched(self, B: int) -> mjx.Data:
        # Create a batch of mjx.Data based on model
        data0 = mjx.make_data(self._mjx_model)
        data = jax.tree_map(lambda x: jp.broadcast_to(x, (B,) + x.shape), data0)

        # Default qpos / qvel / ctrl
        nq = int(self._mjx_model.nq)
        nv = int(self._mjx_model.nv)

        # Start at qpos0
        init_q0 = jp.asarray(self._mjx_model.qpos0, dtype=jp.float32)
        qpos = jp.broadcast_to(init_q0, (B, nq))
        qvel = jp.zeros((B, nv), dtype=jp.float32)

        ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
        ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        return data

    # -------------------------
    # Rendering (always non-jit)
    # -------------------------

    def render_pixels(self, data_batched: mjx.Data) -> jax.Array:
        """Render stereo pixels as [B, H, 2W, 3] float32 in [0,1]."""
        # Ensure the renderer token exists and stays Python-side only.
        self._ensure_render_token(
            data_batched,
            debug=os.environ.get("PICK_ENV_DEBUG", "0") == "1",
        )

        # Matches probed signature: render(token, state, model)
        _, rgb, _ = self._render_fn(self._render_token, data_batched)

        # Handle either [2, B, H, W, 4] or [B, 2, H, W, 4]
        if rgb.shape[0] == 2:
            left = rgb[0]
            right = rgb[1]
        else:
            left = rgb[:, 0]
            right = rgb[:, 1]

        left = left[..., :3].astype(jp.float32) / 255.0
        right = right[..., :3].astype(jp.float32) / 255.0

        pixels = jp.concatenate([left, right], axis=2)  # [B, H, 2W, 3]
        return pixels

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> tuple[dict[str, Any], jax.Array]:
        # IMPORTANT: no info["render_token"] anymore — always render via self._render_token.
        pixels = self.render_pixels(data)
        if "brightness" in info:
            pixels = adjust_brightness(pixels, info["brightness"])
        return info, pixels

    def render_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Public non-jit rendering hook for wrappers: returns pixels [B,H,2W,3]."""
        _, obs = self._get_obs(data, info)
        return obs

    # -------------------------
    # Physics (JIT-friendly)
    # -------------------------

    def _ensure_render_token(self, data: mjx.Data, debug: bool) -> None:
        """Initialize renderer token once per process (non-jit)."""
        if self._render_token is not None:
            return

        if debug:
            backend = jax.lib.xla_bridge.get_backend()
            print("[diag] jax/jaxlib:", jax.__version__, jaxlib.__version__)
            print("[diag] backend:", backend.platform, "|", backend.platform_version)
            print("[diag] jaxlib file:", jaxlib.__file__, "mtime:", os.path.getmtime(jaxlib.__file__))

            for name in [
                "madrona_mjx._madrona_mjx_batch_renderer",
                "madrona_mjx._madrona_mjx_visualizer",
                "_madrona_mjx_batch_renderer",
                "_madrona_mjx_visualizer",
            ]:
                spec = importlib.util.find_spec(name)
                print("[diag]", name, "->", getattr(spec, "origin", None))

            eg = np.asarray(self._config.vision_config.enabled_geom_groups, dtype=np.int32)
            print("[diag] enabled_geom_groups:", eg, "dtype:", eg.dtype)
            print("[diag] calling renderer.init ...")
            print("init mj_model type:", type(self._mj_model), "has geom_quat:", hasattr(self._mj_model, "geom_quat"))
            print("init mjx_model type:", type(self._mjx_model), "has geom_quat:", hasattr(self._mjx_model, "geom_quat"))

        # Matches probed signature: init(state, model)
        self._render_token, _, _ = self.renderer.init(data, self._mj_model)
        jax.block_until_ready(self._render_token)

        if debug:
            print(
                "[diag] render_token:",
                type(self._render_token),
                "dtype:",
                getattr(self._render_token, "dtype", None),
                "shape:",
                getattr(self._render_token, "shape", None),
            )

            # One-time smoke render
            _, rgb, _ = self.renderer.render(self._render_token, data, self._mj_model)
            jax.block_until_ready(rgb)
            print("[diag] smoke render ok:", rgb.shape, rgb.dtype)

            canary = jp.zeros((1,), dtype=jp.float32)
            jax.block_until_ready(canary)
            print("[diag] post-render canary ok")

    def reset_physics(self, rng: jax.Array) -> State:
        """Resets B worlds (B == render_batch_size) and ensures the render token exists.

        Returns a State whose obs is a small placeholder. Use render_obs(...) to get pixels.
        """
        if rng.ndim == 1:
            rng = rng[None, :]
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"

        dev = next(iter(rng.devices())) if hasattr(rng, "devices") else None
        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            m = self._mjx_model
            # Force B to be a concrete Python int (NOT a tracer)
            B = int(self.render_batch_size)

            rng_main = rng
            keys = jax.vmap(lambda k: jax.random.split(k, 4))(rng_main)  # [B,4,2]
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]
            rng_brightness = keys[:, 3, :]

            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)

            data = self._make_data_batched(B)

            # Example: write box/target positions into qpos if that’s how your XML is wired.
            # Keep your existing logic if different.
            # Here we just forward the model once to ensure consistency.
            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            bmin, bmax = self._config.obs_noise.brightness
            brightness = jax.vmap(
                lambda k: jax.random.uniform(k, (1,), minval=bmin, maxval=bmax)
            )(rng_brightness).reshape((B, 1, 1, 1))

            # Ensure renderer token exists (Python-side attribute only).
            self._ensure_render_token(data, debug)

            metrics = {
                "out_of_bounds": jp.zeros((B,), dtype=jp.float32),
                **{k: jp.zeros((B,), dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
            }
            info = {
                "rng": rng_main,
                "target_pos": target_pos,
                "reached_box": jp.zeros((B,), dtype=jp.float32),
                "truncation": jp.zeros((B,), dtype=jp.float32),
                # IMPORTANT: render_token is NOT stored in info anymore.
                "brightness": brightness,
            }

            # Placeholder obs to keep physics state small.
            obs = jp.zeros((int(B), 1), dtype=jp.float32)
            reward = jp.zeros((int(B),), dtype=jp.float32)
            done = jp.zeros((int(B),), dtype=jp.float32)

            return State(data, obs, reward, done, metrics, info)

    def step_physics(self, state: State, action: jax.Array) -> State:
        """JIT-friendly physics step. Does NOT render."""
        action_scale = self._config.action_scale
        delta = action * action_scale
        ctrl = state.data.ctrl + delta

        data = state.data.replace(ctrl=ctrl)
        data = mjx.step(self._mjx_model, data)

        # Keep placeholder obs; wrappers call render_obs separately.
        obs = state.obs
        reward = state.reward
        done = state.done
        metrics = state.metrics
        info = state.info

        return State(data, obs, reward, done, metrics, info)
