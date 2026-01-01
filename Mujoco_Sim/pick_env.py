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

from Custom_Mujoco_Playground._src import mjx_env


def adjust_brightness(pixels: jax.Array, brightness: jax.Array) -> jax.Array:
    return jp.clip(pixels * brightness, 0.0, 1.0)


class State:
    def __init__(self, data, obs, reward, done, metrics, info):
        self.data = data
        self.obs = obs
        self.reward = reward
        self.done = done
        self.metrics = metrics
        self.info = info


def _batched_body_xpos(data: mjx.Data, body_id: int) -> jax.Array:
    xpos = data.xpos
    if xpos.ndim == 3:
        return xpos[:, body_id, :]
    return xpos[body_id, :]


def _batched_body_xmat(data: mjx.Data, body_id: int) -> jax.Array:
    xmat = data.xmat
    if xmat.ndim == 3:
        return xmat[:, body_id, :]
    return xmat[body_id, :]


def _batched_geom_xpos(data: mjx.Data, geom_id: int) -> jax.Array:
    xpos = data.geom_xpos
    if xpos.ndim == 3:
        return xpos[:, geom_id, :]
    return xpos[geom_id, :]


def _batched_geom_xmat(data: mjx.Data, geom_id: int) -> jax.Array:
    xmat = data.geom_xmat
    if xmat.ndim == 3:
        return xmat[:, geom_id, :]
    return xmat[geom_id, :]


def _batched_site_xpos(data: mjx.Data, site_id: int) -> jax.Array:
    xpos = data.site_xpos
    if xpos.ndim == 3:
        return xpos[:, site_id, :]
    return xpos[site_id, :]


def _batched_site_xmat(data: mjx.Data, site_id: int) -> jax.Array:
    xmat = data.site_xmat
    if xmat.ndim == 3:
        return xmat[:, site_id, :]
    return xmat[site_id, :]


class StereoPickCube:
    def __init__(
        self,
        render_batch_size: int = 32,
        config: Optional[config_dict.ConfigDict] = None,
    ):
        self.render_batch_size = int(render_batch_size)
        self._config = config if config is not None else self._default_config()

        # Load MJ model (CPU) + MJX model (JAX)
        xml_path = Path(self._config.xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._mjx_model = mjx.put_model(self._mj_model)

        self.n_substeps = int(self._config.n_substeps)

        # IDs
        self._obj_body = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, self._config.object_body)
        self._obj_geom = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, self._config.object_geom)
        self._target_site = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, self._config.target_site)
        self._gripper_site = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, self._config.gripper_site)

        # Control ranges
        self._lowers = jp.array(self._mj_model.actuator_ctrlrange[:, 0], dtype=jp.float32)
        self._uppers = jp.array(self._mj_model.actuator_ctrlrange[:, 1], dtype=jp.float32)

        # Madrona renderer (non-jit)
        from madrona_mjx.renderer import BatchRenderer  # noqa: WPS433

        self.renderer = BatchRenderer(self._config.vision_config)
        self._render_fn = self.renderer.render  # (token, state, model)

        self._render_token: Optional[jax.Array] = None

        self.render_height = int(self._config.vision_config.height)
        self.render_width = int(self._config.vision_config.width)

    def _default_config(self) -> config_dict.ConfigDict:
        cfg = config_dict.ConfigDict()

        cfg.xml_path = "panda_pick_cube.xml"

        cfg.n_substeps = 20

        cfg.object_body = "box"
        cfg.object_geom = "box"
        cfg.target_site = "target"
        cfg.gripper_site = "gripper_site"

        cfg.action_scale = 0.02

        cfg.obs_noise = config_dict.ConfigDict()
        cfg.obs_noise.brightness = [0.85, 1.15]

        cfg.vision_config = config_dict.ConfigDict()
        cfg.vision_config.enabled_geom_groups = [0, 1, 2]
        cfg.vision_config.height = 64
        cfg.vision_config.width = 64

        cfg.reward_config = config_dict.ConfigDict()
        cfg.reward_config.scales = config_dict.ConfigDict()
        cfg.reward_config.scales.reach = 1.0
        cfg.reward_config.scales.lift = 1.0
        cfg.reward_config.scales.place = 1.0

        return cfg

    def _make_data_batched(self, B: int) -> mjx.Data:
        data0 = mjx.make_data(self._mjx_model)
        data = jax.tree_map(lambda x: jp.broadcast_to(x, (B,) + x.shape), data0)

        nq = int(self._mjx_model.nq)
        nv = int(self._mjx_model.nv)

        init_q0 = jp.asarray(self._mjx_model.qpos0, dtype=jp.float32)
        qpos = jp.broadcast_to(init_q0, (B, nq))
        qvel = jp.zeros((B, nv), dtype=jp.float32)

        ctrl0 = jp.zeros((int(self._mj_model.nu),), dtype=jp.float32)
        ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        return data

    # -------------------------
    # Rendering (always non-jit)
    # -------------------------

    def render_pixels(self, render_token: jax.Array, data_batched: mjx.Data) -> jax.Array:
        # Matches probed signature: render(token, state, model)
        _, rgb, _ = self._render_fn(render_token, data_batched)

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
        # CHANGED: do NOT read render_token from info; use persistent self._render_token
        pixels = self.render_pixels(self._render_token, data)
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
            print("[diag] render_token:", type(self._render_token),
                  "dtype:", getattr(self._render_token, "dtype", None),
                  "shape:", getattr(self._render_token, "shape", None))

            # One-time smoke render
            _, rgb, _ = self.renderer.render(self._render_token, data, self._mj_model)
            jax.block_until_ready(rgb)
            print("[diag] smoke render ok:", rgb.shape, rgb.dtype)

            canary = jp.zeros((1,), dtype=jp.float32)
            jax.block_until_ready(canary)
            print("[diag] post-render canary ok")

    def reset_physics(self, rng: jax.Array) -> State:
        if rng.ndim == 1:
            rng = rng[None, :]
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"

        dev = next(iter(rng.devices())) if hasattr(rng, "devices") else None
        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            m = self._mjx_model
            B = int(self.render_batch_size)

            rng_main = rng
            keys = jax.vmap(lambda k: jax.random.split(k, 4))(rng_main)
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]
            rng_brightness = keys[:, 3, :]

            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx)

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)

            data = self._make_data_batched(B)

            # Set object position into qpos via free joint (assumes object is first free joint).
            # Keep your original indexing/logic exactly:
            qpos = data.qpos
            if qpos.ndim == 2:
                qpos = qpos.at[:, 0:3].set(box_pos)
            else:
                qpos = qpos.at[0:3].set(box_pos[0])
            data = data.replace(qpos=qpos)

            # Mocap target
            target_quat0 = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
            target_quat = jp.broadcast_to(target_quat0, (B, 4))

            mpos = data.mocap_pos
            mquat = data.mocap_quat

            if mpos.ndim == 2:
                mpos = jp.broadcast_to(mpos, (B,) + mpos.shape)
            if mquat.ndim == 2:
                mquat = jp.broadcast_to(mquat, (B,) + mquat.shape)

            mpos = mpos.at[:, 0, :].set(target_pos)
            mquat = mquat.at[:, 0, :].set(target_quat)
            data = data.replace(mocap_pos=mpos, mocap_quat=mquat)

            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            bmin, bmax = self._config.obs_noise.brightness
            brightness = jax.vmap(
                lambda k: jax.random.uniform(k, (1,), minval=bmin, maxval=bmax)
            )(rng_brightness).reshape((B, 1, 1, 1))

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
                # CHANGED: do NOT store render_token in info
                "brightness": brightness,
            }

            obs = jp.zeros((int(B), 1), dtype=jp.float32)
            reward = jp.zeros((int(B),), dtype=jp.float32)
            done = jp.zeros((int(B),), dtype=jp.float32)

            return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, jax.Array]]:
        # --- original logic unchanged ---
        target_pos = info["target_pos"]
        gripper_pos = _batched_site_xpos(data, self._gripper_site)
        box_pos = _batched_body_xpos(data, self._obj_body)

        if gripper_pos.ndim == 2:
            reach_dist = jp.linalg.norm(gripper_pos - box_pos, axis=1)
        else:
            reach_dist = jp.linalg.norm(gripper_pos - box_pos)

        reach = -reach_dist

        if box_pos.ndim == 2:
            lift = box_pos[:, 2]
        else:
            lift = box_pos[2]

        if box_pos.ndim == 2:
            place_dist = jp.linalg.norm(box_pos - target_pos, axis=1)
        else:
            place_dist = jp.linalg.norm(box_pos - target_pos[0])

        place = -place_dist

        rewards = {
            "reach": reach,
            "lift": lift,
            "place": place,
        }
        return info, rewards

    def step_physics(self, state: State, action: jax.Array) -> State:
        action_scale = self._config.action_scale
        delta = action * action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = _batched_body_xpos(data, self._obj_body)
        if box_pos.ndim == 2:
            out_of_bounds = jp.any(jp.abs(box_pos) > 1.0, axis=1) | (box_pos[:, 2] < 0.0)
        else:
            out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)

        done = out_of_bounds.astype(jp.float32)

        metrics = state.metrics
        metrics["out_of_bounds"] = out_of_bounds.astype(jp.float32)

        return State(data, state.obs, total_reward, done, metrics, info)
