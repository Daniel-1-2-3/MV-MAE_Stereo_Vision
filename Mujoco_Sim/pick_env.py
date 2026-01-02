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

"""Bring a box to a target and orientation.

This version mirrors the *working* approach you showed:

- **Env physics is single-world** (reset_physics / step_physics take one world).
- **Batching happens outside the env** via `jax.vmap` in the wrapper.
- **Renderer is never traced** into the physics JIT: rendering happens from Python
  using the exact `renderer.init` / `renderer.render` call pattern.

Public reset/step still exist for convenience; they render by temporarily broadcasting
a single world to `render_batch_size` and returning only the first image (slow path).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import contextlib
import importlib.util
import os
from pathlib import Path

# Reduce noisy XLA logging (includes the "backend config ... non UTF-8" warnings).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import jax
import jax.numpy as jp
import jaxlib
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from Custom_Mujoco_Playground._src import mjx_env
from Custom_Mujoco_Playground._src.mjx_env import State
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda

# Madrona MJX renderer
from madrona_mjx.renderer import BatchRenderer  # type: ignore


def _add_assets(assets: dict[str, bytes], root: Path) -> dict[str, bytes]:
    used_basenames = {Path(k).name for k in assets.keys()}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        base = p.name
        if base in used_basenames:
            continue
        rel_key = p.relative_to(root).as_posix()
        assets[rel_key] = p.read_bytes()
        used_basenames.add(base)

    return assets


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=1024,
        render_width=64,
        render_height=64,
        use_rasterizer=False,
        enabled_geom_groups=[0, 1, 2],
    )


def default_config():
    config = config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                gripper_box=4.0,
                box_target=8.0,
                no_floor_collision=0.25,
                no_box_collision=0.05,
                robot_target_qpos=0.0,
            ),
            action_rate=-0.0005,
            no_soln_reward=-0.01,
            lifted_reward=0.5,
            success_reward=2.0,
        ),
        vision=True,
        vision_config=default_vision_config(),
        obs_noise=config_dict.create(brightness=[1.0, 1.0]),
        impl="jax",
        nconmax=24 * 2048,
        njmax=128,
    )
    return config


def adjust_brightness(img: jax.Array, scale: jax.Array) -> jax.Array:
    """Scales pixel values by per-world brightness and clamps to [0, 1]."""
    return jp.clip(img * scale, 0.0, 1.0)


def _batched_body_xpos(data: mjx.Data, body_id: int) -> jax.Array:
    """Returns body position as [B,3] if batched, else [3]."""
    xpos = data.xpos
    if hasattr(xpos, "ndim") and xpos.ndim == 3:
        return xpos[:, body_id, :]
    return xpos[body_id]


def _any_nan(x: jax.Array) -> jax.Array:
    if x.ndim <= 1:
        return jp.any(jp.isnan(x))
    axes = tuple(range(1, x.ndim))
    return jp.any(jp.isnan(x), axis=axes)


def _broadcast_tree_to_batch(tree, B: int):
    """Broadcast a pytree of arrays/scalars to a leading batch dim B."""
    def _bcast(x):
        if not hasattr(x, "ndim"):
            return x
        if x.ndim == 0:
            return jp.broadcast_to(x, (B,))
        return jp.broadcast_to(x, (B,) + x.shape)
    return jax.tree_util.tree_map(_bcast, tree)


def _batch_model_for_vmap(model, B: int):
    """Create a per-world batched MJX model + matching in_axes pytree for vmap.

    Important: we ONLY batch array leaves with ndim>=1. Scalars / 0d stay static (in_axes=None).
    """
    def _bcast(x):
        if hasattr(x, "ndim") and x.ndim >= 1:
            return jp.broadcast_to(x, (B,) + x.shape)
        return x

    def _in_axes(x):
        if hasattr(x, "ndim") and x.ndim >= 1:
            return 0
        return None

    model_b = jax.tree_util.tree_map(_bcast, model)
    in_axes = jax.tree_util.tree_map(_in_axes, model)
    return model_b, in_axes


class StereoPickCube(panda.PandaBase):
    """Bring a box to a target."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        render_batch_size: int = 128,
        render_width: int = 64,
        render_height: int = 64,
    ):
        # ---- Render params (num_worlds for Madrona) ----
        self.render_batch_size = int(render_batch_size)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        # ---- Base MJX env init (DO NOT call super()) ----
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        # ---- XML path ----
        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )
        self._xml_path = xml_path.as_posix()

        # ---- Assets: start from panda.get_assets(), then merge Menagerie dir ----
        self._model_assets = dict(panda.get_assets())
        menagerie_dir = (
            Path.cwd()
            / "mujoco_playground_external_deps"
            / "mujoco_menagerie"
            / "franka_emika_panda"
        )
        self._model_assets = _add_assets(self._model_assets, menagerie_dir)

        # ---- Build mujoco model using merged assets, apply modify_model ----
        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(
                xml_path.read_text(),
                assets=self._model_assets,
            )
        )
        mj_model.opt.timestep = self._config.sim_dt

        # ---- Store MuJoCo and MJX models ----
        self._mj_model: mujoco.MjModel = mj_model
        self._mjx_model: mjx.Model = mjx.put_model(mj_model, impl=self._config.impl)

        # ---- Task post init ----
        self._post_init(obj_name="box", keyframe="low_home")

        # ---- Floor collision geom ids ----
        self._floor_hand_geom_ids = [
            self._mj_model.geom(geom).id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]
        self._floor_geom_id = self._mj_model.geom("floor").id

        # ---- Renderer: create now, init token lazily (outside jit) ----
        self.renderer: BatchRenderer = self._create_renderer()
        self._render_token: Optional[jax.Array] = None

        # ---- Renderer vmap wrappers (LAZY, created on first init) ----
        self._v_mjx_model = None
        self._v_mjx_in_axes = None
        self._init_vmap = None
        self._render_vmap = None

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        # PandaBase post-init can mutate MuJoCo model; re-upload to MJX.
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._sample_orientation = False

    def _create_renderer(self) -> BatchRenderer:
        enabled_geom_groups = np.asarray(
            self._config.vision_config.enabled_geom_groups, dtype=np.int32
        )
        return BatchRenderer(
            m=self._mjx_model,
            gpu_id=int(self._config.vision_config.gpu_id),
            num_worlds=int(self.render_batch_size),
            batch_render_view_width=int(self.render_width),
            batch_render_view_height=int(self.render_height),
            enabled_geom_groups=enabled_geom_groups,
            enabled_cameras=None,
            add_cam_debug_geo=False,
            use_rasterizer=bool(self._config.vision_config.use_rasterizer),
            viz_gpu_hdls=None,
        )

    # -------------------------
    # Rendering (always non-jit)
    # -------------------------

    def render_pixels(self, render_token: jax.Array, data_batched: mjx.Data) -> jax.Array:
        # Ensure wrappers exist (they are created inside _ensure_render_token)
        if self._render_vmap is None or self._v_mjx_model is None:
            # This should normally be called by reset/step before rendering
            self._ensure_render_token(data_batched, debug=False)

        # Correct: vmap render over (token_i, data_i, model_i)
        new_token, rgb, _depth = self._render_vmap(render_token, data_batched, self._v_mjx_model)

        # Thread token for side-effect sequencing
        self._render_token = new_token

        # Expect rgb as [B, num_cams, H, W, 4] (or occasionally [num_cams, B, ...])
        B = int(getattr(data_batched.geom_xpos, "shape", [0])[0])

        # Handle camera/world axis swap defensively
        if rgb.ndim == 5 and rgb.shape[0] == 2 and rgb.shape[1] == B:
            # [2, B, H, W, 4]
            left = rgb[0]
            right = rgb[1]
        else:
            # [B, 2, H, W, 4] expected
            left = rgb[:, 0]
            right = rgb[:, 1]

        left = left[..., :3].astype(jp.float32) / 255.0
        right = right[..., :3].astype(jp.float32) / 255.0

        pixels = jp.concatenate([left, right], axis=2)  # [B, H, 2W, 3]
        return pixels

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> tuple[dict[str, Any], jax.Array]:
        pixels = self.render_pixels(self._render_token, data)
        if "brightness" in info:
            pixels = adjust_brightness(pixels, info["brightness"])
        return info, pixels

    def render_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Public non-jit rendering hook: returns pixels [B,H,2W,3]."""
        _, obs = self._get_obs(data, info)
        return obs

    def _ensure_render_token(self, data_batched: mjx.Data, debug: bool) -> None:
        """Initialize renderer token once per process (non-jit)."""
        if self._render_token is not None and self._init_vmap is not None and self._render_vmap is not None:
            return

        # Build per-world model + vmap wrappers once
        if self._init_vmap is None or self._render_vmap is None or self._v_mjx_model is None:
            B = int(self.render_batch_size)
            self._v_mjx_model, self._v_mjx_in_axes = _batch_model_for_vmap(self._mjx_model, B)

            # vmap over worlds: init(state_i, model_i) and render(token_i, state_i, model_i)
            self._init_vmap = jax.vmap(
                lambda d, m: self.renderer.init(d, m),
                in_axes=(0, self._v_mjx_in_axes),
            )
            self._render_vmap = jax.vmap(
                lambda t, d, m: self.renderer.render(t, d, m),
                in_axes=(0, 0, self._v_mjx_in_axes),
            )

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
            print("[diag] calling renderer.init(vmap) ...")
            print("init mj_model type:", type(self._mj_model), "has geom_quat:", hasattr(self._mj_model, "geom_quat"))
            print("init mjx_model type:", type(self._mjx_model), "has geom_quat:", hasattr(self._mjx_model, "geom_quat"))
            print("[diag] data.geom_xpos shape:", getattr(data_batched.geom_xpos, "shape", None))

        # Correct: vmap init across worlds, and pass MJX model (batched)
        render_token, _init_rgb, _init_depth = self._init_vmap(data_batched, self._v_mjx_model)
        self._render_token = render_token
        jax.block_until_ready(self._render_token)

        if debug:
            # One-time smoke render (also vmapped). Thread token properly.
            new_token, rgb, _ = self._render_vmap(self._render_token, data_batched, self._v_mjx_model)
            self._render_token = new_token
            jax.block_until_ready(rgb)
            print("[diag] smoke render ok:", rgb.shape, rgb.dtype)

    # -------------------------
    # Physics (single-world, JIT-friendly)
    # -------------------------

    def reset_physics(self, rng: jax.Array) -> State:
        """Single-world reset. Intended to be vmapped externally for batch."""
        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        m = self._mjx_model
        rng, rng_box, rng_target, rng_brightness = jax.random.split(rng, 4)

        min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
        max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
        min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
        max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
        base = jp.asarray(self._init_obj_pos, dtype=jp.float32)

        def _sample_pos(key, mn, mx):
            return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

        box_pos = _sample_pos(rng_box, min_box, max_box)
        target_pos = _sample_pos(rng_target, min_tgt, max_tgt)

        data = mjx.make_data(m)

        # Initialize to home pose (same logic as before)
        nq = int(m.nq)
        nv = int(m.nv)
        qpos0 = jp.asarray(self._init_q, dtype=jp.float32)[..., :nq]
        qvel0 = jp.zeros((nv,), dtype=jp.float32)
        ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
        data = data.replace(qpos=qpos0, qvel=qvel0, ctrl=ctrl0)

        # Place box
        data = data.replace(
            qpos=data.qpos.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        )

        # Set mocap target (camera / target)
        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        mpos = data.mocap_pos
        mquat = data.mocap_quat
        mpos = mpos.at[0, :].set(target_pos)
        mquat = mquat.at[0, :].set(target_quat)
        data = data.replace(mocap_pos=mpos, mocap_quat=mquat)

        data = mjx.forward(m, data)

        bmin, bmax = self._config.obs_noise.brightness
        brightness = jax.random.uniform(rng_brightness, (1,), minval=bmin, maxval=bmax).reshape((1, 1, 1))

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=jp.float32),
            **{k: jp.array(0.0, dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=jp.float32),
            "truncation": jp.array(0.0, dtype=jp.float32),
            "brightness": brightness,
        }

        # Placeholder obs to keep physics state small.
        obs = jp.zeros((1,), dtype=jp.float32)
        reward = jp.array(0.0, dtype=jp.float32)
        done = jp.array(0.0, dtype=jp.float32)

        return State(data, obs, reward, done, metrics, info)

    def step_physics(self, state: State, action: jax.Array) -> State:
        """Single-world physics step. Intended to be vmapped externally for batch."""
        action_scale = self._config.action_scale
        delta = action * action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = _batched_body_xpos(data, self._obj_body)  # [3] in single-world
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)

        new_metrics = dict(state.metrics)
        new_metrics["out_of_bounds"] = out_of_bounds.astype(jp.float32)
        for k, v in raw_rewards.items():
            new_metrics[k] = v

        done = out_of_bounds | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        # Keep obs small in physics state.
        obs = state.obs

        return state.replace(
            data=data,
            obs=obs,
            reward=total_reward,
            done=done.astype(jp.float32),
            metrics=new_metrics,
            info=info,
        )

    # -------------------------
    # Public API (slow path, keeps compatibility)
    # -------------------------

    def reset(self, rng: jax.Array) -> State:
        """Public reset: single-world physics + broadcasted render -> pixels in obs (slow)."""
        st1 = self.reset_physics(rng)

        # Broadcast a single world to num_worlds for renderer, then return only world 0 image.
        B = self.render_batch_size
        data_b = _broadcast_tree_to_batch(st1.data, B)
        info_b = _broadcast_tree_to_batch(st1.info, B)
        # brightness should be [B,1,1,1] for adjust_brightness
        if "brightness" in info_b:
            b = info_b["brightness"]
            if b.ndim == 3:
                info_b["brightness"] = b.reshape((B, 1, 1, 1))

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        self._ensure_render_token(data_b, debug)
        obs_b = self.render_obs(data_b, info_b)  # [B,H,2W,3]
        return st1.replace(obs=obs_b[0])

    def step(self, state: State, action: jax.Array) -> State:
        """Public step: single-world physics + broadcasted render -> pixels in obs (slow)."""
        st1 = self.step_physics(state, action)

        B = self.render_batch_size
        data_b = _broadcast_tree_to_batch(st1.data, B)
        info_b = _broadcast_tree_to_batch(st1.info, B)
        if "brightness" in info_b:
            b = info_b["brightness"]
            if b.ndim == 3:
                info_b["brightness"] = b.reshape((B, 1, 1, 1))

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        self._ensure_render_token(data_b, debug)
        obs_b = self.render_obs(data_b, info_b)
        return st1.replace(obs=obs_b[0])

    # -------------------------
    # Rewards (unchanged, except robust body indexing)
    # -------------------------

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]):
        box_pos = _batched_body_xpos(data, self._obj_body)
        target_pos = info["target_pos"]

        gripper_pos = _batched_body_xpos(data, self._hand_body)
        d_gripper_box = jp.linalg.norm(gripper_pos - box_pos, axis=-1)
        d_box_target = jp.linalg.norm(box_pos - target_pos, axis=-1)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05), 1.0)
        info = {**info, "reached_box": reached_box}

        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        floor_coll = jp.zeros_like(gripper_box)
        for sensor_id in self._floor_hand_found_sensor:
            floor_coll = floor_coll + (
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            ).astype(jp.float32)
        no_floor_collision = jp.where(floor_coll > 0, 0.0, 1.0)

        hand_box = (
            data.sensordata[self._mj_model.sensor_adr[self._box_hand_found_sensor]] > 0
        )
        no_box_collision = jp.where(hand_box, 0.0, 1.0)

        raw_rewards = dict(
            gripper_box=gripper_box,
            box_target=box_target,
            no_floor_collision=no_floor_collision,
            no_box_collision=no_box_collision,
            robot_target_qpos=jp.zeros_like(gripper_box),
        )
        return info, raw_rewards

    def modify_model(self, mj_model: mujoco.MjModel):
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        mesh_id = mj_model.mesh("finger_1").id
        geoms = [
            idx
            for idx, data_id in enumerate(mj_model.geom_dataid)
            if data_id == mesh_id
        ]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        return mj_model

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
