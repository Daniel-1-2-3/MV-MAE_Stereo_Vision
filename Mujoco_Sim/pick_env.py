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

Matches your intended architecture:

FAST PATH (training / parallel):
- reset_physics(rng_single) -> State (single-world)
- step_physics(state_single, action_single) -> State (single-world)
- Wrapper does:
    reset_batched = jax.vmap(env.reset_physics, in_axes=0, out_axes=0)
    step_batched  = jax.vmap(env.step_physics,  in_axes=(0,0), out_axes=0)
    step_batched  = jax.jit(step_batched, donate_argnums=(0,))
  Rendering is called OUTSIDE the physics JIT:
    pixels = env.render_pixels_batched(state_batched.data, state_batched.info)

SLOW PATH (public compatibility):
- reset(rng_single) and step(state_single, action_single) do:
    single-world physics -> broadcast to render_batch_size -> render -> return pixels for world 0
  This is intentionally slow; do NOT use it in training.

Renderer init safety:
- renderer.init(data_batched, mjx_model) is called once with correctly batched data matching num_worlds
- optional smoke render in debug mode (keeps your “fixed CUDA error after smoke” behavior)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import jax
import jax.numpy as jp
import jaxlib
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from Custom_Mujoco_Playground._src import mjx_env
from Custom_Mujoco_Playground._src.mjx_env import State
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda
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


def _broadcast_tree_to_batch(tree, B: int):
    """Broadcast a pytree of arrays/scalars to a leading batch dim B."""
    def _bcast(x):
        if not hasattr(x, "ndim"):
            return x
        if x.ndim == 0:
            return jp.broadcast_to(x, (B,))
        return jp.broadcast_to(x, (B,) + x.shape)
    return jax.tree_util.tree_map(_bcast, tree)


def _normalize_brightness_shape(info_batched: Dict[str, Any], B: int) -> Dict[str, Any]:
    """Ensure brightness is [B,1,1,1] if present."""
    if "brightness" not in info_batched:
        return info_batched
    b = info_batched["brightness"]
    # Accept [1,1,1] (single-world) or [B,1,1,1] (batched)
    if hasattr(b, "ndim"):
        if b.ndim == 3:
            info_batched = dict(info_batched)
            info_batched["brightness"] = jp.broadcast_to(b, (B, 1, 1, 1))
        elif b.ndim == 4 and b.shape[0] == B:
            pass
        elif b.ndim == 1 and b.shape[0] == B:
            info_batched = dict(info_batched)
            info_batched["brightness"] = b.reshape((B, 1, 1, 1))
    return info_batched


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
        # Render params (Madrona num_worlds)
        self.render_batch_size = int(render_batch_size)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        # Cache debug flag ONCE (avoid env-var reads inside jitted code)
        self._debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"

        # Base MJX env init (DO NOT call super())
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        # XML path
        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )
        self._xml_path = xml_path.as_posix()

        # Assets
        self._model_assets = dict(panda.get_assets())
        menagerie_dir = (
            Path.cwd()
            / "mujoco_playground_external_deps"
            / "mujoco_menagerie"
            / "franka_emika_panda"
        )
        self._model_assets = _add_assets(self._model_assets, menagerie_dir)

        # Build MuJoCo model (merged assets) + tweak
        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        )
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model: mujoco.MjModel = mj_model
        self._mjx_model: mjx.Model = mjx.put_model(mj_model, impl=self._config.impl)

        # Task post init
        self._post_init(obj_name="box", keyframe="low_home")

        # IDs used by rewards
        hand_geom_id = self._mj_model.geom("hand_capsule").id
        self._hand_body = int(self._mj_model.geom_bodyid[hand_geom_id])
        self._floor_geom_id = self._mj_model.geom("floor").id

        # Sensors used by rewards (resolve by name; keep robust behavior)
        sensor_names = [self._mj_model.sensor(i).name for i in range(self._mj_model.nsensor)]

        floor_ids = []
        for i, n in enumerate(sensor_names):
            nl = n.lower()
            if ("floor" in nl) and (("hand" in nl) or ("finger" in nl) or ("gripper" in nl)):
                floor_ids.append(i)
        if not floor_ids:
            for i, n in enumerate(sensor_names):
                if "floor" in n.lower():
                    floor_ids.append(i)

        box_hand_ids = []
        for i, n in enumerate(sensor_names):
            nl = n.lower()
            if (("box" in nl) or ("cube" in nl) or ("object" in nl)) and (("hand" in nl) or ("finger" in nl) or ("gripper" in nl)):
                box_hand_ids.append(i)
        if not box_hand_ids:
            for i, n in enumerate(sensor_names):
                nl = n.lower()
                if ("hand" in nl) or ("finger" in nl) or ("gripper" in nl):
                    box_hand_ids.append(i)

        self._floor_hand_found_sensor = [int(i) for i in floor_ids]
        self._box_hand_found_sensor = int(box_hand_ids[0]) if box_hand_ids else -1

        if self._debug:
            print("[diag] nsensor:", self._mj_model.nsensor)
            print("[diag] floor sensors:", [(i, sensor_names[i]) for i in self._floor_hand_found_sensor])
            if self._box_hand_found_sensor >= 0:
                print("[diag] box-hand sensor:", (self._box_hand_found_sensor, sensor_names[self._box_hand_found_sensor]))
            else:
                print("[diag] box-hand sensor: NOT FOUND (set to -1)")

        # Renderer created now; token initialized lazily with correctly batched data
        self.renderer: BatchRenderer = self._create_renderer()
        self._render_token: Optional[jax.Array] = None

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        # PandaBase post-init can mutate the MuJoCo model; re-upload to MJX.
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._sample_orientation = False

    def _create_renderer(self) -> BatchRenderer:
        enabled_geom_groups = np.asarray(self._config.vision_config.enabled_geom_groups, dtype=np.int32)
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
    # Rendering (batched, non-jit)
    # -------------------------

    def ensure_render_initialized(self, data_batched: mjx.Data) -> None:
        """Initialize renderer token once with correctly batched data.

        Critical invariant: leading batch dim must equal self.render_batch_size (num_worlds).
        """
        if self._render_token is not None:
            return

        if self._debug:
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

            print("[diag] data.geom_xpos shape:", getattr(data_batched.geom_xpos, "shape", None))
            print("[diag] calling renderer.init(...)")

        token, rgb0, depth0 = self.renderer.init(data_batched, self._mjx_model)
        self._render_token = token

        # One-time smoke render (ONLY debug): this is where your old version proved CUDA was OK.
        if self._debug:
            token2, rgb, _ = self.renderer.render(self._render_token, data_batched, self._mjx_model)
            self._render_token = token2
            jax.block_until_ready(rgb)
            print("[diag] smoke render ok:", rgb.shape, rgb.dtype)

    def render_pixels_batched(self, data_batched: mjx.Data, info_batched: Dict[str, Any]) -> jax.Array:
        """Render stereo pixels for batched mjx.Data. Returns uint8 [B,H,2W,3]."""
        self.ensure_render_initialized(data_batched)
        assert self._render_token is not None

        token2, rgb, _depth = self.renderer.render(self._render_token, data_batched, self._mjx_model)
        self._render_token = token2

        # Handle possible camera/world axis order
        B = int(getattr(data_batched.geom_xpos, "shape", [0])[0])

        if rgb.ndim == 5 and rgb.shape[0] == 2 and rgb.shape[1] == B:
            left = rgb[0]   # [B,H,W,4]
            right = rgb[1]
        else:
            left = rgb[:, 0]
            right = rgb[:, 1]

        left = left[..., :3]
        right = right[..., :3]
        pixels = jp.concatenate([left, right], axis=2)  # [B,H,2W,3]

        # Optional brightness only if configured to vary
        bmin, bmax = self._config.obs_noise.brightness
        if not (float(bmin) == 1.0 and float(bmax) == 1.0):
            info_batched = _normalize_brightness_shape(info_batched, B)
            if "brightness" in info_batched:
                b = info_batched["brightness"]  # [B,1,1,1]
                x = pixels.astype(jp.float32) * b
                x = jp.clip(x, 0.0, 255.0)
                pixels = x.astype(jp.uint8)

        return pixels.astype(jp.uint8)

    # -------------------------
    # Physics (single-world, JIT-friendly)
    # -------------------------

    def reset_physics(self, rng: jax.Array) -> State:
        """Single-world reset. Intended to be vmapped externally for batch."""
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

        nq = int(m.nq)
        nv = int(m.nv)
        qpos0 = jp.asarray(self._init_q, dtype=jp.float32)[..., :nq]
        qvel0 = jp.zeros((nv,), dtype=jp.float32)
        ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)

        data = data.replace(qpos=qpos0, qvel=qvel0, ctrl=ctrl0)

        # Place box
        data = data.replace(qpos=data.qpos.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos))

        # Set mocap target (index 0 for single-world)
        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[0, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[0, :].set(target_quat),
        )

        data = mjx.forward(m, data)

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=jp.float32),
            **{k: jp.array(0.0, dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
        }

        info: Dict[str, Any] = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=jp.float32),
            "truncation": jp.array(0.0, dtype=jp.float32),
        }

        # Only include brightness if it varies (avoid extra per-step work)
        bmin, bmax = self._config.obs_noise.brightness
        if not (float(bmin) == 1.0 and float(bmax) == 1.0):
            brightness = jax.random.uniform(rng_brightness, (1,), minval=bmin, maxval=bmax).reshape((1, 1, 1, 1))
            info["brightness"] = brightness

        # Keep obs small in physics state for fast vmapped/jitted stepping
        obs = jp.zeros((1,), dtype=jp.float32)
        reward = jp.array(0.0, dtype=jp.float32)
        done = jp.array(0.0, dtype=jp.float32)

        return State(data, obs, reward, done, metrics, info)

    def step_physics(self, state: State, action: jax.Array) -> State:
        """Single-world physics step. Intended to be vmapped externally for batch."""
        delta = action * self._config.action_scale
        ctrl = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[self._obj_body, :]  # (3,)
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)
        done = out_of_bounds | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        new_metrics = dict(state.metrics)
        new_metrics["out_of_bounds"] = out_of_bounds.astype(jp.float32)
        for k, v in raw_rewards.items():
            new_metrics[k] = v

        return state.replace(
            data=data,
            obs=state.obs,  # keep tiny
            reward=total_reward,
            done=done.astype(jp.float32),
            metrics=new_metrics,
            info=info,
        )

    # -------------------------
    # Public API (slow path, like you did)
    # -------------------------

    def reset(self, rng: jax.Array) -> State:
        """Public reset: single-world physics + broadcasted render -> pixels in obs (SLOW)."""
        st1 = self.reset_physics(rng)

        B = int(self.render_batch_size)
        data_b = _broadcast_tree_to_batch(st1.data, B)
        info_b = _broadcast_tree_to_batch(st1.info, B)
        info_b = _normalize_brightness_shape(info_b, B)

        # init token once (this is your smoke-test-safe init)
        self.ensure_render_initialized(data_b)

        # render and return only world 0 image
        pixels_b = self.render_pixels_batched(data_b, info_b)  # [B,H,2W,3] uint8
        return st1.replace(obs=pixels_b[0])

    def step(self, state: State, action: jax.Array) -> State:
        """Public step: single-world physics + broadcasted render -> pixels in obs (SLOW)."""
        st1 = self.step_physics(state, action)

        B = int(self.render_batch_size)
        data_b = _broadcast_tree_to_batch(st1.data, B)
        info_b = _broadcast_tree_to_batch(st1.info, B)
        info_b = _normalize_brightness_shape(info_b, B)

        self.ensure_render_initialized(data_b)
        pixels_b = self.render_pixels_batched(data_b, info_b)
        return st1.replace(obs=pixels_b[0])

    # -------------------------
    # Rewards
    # -------------------------

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]):
        box_pos = data.xpos[self._obj_body, :]
        target_pos = info["target_pos"]
        gripper_pos = data.xpos[self._hand_body, :]

        d_gripper_box = jp.linalg.norm(gripper_pos - box_pos)
        d_box_target = jp.linalg.norm(box_pos - target_pos)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05).astype(jp.float32), 1.0)
        info = {**info, "reached_box": reached_box}

        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        floor_coll = jp.array(0.0, dtype=jp.float32)
        for sensor_id in self._floor_hand_found_sensor:
            adr = self._mj_model.sensor_adr[sensor_id]
            floor_coll = floor_coll + (data.sensordata[adr] > 0).astype(jp.float32)
        no_floor_collision = jp.where(floor_coll > 0, 0.0, 1.0)

        if self._box_hand_found_sensor >= 0:
            adr = self._mj_model.sensor_adr[self._box_hand_found_sensor]
            hand_box = (data.sensordata[adr] > 0)
        else:
            hand_box = jp.array(False)
        no_box_collision = jp.where(hand_box, 0.0, 1.0)

        raw_rewards = dict(
            gripper_box=gripper_box,
            box_target=box_target,
            no_floor_collision=no_floor_collision,
            no_box_collision=no_box_collision,
            robot_target_qpos=jp.array(0.0, dtype=jp.float32),
        )
        return info, raw_rewards

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size to non-zero so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make the finger pads white for increased visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [idx for idx, data_id in enumerate(mj_model.geom_dataid) if data_id == mesh_id]
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
