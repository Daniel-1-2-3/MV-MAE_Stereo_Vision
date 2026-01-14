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
"""Bring a box to a target and orientation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import contextlib
import importlib.util
import os
from pathlib import Path

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
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda_kinematics

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
        self.render_batch_size = int(render_batch_size)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )
        self._xml_path = xml_path.as_posix()

        self._model_assets = dict(panda.get_assets())
        menagerie_dir = (
            Path.cwd()
            / "mujoco_playground_external_deps"
            / "mujoco_menagerie"
            / "franka_emika_panda"
        )
        self._model_assets = _add_assets(self._model_assets, menagerie_dir)

        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(
                xml_path.read_text(),
                assets=self._model_assets,
            )
        )
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model: mujoco.MjModel = mj_model
        self._mjx_model: mjx.Model = mjx.put_model(mj_model, impl=self._config.impl)

        self._post_init(obj_name="box", keyframe="low_home")

        self._floor_hand_geom_ids = [
            self._mj_model.geom(geom).id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]
        self._floor_geom_id = self._mj_model.geom("floor").id
        self.renderer: BatchRenderer = self._create_renderer()
        self._render_token: Optional[jax.Array] = None

        self._render_fn = lambda token, data: self.renderer.render(token, data, self._mj_model)

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        # PandaBase post-init can mutate MuJoCo model
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

    def _make_batched_data(self, B: int) -> mjx.Data:
        """Create a batched mjx.Data with leading batch axis B and overwrite qpos/qvel/ctrl."""
        m = self._mjx_model

        # single-world data, then broadcast all leaves to [B, ...]
        data0 = mjx.make_data(m)

        def _batch_leaf(x):
            if not hasattr(x, "ndim"):
                return x
            if x.ndim == 0:
                return jp.broadcast_to(x, (B,))
            # Always prepend a batch axis; do NOT special-case x.shape[0] == B
            return jp.broadcast_to(x, (B,) + x.shape)

        data = jax.tree_util.tree_map(_batch_leaf, data0)

        nq = int(m.nq)
        nv = int(m.nv)

        init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[..., :nq]  # (nq,)
        qpos = jp.broadcast_to(init_q0, (B, nq))

        qvel = jp.zeros((B, nv), dtype=jp.float32)

        ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
        ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        return data

    def reset(self, rng: jax.Array) -> State:
        if rng.ndim == 1:
            rng = rng[None, :]
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"

        dev = next(iter(rng.devices())) if hasattr(rng, "devices") else None
        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            m = self._mjx_model
            B = int(rng.shape[0])
            print("test batch size = ", B)

            if debug:
                print(f"[pick_env] reset: B={B} render_batch_size={self.render_batch_size}")
                if dev is not None:
                    print(f"[pick_env] reset device: {dev}")

            keys = jax.vmap(lambda k: jax.random.split(k, 5))(rng)
            rng_main = keys[:, 0, :]
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]
            rng_brightness = keys[:, 3, :]
            rng_robot = keys[:, 4, :]

            # Sample box + target positions (both [B, 3])
            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)

            data = self._make_batched_data(B)

            # Make robot joints unique per-world (object qpos is set separately below).
            eps = jp.float32(0.01)
            qpos_noise = jax.vmap(
                lambda k: jax.random.uniform(
                    k, (int(self._mjx_model.nq),), minval=-eps, maxval=eps
                )
            )(rng_robot)
            data = data.replace(
                qpos=data.qpos.at[:, : self._obj_qposadr].add(
                    qpos_noise[:, : self._obj_qposadr]
                )
            )

            # Set object position into qpos (batched)
            data = data.replace(
                qpos=data.qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
            )

            # Set target mocap (batched)
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

            if self._render_token is None:
                if debug:
                    backend = jax.lib.xla_bridge.get_backend()
                    print("[diag] jax/jaxlib:", jax.__version__, jaxlib.__version__)
                    print("[diag] backend:", backend.platform, "|", backend.platform_version)
                    print(
                        "[diag] jaxlib file:",
                        jaxlib.__file__,
                        "mtime:",
                        os.path.getmtime(jaxlib.__file__),
                    )

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
                    print(
                        "init mj_model type:",
                        type(self._mj_model),
                        "has geom_quat:",
                        hasattr(self._mj_model, "geom_quat"),
                    )
                    print(
                        "init mjx_model type:",
                        type(self._mjx_model),
                        "has geom_quat:",
                        hasattr(self._mjx_model, "geom_quat"),
                    )

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

                    try:
                        _, rgb, _ = self.renderer.render(self._render_token, data, self._mj_model)
                        jax.block_until_ready(rgb)
                        print("[diag] smoke render ok:", rgb.shape, rgb.dtype)
                    except Exception as e:
                        print("[diag] smoke render FAILED:", type(e).__name__, e)
                        raise

                    canary = jp.zeros((1,), dtype=jp.float32)
                    jax.block_until_ready(canary)
                    print("[diag] post-render canary ok")

            render_token = self._render_token

            metrics = {
                "out_of_bounds": jp.zeros((B,), dtype=jp.float32),
                **{k: jp.zeros((B,), dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
            }
            info = {
                "rng": rng_main,
                "target_pos": target_pos,
                "reached_box": jp.zeros((B,), dtype=jp.float32),
                "truncation": jp.zeros((B,), dtype=jp.float32),
                "render_token": render_token,
                "brightness": brightness,
            }

            info, obs = self._get_obs(data, info)

            reward = jp.zeros((B,), dtype=jp.float32)
            done = jp.zeros((B,), dtype=jp.float32)

            return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action_scale = self._config.action_scale
        delta = action * action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0, axis=-1) | (box_pos[:, 2] < 0.0)

        state.metrics.update(
            out_of_bounds=out_of_bounds.astype(jp.float32),
            **{k: v for k, v in raw_rewards.items()},
        )

        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        info, obs = self._get_obs(data, info)

        return state.replace(
            data=data,
            obs=obs,
            reward=total_reward,
            done=done.astype(jp.float32),
            metrics=state.metrics,
            info=info,
        )

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]):
        box_pos = data.xpos[self._obj_body]
        target_pos = info["target_pos"]

        gripper_pos = data.xpos[self._hand_body]
        d_gripper_box = jp.linalg.norm(gripper_pos - box_pos, axis=-1)
        d_box_target = jp.linalg.norm(box_pos - target_pos, axis=-1)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05), 1.0)
        info = {**info, "reached_box": reached_box}

        # reward components
        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        # Penalize collision with the floor.
        floor_coll = jp.zeros_like(gripper_box)
        for sensor_id in self._floor_hand_found_sensor:
            floor_coll = floor_coll + (
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            ).astype(jp.float32)
        no_floor_collision = jp.where(floor_coll > 0, 0.0, 1.0)

        # Penalize collision with box (hand-box contact sensor).
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

    def render_pixels(self, render_token: jax.Array, data_batched: mjx.Data) -> jax.Array:
        _, rgb, _ = self._render_fn(render_token, data_batched)
        print("rgb made")

        if rgb.shape[0] == 2:
            left = rgb[0]
            right = rgb[1]
        else:
            left = rgb[:, 0]
            right = rgb[:, 1]
        print("fail here 1")

        left = left[..., :3].astype(jp.float32) / 255.0
        right = right[..., :3].astype(jp.float32) / 255.0
        print("fail here 2")

        pixels = jp.concatenate([left, right], axis=2)
        print("fail here 3")
        return pixels

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> tuple[dict[str, Any], jax.Array]:
        pixels = self.render_pixels(info["render_token"], data)
        if "brightness" in info:
            pixels = adjust_brightness(pixels, info["brightness"])
        return info, pixels

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size to non-zero so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make the finger pads white for increased visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [
            idx for idx, data_id in enumerate(mj_model.geom_dataid) if data_id == mesh_id
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
