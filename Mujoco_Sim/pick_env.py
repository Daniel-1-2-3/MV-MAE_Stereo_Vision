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

from typing import Any, Dict, Optional, Union

import contextlib
import os
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from Custom_Mujoco_Playground._src import mjx_env
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda_kinematics
from Custom_Mujoco_Playground._src.mjx_env import State  # pylint: disable=g-importing-member
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
        use_rasterizer=False,
        enabled_geom_groups=[0, 1, 2],
    )


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for bring_to_target tasks."""
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                gripper_box=4.0,
                box_target=8.0,
                no_floor_collision=0.25,
                robot_target_qpos=0.3,
            )
        ),
        vision_config=default_vision_config(),
        obs_noise=config_dict.create(brightness=[1.0, 1.0]),
        impl="jax",
        # IMPORTANT: don't make this enormous. Huge contact buffers get threaded through custom calls
        # and can destabilize Madrona/JAX interop. Override upward only if you truly need it.
        nconmax=4096,
        njmax=128,
    )
    return config


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
            mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        )
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)

        self._post_init(obj_name="box", keyframe="low_home")

        # Geom ids for parts of the robot that must not hit the floor.
        self._floor_hand_geom_ids = [
            self._mj_model.geom(geom).id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]
        self._floor_geom_id = self._mj_model.geom("floor").id

        # Renderer (constructed once) + token (created once)
        self._render_token = None
        self._init_renderer()
        self._render_jit = lambda token, data: self.renderer.render(token, data, self._mjx_model)

        # Cache the exact device Madrona should use
        self._render_device = self._get_render_device()

        # Create token ONCE with a deterministic valid state (prevents repeated init/memory churn)
        self._render_token = self._create_render_token()

    def _get_render_device(self):
        """Return the JAX device matching vision_config.gpu_id (or None if unavailable)."""
        try:
            gpu_id = int(self._config.vision_config.gpu_id)
            devs = jax.devices("gpu")
            if 0 <= gpu_id < len(devs):
                return devs[gpu_id]
        except Exception:
            pass
        return None

    def _post_init(self, obj_name="box", keyframe="low_home"):
        super()._post_init(obj_name, keyframe)

        # Ensure mjx_model matches finalized mj_model
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._init_q = jp.asarray(self._mj_model.keyframe(keyframe).qpos, dtype=jp.float32)
        self._guide_q = self._mj_model.keyframe("picked").qpos
        self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
        self._start_tip_transform = panda_kinematics.compute_franka_fk(self._init_ctrl[:7])

    def _init_renderer(self):
        gpu_id = int(self._config.vision_config.gpu_id)
        use_rasterizer = bool(self._config.vision_config.use_rasterizer)

        enabled_geom_groups = np.asarray(self._config.vision_config.enabled_geom_groups, dtype=np.int32)
        enabled_geom_groups.setflags(write=False)

        self.renderer = BatchRenderer(
            m=self._mjx_model,
            gpu_id=gpu_id,
            num_worlds=int(self.render_batch_size),
            batch_render_view_width=int(self.render_width),
            batch_render_view_height=int(self.render_height),
            enabled_geom_groups=enabled_geom_groups,
            enabled_cameras=None,
            add_cam_debug_geo=False,
            use_rasterizer=use_rasterizer,
            viz_gpu_hdls=None,
        )

    def _create_render_token(self):
        """Create Madrona token once, using original vmap style + strict GPU placement."""
        m = self._mjx_model
        B = int(self.render_batch_size)
        nq = int(m.nq)
        nv = int(m.nv)

        # Ensure scalar index (prevents accidental (1,) indexing => (B,1,3) slices)
        mocap_id = int(np.asarray(self._mocap_target).reshape(()))

        # Force everything used for init onto the renderer GPU.
        dev = self._render_device
        ctx = jax.default_device(dev) if dev is not None else contextlib.nullcontext()
        with ctx:
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)
            box_pos = jp.broadcast_to(base + jp.array([0.0, 0.0, 0.0], jp.float32), (B, 3))
            target_pos = jp.broadcast_to(base + jp.array([0.0, 0.0, 0.30], jp.float32), (B, 3))
            target_quat = jp.broadcast_to(jp.array([1.0, 0.0, 0.0, 0.0], jp.float32), (B, 4))

            init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[:nq]
            qpos = jp.broadcast_to(init_q0, (B, nq))
            qpos = qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)

            ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
            ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

            # ORIGINAL STYLE: vmap(make_data) then vmap(forward)
            # This avoids MJX scan shape mismatches you saw with broadcasted batched Data.
            data = jax.vmap(lambda _: mjx.make_data(m))(jp.arange(B, dtype=jp.int32))
            data = data.replace(
                qpos=qpos,
                qvel=jp.zeros((B, nv), dtype=jp.float32),
                ctrl=ctrl,
            )
            data = data.replace(
                mocap_pos=data.mocap_pos.at[:, mocap_id, :].set(target_pos),
                mocap_quat=data.mocap_quat.at[:, mocap_id, :].set(target_quat),
            )

            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            # CRITICAL: materialize data BEFORE calling renderer.init (prevents bad pointers into custom call)
            jax.tree_util.tree_map(jax.block_until_ready, data)

            token, _, _ = self.renderer.init(data, m)
            jax.tree_util.tree_map(jax.block_until_ready, token)
            return token

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size to non-zero so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make the finger pads white for increased visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [idx for idx, data_id in enumerate(mj_model.geom_dataid) if data_id == mesh_id]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        return mj_model

    def _has_contact_with_floor(self, data: mjx.Data, geom_id: int) -> jax.Array:
        g1 = data.contact.geom1
        g2 = data.contact.geom2
        idx = jp.arange(g1.shape[-1])
        valid = idx[None, :] < data.ncon[:, None]

        pair = jp.logical_or(
            jp.logical_and(g1 == geom_id, g2 == self._floor_geom_id),
            jp.logical_and(g2 == geom_id, g1 == self._floor_geom_id),
        )
        return jp.any(pair & valid, axis=-1)

    def reset(self, rng: jax.Array) -> State:
        if rng.ndim == 1:
            rng = rng[None, :]

        # Force batch size to match renderer num_worlds.
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        # Keep reset on the renderer GPU too (so render sees consistent device buffers)
        dev = self._render_device
        ctx = jax.default_device(dev) if dev is not None else contextlib.nullcontext()
        with ctx:
            if os.environ.get("PICK_ENV_DEBUG", "0") == "1":
                print(f"[pick_env] running from: {__file__}")

            m = self._mjx_model
            B = int(rng.shape[0])
            nq = int(m.nq)
            nv = int(m.nv)

            mocap_id = int(np.asarray(self._mocap_target).reshape(()))

            keys = jax.vmap(lambda k: jax.random.split(k, 3))(rng)
            rng_main = keys[:, 0, :]
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]

            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)

            init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[:nq]
            qpos = jp.broadcast_to(init_q0, (B, nq))
            qpos = qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)

            ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
            ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

            target_quat0 = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
            target_quat = jp.broadcast_to(target_quat0, (B, 4))

            # ORIGINAL STYLE: per-env make_data + forward
            data = jax.vmap(lambda _: mjx.make_data(m))(jp.arange(B, dtype=jp.int32))
            data = data.replace(
                qpos=qpos,
                qvel=jp.zeros((B, nv), dtype=jp.float32),
                ctrl=ctrl,
            )
            data = data.replace(
                mocap_pos=data.mocap_pos.at[:, mocap_id, :].set(target_pos),
                mocap_quat=data.mocap_quat.at[:, mocap_id, :].set(target_quat),
            )
            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            metrics = {
                "out_of_bounds": jp.zeros((B,), dtype=jp.float32),
                **{k: jp.zeros((B,), dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
            }
            info = {
                "rng": rng_main,
                "target_pos": target_pos,
                "reached_box": jp.zeros((B,), dtype=jp.float32),
                "render_token": self._render_token,
            }

            info, obs = self._get_obs(data, info)

            reward = jp.zeros((B,), dtype=jp.float32)
            done = jp.zeros((B,), dtype=jp.float32)
            return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        # Step can run on whatever device state.data lives on; renderer uses _render_device anyway.
        delta = action * self._config.action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[:, self._obj_body, :]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0, axis=-1)
        out_of_bounds |= box_pos[:, 2] < 0.0
        nan_qpos = jp.any(jp.isnan(data.qpos), axis=-1)
        nan_qvel = jp.any(jp.isnan(data.qvel), axis=-1)
        done = (out_of_bounds | nan_qpos | nan_qvel).astype(jp.float32)

        metrics = {**state.metrics, **raw_rewards, "out_of_bounds": out_of_bounds.astype(float)}
        info, obs = self._get_obs(data, info)
        return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]

        box_pos = data.xpos[:, self._obj_body, :]
        gripper_pos = data.site_xpos[:, self._gripper_site, :]
        pos_err = jp.linalg.norm(target_pos - box_pos, axis=-1)

        box_mat_raw = data.xmat[:, self._obj_body, ...]
        if box_mat_raw.ndim == 3 and box_mat_raw.shape[-1] == 9:
            box_mat = box_mat_raw.reshape((box_mat_raw.shape[0], 3, 3))
        else:
            box_mat = box_mat_raw

        target_mat = math.quat_to_mat(data.mocap_quat[:, self._mocap_target, :])

        target_6 = target_mat[:, :, :2].reshape((target_mat.shape[0], 6))
        box_6 = box_mat[:, :, :2].reshape((box_mat.shape[0], 6))
        rot_err = jp.linalg.norm(target_6 - box_6, axis=-1)

        box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
        gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos, axis=-1))

        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[:, self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr],
                axis=-1,
            )
        )

        hand_floor = jp.stack(
            [self._has_contact_with_floor(data, gid) for gid in self._floor_hand_geom_ids],
            axis=0,
        )
        floor_collision = jp.any(hand_floor, axis=0)
        no_floor_collision = 1.0 - floor_collision.astype(jp.float32)

        close = jp.linalg.norm(box_pos - gripper_pos, axis=-1) < 0.012
        reached_box = jp.maximum(info["reached_box"], close.astype(jp.float32))
        info = {**info, "reached_box": reached_box}

        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target * info["reached_box"],
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
        return info, rewards

    def render_pixels(self, render_token: Any, data_batched: mjx.Data) -> jax.Array:
        # Ensure render inputs live on the renderer GPU.
        dev = self._render_device
        ctx = jax.default_device(dev) if dev is not None else contextlib.nullcontext()
        with ctx:
            data_batched = jax.device_put(data_batched, dev) if dev is not None else data_batched
            jax.tree_util.tree_map(jax.block_until_ready, data_batched)

            _, rgb, _ = self._render_jit(render_token, data_batched)

        # Support either [2, B, H, W, 4] or [B, 2, H, W, 4]
        if rgb.shape[0] == 2:
            left = rgb[0, ..., :3]
            right = rgb[1, ..., :3]
        elif rgb.shape[1] == 2:
            left = rgb[:, 0, ..., :3]
            right = rgb[:, 1, ..., :3]
        else:
            raise ValueError(f"Unexpected rgb shape: {rgb.shape}")

        left = left.astype(jp.float32) / 255.0
        right = right.astype(jp.float32) / 255.0
        return jp.concatenate([left, right], axis=2)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        pixels = self.render_pixels(info["render_token"], data)
        return info, pixels
