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
                gripper_box=4.0,  # Gripper goes to the box.
                box_target=8.0,  # Box goes to the target mocap.
                no_floor_collision=0.25,  # Do not collide the gripper with the floor.
                robot_target_qpos=0.3,  # Arm stays close to target pose.
            )
        ),
        vision_config=default_vision_config(),
        obs_noise=config_dict.create(brightness=[1.0, 1.0]),
        impl="jax",
        nconmax=24 * 2048,
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
        self.render_batch_size = render_batch_size
        self.render_width = render_width
        self.render_height = render_height

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

        # Cache render token to not re-run renderer.init on every reset.
        # Renderer + token are initialized lazily on first reset() with real data
        self._render_token = None
        self._init_renderer()  # constructs BatchRenderer (no init/render here)
        
    def _post_init(self, obj_name="box", keyframe="low_home"):
        super()._post_init(obj_name, keyframe)

        # IMPORTANT: ensure mjx_model matches whatever model state PandaBase finalized
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._init_q = jp.asarray(self._mj_model.keyframe(keyframe).qpos, dtype=jp.float32)
        self._guide_q = self._mj_model.keyframe("picked").qpos
        self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
        self._start_tip_transform = panda_kinematics.compute_franka_fk(self._init_ctrl[:7])

    def _validate_render_model(self, m_render: mjx.Model, num_worlds: int) -> None:
        """Fail fast if the renderer model isn't batched as [num_worlds, ...]."""
        bad = []

        def _check(x):
            nonlocal bad
            if len(bad) >= 5:
                return
            if isinstance(x, (jax.Array, np.ndarray)) and getattr(x, "ndim", 0) > 0:
                if x.shape[0] != num_worlds:
                    bad.append((tuple(x.shape), str(getattr(x, "dtype", ""))))

        jax.tree_util.tree_map(_check, m_render)
        if bad:
            raise ValueError(
                "Madrona render model is not batched for num_worlds="
                f"{num_worlds}. Example leaf shapes/dtypes: {bad}"
            )

    def _make_render_model(self, m: mjx.Model) -> mjx.Model:
        """Tiles the MJX model so every array leaf is [num_worlds, ...] for Madrona."""
        num_worlds = self.render_batch_size

        # Put everything on the same GPU Madrona is targeting.
        dev = None
        try:
            gpu_id = int(self._config.vision_config.gpu_id)
            dev = jax.devices("gpu")[gpu_id]
        except Exception:
            dev = None

        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            def _tile_leaf(x):
                if isinstance(x, (jax.Array, np.ndarray)) and getattr(x, "ndim", 0) > 0:
                    if x.shape[0] == num_worlds:
                        return x
                    xj = jp.asarray(x)
                    return jp.repeat(xj[None, ...], num_worlds, axis=0)
                return x

            m_render = jax.tree_util.tree_map(_tile_leaf, m)

        self._validate_render_model(m_render, num_worlds)
        return m_render

    def _init_renderer(self):
        # Madrona pybind is strict about scalar types + numpy dtypes for static buffers.
        gpu_id = int(self._config.vision_config.gpu_id)
        use_rasterizer = bool(self._config.vision_config.use_rasterizer)

        enabled_geom_groups = np.asarray(
            self._config.vision_config.enabled_geom_groups, dtype=np.int32
        )
        # Many pybind signatures require non-writable buffers.
        enabled_geom_groups.setflags(write=False)

        self.renderer = BatchRenderer(
            # IMPORTANT: keep the model unbatched (renderer expects unbatched mesh buffers).
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

        # Called from inside jitted reset/step; keep as a pure function wrapper.
        def _render(token, data):
            return self.renderer.render(token, data, self._mjx_model)

        self._render_jit = _render

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size to non-zero so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make the finger pads white for increased visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [
            idx
            for idx, data_id in enumerate(mj_model.geom_dataid)
            if data_id == mesh_id
        ]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        return mj_model

    def _has_contact_with_floor(self, data: mjx.Data, geom_id: int) -> jax.Array:
        g1 = data.contact.geom1  # [B, nconmax]
        g2 = data.contact.geom2
        idx = jp.arange(g1.shape[-1])  # [nconmax]
        valid = idx[None, :] < data.ncon[:, None]  # [B, nconmax]

        pair = jp.logical_or(
            jp.logical_and(g1 == geom_id, g2 == self._floor_geom_id),
            jp.logical_and(g2 == geom_id, g1 == self._floor_geom_id),
        )
        return jp.any(pair & valid, axis=-1)

    def reset(self, rng: jax.Array) -> State:
        if rng.ndim == 1:
            rng = rng[None, :]  # -> (1, 2)

        # Force batch size to match Madrona renderer num_worlds.
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        dev = next(iter(rng.devices())) if hasattr(rng, "devices") else None
        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            if os.environ.get("PICK_ENV_DEBUG", "0") == "1":
                print(f"[pick_env] running from: {__file__}")

            m = self._mjx_model
            B = rng.shape[0]

            # Ensure mocap index is a scalar, not shape-(1,) array / list.
            mocap_id = int(np.asarray(self._mocap_target).reshape(()))

            keys = jax.vmap(lambda k: jax.random.split(k, 3))(rng)
            rng_main = keys[:, 0, :]
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]

            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)  # (3,)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)  # (B, 3)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)  # (B, 3)

            nq = m.nq
            nv = m.nv

            init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[..., :nq]  # (nq,)
            qpos = jp.broadcast_to(init_q0, (B, nq))  # (B, nq)
            qpos = qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)

            ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
            ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

            target_quat0 = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
            target_quat = jp.broadcast_to(target_quat0, (B, 4))

            # Build ONE Data, broadcast to [B, ...], then run one batched forward.
            # This avoids vmapping mjx.make_data(m), which can produce renderer-hostile buffers.
            d0 = mjx.make_data(m)

            def _bcast(x):
                # Broadcast arrays to [B, ...] while leaving scalars alone.
                # (Most mjx.Data leaves are arrays.)
                if hasattr(x, "ndim") and x.ndim >= 1:
                    return jp.broadcast_to(x, (B,) + x.shape)
                return x

            data = jax.tree_util.tree_map(_bcast, d0)

            data = data.replace(
                qpos=qpos,
                qvel=jp.zeros((B, nv), dtype=jp.float32),
                ctrl=ctrl,
            )

            data = data.replace(
                mocap_pos=data.mocap_pos.at[:, mocap_id, :].set(target_pos),
                mocap_quat=data.mocap_quat.at[:, mocap_id, :].set(target_quat),
            )

            data = mjx.forward(m, data)
            print("data done")

            # ---- Renderer init once ----
            if self._render_token is None:
                token, _, _ = self.renderer.init(data, self._mjx_model)
                jax.tree_util.tree_map(jax.block_until_ready, token)
                self._render_token = token
            print("rendered")

            render_token = self._render_token

            metrics = {
                "out_of_bounds": jp.zeros((B,), dtype=jp.float32),
                **{k: jp.zeros((B,), dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
            }
            info = {
                "rng": rng_main,
                "target_pos": target_pos,
                "reached_box": jp.zeros((B,), dtype=jp.float32),
                "render_token": render_token,
            }

            info, obs = self._get_obs(data, info)

            reward = jp.zeros((B,), dtype=jp.float32)
            done = jp.zeros((B,), dtype=jp.float32)

            return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
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
        state = State(data, obs, reward, done, metrics, info)

        return state

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]  # [B, 3]

        box_pos = data.xpos[:, self._obj_body, :]
        gripper_pos = data.site_xpos[:, self._gripper_site, :]
        pos_err = jp.linalg.norm(target_pos - box_pos, axis=-1)

        # box_mat / target_mat:
        # mjx typically stores xmat as [B, nbody, 3, 3] or [B, nbody, 9].
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
        )  # [n_geoms, B]
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
        return jp.concatenate([left, right], axis=2)  # [B, H, 2W, 3]

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        pixels = self.render_pixels(info["render_token"], data)
        return info, pixels
