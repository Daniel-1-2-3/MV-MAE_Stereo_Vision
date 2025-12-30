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

This version is structured to keep MJX physics + reward fully jittable, while
**never** jitting Madrona raytracing calls. Rendering is exposed via
`init_render(...)` and `render_pixels(...)` and is intended to be called from the
wrapper *outside* any jax.jit region.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math

from Custom_Mujoco_Playground._src import mjx_env
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda_kinematics
from Custom_Mujoco_Playground._src.mjx_env import State  # pylint: disable=g-importing-member

from madrona_mjx.renderer import BatchRenderer  # type: ignore

import numpy as np
from pathlib import Path


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
        use_rasterizer=False,  # False => raytracer
        enabled_geom_groups=[0, 1, 2],
    )


def default_config() -> config_dict.ConfigDict:
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

        # Renderer object exists, but is only called OUTSIDE jit via init_render/render_pixels.
        self._init_renderer()

    def _post_init(self, obj_name="box", keyframe="low_home"):
        super()._post_init(obj_name, keyframe)
        self._guide_q = self._mj_model.keyframe("picked").qpos
        self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
        self._start_tip_transform = panda_kinematics.compute_franka_fk(self._init_ctrl[:7])

    def _init_renderer(self):
        self.renderer = BatchRenderer(
            m=self._mjx_model,
            gpu_id=self._config.vision_config.gpu_id,
            num_worlds=self.render_batch_size,
            batch_render_view_width=self.render_width,
            batch_render_view_height=self.render_height,
            enabled_geom_groups=np.asarray(
                self._config.vision_config.enabled_geom_groups, dtype=np.int32
            ),
            enabled_cameras=None,
            add_cam_debug_geo=False,
            use_rasterizer=self._config.vision_config.use_rasterizer,
            viz_gpu_hdls=None,
        )

        # Warm-up init once (outside jit). Token is NOT stored here.
        data0 = mjx_env.make_data(
            self._mj_model,
            qpos=jp.array(self._init_q, dtype=jp.float32),
            qvel=jp.zeros(self._mjx_model.nv, dtype=jp.float32),
            ctrl=jp.array(self._init_ctrl, dtype=jp.float32),
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        B = self.render_batch_size

        def _batch(x):
            # Batch any array-like leaves (including numpy arrays), leave pure scalars/static alone.
            if x is None or isinstance(x, (bool, int, float, str, bytes)):
                return x
            if hasattr(x, "dtype") and hasattr(x, "shape") and hasattr(x, "ndim"):
                x = jp.asarray(x)
                if x.ndim == 0:
                    return jp.broadcast_to(x, (B,))
                return jp.broadcast_to(x, (B,) + x.shape)
            return x

        data0_batched = jax.tree_util.tree_map(_batch, data0)
        _ = self.renderer.init(data0_batched, self._mjx_model)

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

    # ----------------------------
    # JIT-SAFE physics + reward
    # ----------------------------

    def _has_contact_with_floor(self, data: mjx.Data, geom_id: int) -> jax.Array:
        g1 = data.contact.geom1          # [B, nconmax]
        g2 = data.contact.geom2
        idx = jp.arange(g1.shape[-1])    # [nconmax]
        valid = idx[None, :] < data.ncon[:, None]   # [B, nconmax]

        pair = jp.logical_or(
            jp.logical_and(g1 == geom_id, g2 == self._floor_geom_id),
            jp.logical_and(g2 == geom_id, g1 == self._floor_geom_id),
        )
        return jp.any(pair & valid, axis=-1)

    def reset(self, rng: jax.Array) -> State:
        """Jittable reset: builds MJX state and returns dummy pixels.

        Rendering (raytracer) is intentionally NOT called here.
        """
        rng, rng_box, rng_target = jax.random.split(rng, 3)

        B = rng.shape[0]
        box_pos = (
            jax.random.uniform(
                rng_box,
                (B, 3),
                minval=jp.array([-0.2, -0.2, 0.0]),
                maxval=jp.array([0.2, 0.2, 0.0]),
            )
            + self._init_obj_pos
        )

        target_pos = (
            jax.random.uniform(
                rng_target,
                (B, 3),
                minval=jp.array([-0.2, -0.2, 0.2]),
                maxval=jp.array([0.2, 0.2, 0.4]),
            )
            + self._init_obj_pos
        )

        init_q0 = jp.array(self._init_q, dtype=jp.float32)
        init_q = jp.broadcast_to(init_q0, (B,) + init_q0.shape)
        init_q = init_q.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)

        qvel = jp.zeros((B, self._mjx_model.nv), dtype=jp.float32)

        ctrl0 = jp.array(self._init_ctrl, dtype=jp.float32)
        ctrl = jp.broadcast_to(ctrl0, (B,) + ctrl0.shape)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=qvel,
            ctrl=ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        target_quat0 = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        target_quat = jp.broadcast_to(target_quat0, (B, 4))
        data = data.replace(
            mocap_pos=data.mocap_pos.at[:, self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[:, self._mocap_target, :].set(target_quat),
        )

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.asarray(0.0, jp.float32),
        }

        # Dummy pixels placeholder; wrapper will overwrite with real raytraced pixels.
        obs = jp.zeros((B, self.render_height, 2 * self.render_width, 3), dtype=jp.uint8)

        reward, done = jp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Jittable step: physics + reward only. Pixels are updated in wrapper."""
        delta = action * self._action_scale
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

        # Keep placeholder pixels (wrapper will overwrite with real render).
        obs = state.obs
        return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]  # [B, 3]

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
                data.qpos[:, self._robot_arm_qposadr] - self._init_q[self._robot_arm_qposadr],
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

    def init_render(self, data_batched: mjx.Data):
        """Call OUTSIDE jit. Returns a render token."""
        render_token, _, _ = self.renderer.init(data_batched, self._mjx_model)
        return render_token

    def render_pixels(self, render_token, data_batched: mjx.Data) -> jax.Array:
        """Call OUTSIDE jit. Returns uint8 [B, H, 2W, 3]."""
        _, rgb, _ = self.renderer.render(render_token, data_batched, self._mjx_model)
        left = jp.asarray(rgb[0][..., :3], dtype=jp.uint8)
        right = jp.asarray(rgb[1][..., :3], dtype=jp.uint8)
        fused = jp.concatenate([left, right], axis=2)
        return fused