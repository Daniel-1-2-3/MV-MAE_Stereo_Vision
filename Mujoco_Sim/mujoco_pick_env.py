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

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from mujoco.mjx._src import math
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from pathlib import Path

from Custom_Mujoco_Playground._src import mjx_env
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda_kinematics
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import pick
from MAE_Model.prepare_input import Prepare


def _add_assets_from_dir_unique_basename(
    assets: dict[str, bytes], root: Path
) -> dict[str, bytes]:
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
        render_batch_size=1,
        render_width=64,
        render_height=64,
        use_rasterizer=False,
        enabled_geom_groups=[0, 1, 2],
    )


def default_config():
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=200,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            reward_scales=config_dict.create(
                gripper_box=4.0,
                box_target=8.0,
                no_floor_collision=0.25,
                no_box_collision=0.05,
                robot_target_qpos=0.3,
            ),
            # These exist but will be ignored when vision=True (matches pick_cartesian behavior)
            action_rate=-0.0005,
            no_soln_reward=-0.01,
            lifted_reward=0.5,
            success_reward=2.0,
            # exploration aid probability
            guide_swap_prob=0.05,
        ),
        vision=True,
        vision_config=default_vision_config(),
        obs_noise=config_dict.create(brightness=[1.0, 1.0]),
        box_init_range=0.05,
        success_threshold=0.05,
        action_history_length=1,
        impl="jax",
        nconmax=12 * 1024,
        njmax=128,
    )
    return config


def adjust_brightness(img, scale):
    """Adjusts brightness by scaling pixel values (expects [0,1])."""
    return jp.clip(img * scale, 0, 1)


class StereoPickCube(pick.PandaPickCube):
    """Stereo pixels version of PandaPickCube, with Madrona rendering and progress rewards."""

    def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
        self,
        config=default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        self._vision = self._config.vision

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
        self._model_assets = _add_assets_from_dir_unique_basename(self._model_assets, menagerie_dir)

        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        )
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)

        # Set gripper in sight of camera
        self._post_init(obj_name="box", keyframe="low_home")
        self._box_geom = self._mj_model.geom("box").id

        # Geom IDs for distance-based "contacts"
        self._hand_capsule_geom = self._mj_model.geom("hand_capsule").id
        self._left_pad_geom = self._mj_model.geom("left_finger_pad").id
        self._right_pad_geom = self._mj_model.geom("right_finger_pad").id
        self._floor_geom_id = self._mj_model.geom("floor").id

        if self._vision:
            try:
                from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
            except ImportError as e:
                raise ImportError(
                    "Madrona MJX not installed, but vision=True was requested for StereoPickCube."
                ) from e

            self.renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id=self._config.vision_config.gpu_id,
                num_worlds=self._config.vision_config.render_batch_size,
                batch_render_view_width=self._config.vision_config.render_width,
                batch_render_view_height=self._config.vision_config.render_height,
                enabled_geom_groups=np.asarray(self._config.vision_config.enabled_geom_groups, dtype=np.int32),
                enabled_cameras=None,  # render all cameras
                add_cam_debug_geo=False,
                use_rasterizer=self._config.vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

            # Cache render token so we don't re-run renderer.init on every reset.
            self._render_token_cached = None
            self._render_jit = jax.jit(lambda token, data: self.renderer.render(token, data, self._mjx_model))

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._guide_q = self._mj_model.keyframe("picked").qpos
        self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
        self._start_tip_transform = panda_kinematics.compute_franka_fk(self._init_ctrl[:7])
        self._sample_orientation = False

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make finger pads white for visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [idx for idx, data_id in enumerate(mj_model.geom_dataid) if data_id == mesh_id]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        return mj_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state (NOT meant to be jitted)."""
        obs = None

        x_plane = self._start_tip_transform[0, 3] - 0.03  # account for finite gain

        rng, rng_box = jax.random.split(rng)
        r_range = self._config.box_init_range
        box_pos = jp.array(
            [
                x_plane,
                jax.random.uniform(rng_box, (), minval=-r_range, maxval=r_range),
                0.0,
            ]
        )

        target_pos = jp.array([x_plane, 0.0, 0.20])

        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(box_pos)
        )
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        metrics = {
            "out_of_bounds": jp.array(0.0),
            **{f"reward/{k}": jp.array(0.0) for k in self._config.reward_config.reward_scales.keys()},
            "reward/lifted": jp.array(0.0),
            "reward/success": jp.array(0.0),
            "reward/action_rate": jp.array(0.0),
            "truncated": jp.array(0.0),
        }

        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=jp.float32),
            "prev_action": jp.zeros((int(self._mjx_model.nu),), dtype=jp.float32),
            "_steps": jp.array(0, dtype=jp.int32),
            # For reward progress:
            "prev_total_reward": jp.array(0.0, dtype=jp.float32),
        }

        reward = jp.asarray(0.0, jp.float32)
        done = jp.asarray(0.0, jp.float32)

        if self._vision:
            rng_brightness, rng = jax.random.split(rng)
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._config.obs_noise.brightness[0],
                maxval=self._config.obs_noise.brightness[1],
            )
            info = {**info, "brightness": brightness}

            if self._render_token_cached is None:
                render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
                self._render_token_cached = render_token
            else:
                render_token = self._render_token_cached
                _, rgb, _ = self._render_jit(render_token, data)

            info = {**info, "render_token": render_token, "rng": rng}
            img_left = adjust_brightness(jp.asarray(rgb[0, 0, ..., :3], dtype=jp.float32) / 255.0, brightness)
            img_right = adjust_brightness(jp.asarray(rgb[1, 0, ..., :3], dtype=jp.float32) / 255.0, brightness)
            obs = Prepare.fuse_normalize([img_left, img_right])  # (1,H,2W,3)

        assert obs is not None, "vision must be enabled to produce pixel observations"
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Runs one timestep. JIT-safe."""
        info = state.info
        data = state.data

        # Episode-start flag (used for exploration + reward progress reset)
        newly_reset = info["_steps"] == jp.asarray(0, jp.int32)

        # Reset progress-tracking at episode start (important if you auto-reset on done)
        prev_total_reward = jp.where(newly_reset, jp.asarray(0.0, jp.float32), info["prev_total_reward"])
        reached_box = jp.where(newly_reset, jp.asarray(0.0, jp.float32), info["reached_box"])
        prev_action = jp.where(newly_reset, jp.zeros_like(info["prev_action"]), info["prev_action"])

        # (2) Occasionally aid exploration by swapping to guide keyframe at episode start
        rng, key_swap = jax.random.split(info["rng"])
        to_sample = newly_reset & jax.random.bernoulli(key_swap, self._config.reward_config.guide_swap_prob)

        swapped_data = data.replace(qpos=self._guide_q, ctrl=self._guide_ctrl)

        # Mix state data <-> swapped_data only at top-level leaves (same trick as pick_cartesian)
        data = jax.tree_util.tree_map_with_path(
            lambda path, x, y: (
                ((1 - to_sample.astype(x.dtype)) * x + to_sample.astype(x.dtype) * y).astype(x.dtype)
                if len(path) == 1
                else x
            ),
            data,
            swapped_data,
        )

        # Motor-space control: ctrl += action * scale
        delta = action * self._config.action_scale
        ctrl = data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        # Simulator step
        data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

        # Dense reward components (unscaled)
        # NOTE: _get_reward reads info["reached_box"]; we pass the locally-reset value via a temp dict
        tmp_info = {**info, "reached_box": reached_box}
        raw_rewards = self._get_reward(data, tmp_info)
        new_reached_box = raw_rewards["_reached_box"]
        raw_rewards = {k: v for k, v in raw_rewards.items() if k != "_reached_box"}

        # Apply declared reward_scales
        scaled = {
            k: raw_rewards[k] * self._config.reward_config.reward_scales[k]
            for k in self._config.reward_config.reward_scales.keys()
        }
        total_reward = jp.clip(sum(scaled.values()), -1e4, 1e4)

        # (4) For vision policies, skip action_rate / no_soln_reward (matches pick_cartesian intent)
        # You also don't have no_soln in this env, so nothing to add here.
        if not self._vision:
            act_rate = jp.sum(jp.square(action - prev_action))
            total_reward = total_reward + self._config.reward_config.action_rate * act_rate

        # Sparse bonuses
        lifted = self._get_lifted(data).astype(jp.float32)
        total_reward = total_reward + self._config.reward_config.lifted_reward * lifted

        success = self._get_success(data, tmp_info).astype(jp.float32)
        total_reward = total_reward + self._config.reward_config.success_reward * success

        # (1) Reward progress (only pay improvement over best-so-far)
        reward = jp.maximum(total_reward - prev_total_reward, jp.asarray(0.0, jp.float32))
        new_prev_total_reward = jp.maximum(prev_total_reward, total_reward)

        # Termination: OOB / NaNs / time-limit truncation
        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)
        nan_bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        steps = info["_steps"] + jp.asarray(self._config.action_repeat, jp.int32)
        truncated = steps >= jp.asarray(self._config.episode_length, jp.int32)

        done = out_of_bounds | nan_bad | truncated

        # Update info (functional, jittable)
        info = {
            **info,
            "rng": rng,
            "_steps": jp.where(done, jp.asarray(0, jp.int32), steps),
            "prev_action": jp.where(done, jp.zeros_like(prev_action), action.astype(jp.float32)),
            "reached_box": jp.where(done, jp.asarray(0.0, jp.float32), new_reached_box),
            "prev_total_reward": jp.where(done, jp.asarray(0.0, jp.float32), new_prev_total_reward),
        }

        # Render obs
        _, rgb, _ = self._render_jit(info["render_token"], data)
        img_left = adjust_brightness(jp.asarray(rgb[0, 0, ..., :3], dtype=jp.float32) / 255.0, info["brightness"])
        img_right = adjust_brightness(jp.asarray(rgb[1, 0, ..., :3], dtype=jp.float32) / 255.0, info["brightness"])
        obs = Prepare.fuse_normalize([img_left, img_right])

        # Metrics
        metrics = {
            **state.metrics,
            "out_of_bounds": out_of_bounds.astype(jp.float32),
            "truncated": truncated.astype(jp.float32),
            "reward/lifted": lifted.astype(jp.float32),
            "reward/success": success.astype(jp.float32),
        }
        if not self._vision:
            metrics = {**metrics, "reward/action_rate": jp.sum(jp.square(action - prev_action)).astype(jp.float32)}
        else:
            metrics = {**metrics, "reward/action_rate": jp.asarray(0.0, jp.float32)}

        for k in self._config.reward_config.reward_scales.keys():
            metrics = {**metrics, f"reward/{k}": raw_rewards[k].astype(jp.float32)}

        return state.replace(
            data=data,
            obs=obs,
            reward=reward.astype(jp.float32),
            done=done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def _get_lifted(self, data: mjx.Data) -> jax.Array:
        box_z = data.xpos[self._obj_body][2]
        return box_z > jp.asarray(0.05, jp.float32)

    def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        box_pos = data.xpos[self._obj_body]
        target_pos = info["target_pos"]
        if self._vision:
            box_pos, target_pos = box_pos[2], target_pos[2]
        return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]) -> dict[str, Any]:
        target_pos = info["target_pos"]

        box_pos = data.xpos[self._obj_body]
        box_mat = data.xmat[self._obj_body]

        target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
        pos_err = jp.linalg.norm(target_pos - box_pos)
        rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

        hand_pos = data.geom_xpos[self._hand_capsule_geom]
        gripper_dist = jp.linalg.norm(box_pos - hand_pos)

        box_target = 1.0 - jp.tanh(5.0 * (0.9 * pos_err + 0.1 * rot_err))
        gripper_box = 1.0 - jp.tanh(5.0 * gripper_dist)

        obj_adr = self._obj_qposadr
        qpos_robot = jp.concatenate([data.qpos[:obj_adr], data.qpos[obj_adr + 7 :]])
        init_q = jp.array(self._init_q)
        init_q_robot = jp.concatenate([init_q[:obj_adr], init_q[obj_adr + 7 :]])
        robot_target_qpos = 1.0 - jp.tanh(jp.linalg.norm(qpos_robot - init_q_robot))

        lp_z = data.geom_xpos[self._left_pad_geom][2]
        rp_z = data.geom_xpos[self._right_pad_geom][2]
        hc_z = data.geom_xpos[self._hand_capsule_geom][2]
        floor_hit = (lp_z < 0.002) | (rp_z < 0.002) | (hc_z < 0.002)
        no_floor_collision = 1.0 - floor_hit.astype(jp.float32)

        no_box_collision = (gripper_dist > jp.asarray(0.004, jp.float32)).astype(jp.float32)

        reach_thresh = jp.asarray(0.03, jp.float32)
        reached = (gripper_dist < reach_thresh).astype(jp.float32)
        reached_box = jp.maximum(info["reached_box"], reached)

        return {
            "gripper_box": gripper_box,
            "box_target": box_target * reached_box,
            "no_floor_collision": no_floor_collision,
            "no_box_collision": no_box_collision,
            "robot_target_qpos": robot_target_qpos,
            "_reached_box": reached_box,
        }

    @property
    def action_size(self) -> int:
        return int(self._mjx_model.nu)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
