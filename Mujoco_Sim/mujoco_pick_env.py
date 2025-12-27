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
"""A simple task with demonstrating sim2real transfer for pixels observations.
Pick up a cube to a fixed location using a motor-space delta controller (ctrl += action * scale)."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
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

def _add_assets_from_dir(assets: dict[str, bytes], root: Path) -> dict[str, bytes]:
    """Adds every file under `root` into MuJoCo assets.

    We register BOTH:
      - the relative path key (e.g. 'assets/mesh.obj')
      - the basename key (e.g. 'mesh.obj')
    because different XMLs reference files differently.
    """
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        data = p.read_bytes()

        rel_key = p.relative_to(root).as_posix()
        base_key = p.name

        # Prefer existing entries (don't clobber), but fill missing ones.
        assets.setdefault(rel_key, data)
        assets.setdefault(base_key, data)

    return assets

def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=128,
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
            action_rate=-0.0005,
            no_soln_reward=-0.01,
            lifted_reward=0.5,
            success_reward=2.0,
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
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jp.clip(img * scale, 0, 1)


class StereoPickCube(pick.PandaPickCube):
    """Stereo pixels version of PandaPickCube, but with custom XML and pixels obs."""

    def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
        self,
        config=default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        # FIX 1 (minimal): always read merged config from self._config (respects overrides)
        self._vision = self._config.vision

        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )
        self._xml_path = xml_path.as_posix()
        self._model_assets = dict(panda.get_assets())  # copy, in case it's not a plain dict
        menagerie_dir = (
            Path.cwd()
            / "mujoco_playground_external_deps"
            / "mujoco_menagerie"
            / "franka_emika_panda"
        )
        # This is the missing include target directory
        self._model_assets = _add_assets_from_dir(self._model_assets, menagerie_dir)

        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        )
        # FIX 2 (minimal): use self._config (merged) not the incoming `config`
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
        self._floor_hand_geom_ids = [
            self._hand_capsule_geom,
            self._left_pad_geom,
            self._right_pad_geom,
        ]

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
                enabled_geom_groups=np.asarray(self._config.vision_config.enabled_geom_groups),
                enabled_cameras=None,  # use all cameras
                add_cam_debug_geo=False,
                use_rasterizer=self._config.vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._guide_q = self._mj_model.keyframe("picked").qpos
        self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
        self._start_tip_transform = panda_kinematics.compute_franka_fk(self._init_ctrl[:7])
        self._sample_orientation = False

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make the finger pads white for increased visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [idx for idx, data_id in enumerate(mj_model.geom_dataid) if data_id == mesh_id]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        return mj_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        # FIX 3 (minimal): ensure obs always defined; you said vision is always True, so assert
        obs = None

        x_plane = self._start_tip_transform[0, 3] - 0.03  # account for finite gain

        # initialize box position
        rng, rng_box = jax.random.split(rng)
        r_range = self._config.box_init_range
        box_pos = jp.array(
            [
                x_plane,
                jax.random.uniform(rng_box, (), minval=-r_range, maxval=r_range),
                0.0,
            ]
        )

        # fixed target position to simplify pixels-only training
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

        # FIX 4 (minimal + important): always set mocap_pos + mocap_quat so rewards/targets are consistent
        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        metrics = {
            "out_of_bounds": jp.array(0.0),
            **{f"reward/{k}": 0.0 for k in self._config.reward_config.reward_scales.keys()},
            "reward/success": jp.array(0.0),
            "reward/lifted": jp.array(0.0),
        }

        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=float),
            "prev_reward": jp.array(0.0, dtype=float),
            "newly_reset": jp.array(False, dtype=bool),
            "prev_action": jp.zeros((int(self._mjx_model.nu),), dtype=jp.float32),
            "_steps": jp.array(0, dtype=int),
        }

        reward, done = jp.zeros(2)

        if self._vision:
            rng_brightness, rng = jax.random.split(rng)
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._config.obs_noise.brightness[0],
                maxval=self._config.obs_noise.brightness[1],
            )
            info.update({"brightness": brightness})

            render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
            info.update({"render_token": render_token})

            img_left = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
            img_right = jp.asarray(rgb[1][..., :3], dtype=jp.float32) / 255.0
            obs = Prepare.fuse_normalize([img_left, img_right])  # (1, H, 2W, 3)
            obs = adjust_brightness(obs, brightness)

        assert obs is not None, "vision must be enabled to produce pixel observations"
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Runs one timestep of the environment's dynamics."""
        state.info["newly_reset"] = state.info["_steps"] == 0

        newly_reset = state.info["newly_reset"]
        state.info["prev_reward"] = jp.where(newly_reset, 0.0, state.info["prev_reward"])
        state.info["reached_box"] = jp.where(newly_reset, 0.0, state.info["reached_box"])
        state.info["prev_action"] = jp.where(
            newly_reset,
            jp.zeros((int(self._mjx_model.nu),), dtype=jp.float32),
            state.info["prev_action"],
        )

        # Occasionally aid exploration.
        state.info["rng"], key_swap = jax.random.split(state.info["rng"])
        to_sample = newly_reset * jax.random.bernoulli(key_swap, 0.05)
        swapped_data = state.data.replace(qpos=self._guide_q, ctrl=self._guide_ctrl)
        data = jax.tree_util.tree_map_with_path(
            lambda path, x, y: ((1 - to_sample) * x + to_sample * y).astype(x.dtype)
            if len(path) == 1
            else x,
            state.data,
            swapped_data,
        )

        # Motor-space control (PandaPickCube-style): ctrl += action * scale
        delta = action * self._config.action_scale
        ctrl = data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        no_soln = jp.array(0.0, dtype=jp.float32)

        # Simulator step
        data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

        # Dense rewards (base task)
        raw_rewards = self._get_reward(data, state.info)

        # Extra penalties (must exist in reward_scales dict)
        hand_pos = data.geom_xpos[self._hand_capsule_geom]
        box_pos_g = data.geom_xpos[self._box_geom]
        hand_box_dist = jp.linalg.norm(hand_pos - box_pos_g)
        hand_box = hand_box_dist < 0.02
        raw_rewards["no_box_collision"] = jp.where(hand_box, 0.0, 1.0)

        lp_z = data.geom_xpos[self._left_pad_geom][2]
        rp_z = data.geom_xpos[self._right_pad_geom][2]
        hc_z = data.geom_xpos[self._hand_capsule_geom][2]
        floor_hit = (lp_z < 0.002) | (rp_z < 0.002) | (hc_z < 0.002)
        raw_rewards["no_floor_collision"] = jp.where(floor_hit, 0.0, 1.0)

        # Scale everything
        rewards = {
            k: v * self._config.reward_config.reward_scales[k]
            for k, v in raw_rewards.items()
        }
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        if not self._vision:
            # Vision policy cannot access the required state-based observations.
            da = jp.linalg.norm(action - state.info["prev_action"])
            state.info["prev_action"] = action
            total_reward += self._config.reward_config.action_rate * da
            total_reward += no_soln * self._config.reward_config.no_soln_reward

        # Sparse rewards
        box_pos = data.xpos[self._obj_body]
        lifted = (box_pos[2] > 0.05) * self._config.reward_config.lifted_reward
        total_reward += lifted
        success = self._get_success(data, state.info)
        total_reward += success * self._config.reward_config.success_reward

        # Reward progress
        reward = jp.maximum(total_reward - state.info["prev_reward"], jp.zeros_like(total_reward))
        state.info["prev_reward"] = jp.maximum(total_reward, state.info["prev_reward"])
        reward = jp.where(newly_reset, 0.0, reward)  # prevent first-step artifact

        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
        state.metrics.update({f"reward/{k}": v for k, v in raw_rewards.items()})
        state.metrics.update(
            {
                "reward/lifted": lifted.astype(float),
                "reward/success": success.astype(float),
            }
        )

        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | success

        # Ensure exact sync between newly_reset and the autoresetwrapper.
        state.info["_steps"] += self._config.action_repeat
        state.info["_steps"] = jp.where(
            done | (state.info["_steps"] >= self._config.episode_length),
            0,
            state.info["_steps"],
        )

        # Vision obs (always enabled per your guarantee)
        _, rgb, _ = self.renderer.render(state.info["render_token"], data, self._mjx_model)
        img_left = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
        img_right = jp.asarray(rgb[1][..., :3], dtype=jp.float32) / 255.0
        obs = Prepare.fuse_normalize([img_left, img_right])
        obs = adjust_brightness(obs, state.info["brightness"])

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(float),
            info=state.info,
        )

    def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        box_pos = data.xpos[self._obj_body]
        target_pos = info["target_pos"]
        if self._vision:  # randomized camera positions cannot see location along y line
            box_pos, target_pos = box_pos[2], target_pos[2]
        return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

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
