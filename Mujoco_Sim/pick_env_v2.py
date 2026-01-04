# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0

"""StereoPickCube (single-world physics; renderer call pattern matches debug.py exactly).

Key changes to match debug.py:
- Env NO LONGER constructs BatchRenderer; wrapper owns it (like debug.py).
- mjx.put_model is called without impl=... (debug.py style).
- We optionally warm MJX with mjx.make_data(model) before the wrapper builds the renderer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jp
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
        if p.name in used_basenames:
            continue
        assets[p.relative_to(root).as_posix()] = p.read_bytes()
        used_basenames.add(p.name)
    return assets


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=128,  # must equal num_envs in wrapper
        render_width=64,
        render_height=64,  # kept for compatibility; debug.py effectively uses width for both args
        use_rasterizer=False,
        enabled_geom_groups=[0, 1, 2],
        add_cam_debug_geo=False,
    )


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
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
        impl="jax",  # kept, but we do NOT pass impl into mjx.put_model (debug.py style)
        nconmax=24 * 2048,
        njmax=128,
    )


def _body_xpos(data: mjx.Data, body_id: int) -> jax.Array:
    return data.xpos[body_id]


class StereoPickCube(panda.PandaBase):
    """Bring a box to a target (single-world physics; batching handled by wrapper)."""

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

        # DO NOT call super().__init__()
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

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

        # Build MuJoCo model
        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=self._model_assets)
        )
        mj_model.opt.timestep = float(self._config.sim_dt)

        self._mj_model: mujoco.MjModel = mj_model

        # MATCH debug.py: no impl=...
        self._mjx_model: mjx.Model = mjx.put_model(mj_model)

        # Panda task init can mutate model; re-upload afterwards
        self._post_init(obj_name="box", keyframe="low_home")

        # IDs used by rewards
        hand_geom_id = self._mj_model.geom("hand_capsule").id
        self._hand_body = int(self._mj_model.geom_bodyid[hand_geom_id])

        # Sensors
        sensor_names = [self._mj_model.sensor(i).name for i in range(self._mj_model.nsensor)]
        floor_ids = [
            i for i, n in enumerate(sensor_names)
            if ("floor" in n.lower()) and any(k in n.lower() for k in ("hand", "finger", "gripper"))
        ]
        if not floor_ids:
            floor_ids = [i for i, n in enumerate(sensor_names) if "floor" in n.lower()]

        box_hand_ids = [
            i for i, n in enumerate(sensor_names)
            if any(k in n.lower() for k in ("box", "cube", "object"))
            and any(k in n.lower() for k in ("hand", "finger", "gripper"))
        ]
        if not box_hand_ids:
            box_hand_ids = [
                i for i, n in enumerate(sensor_names)
                if any(k in n.lower() for k in ("hand", "finger", "gripper"))
            ]

        self._floor_hand_found_sensor = [int(i) for i in floor_ids]
        self._box_hand_found_sensor = int(box_hand_ids[0]) if box_hand_ids else -1

        # MATCH debug.py: touch mjx.make_data(model) before renderer init (warm-up)
        _ = mjx.make_data(self._mjx_model)

    def _post_init(self, obj_name: str, keyframe: str):
        super()._post_init(obj_name, keyframe)
        # MATCH debug.py: re-upload without impl=...
        self._mjx_model = mjx.put_model(self._mj_model)
        self._sample_orientation = False

    # -------------------------
    # Debug.py EXACT renderer construction helper (wrapper owns renderer)
    # -------------------------

    def make_renderer_debug(self, num_worlds: int) -> BatchRenderer:
        """Construct BatchRenderer with the exact positional args used by debug.py."""
        vc = self._config.vision_config
        view_w = int(getattr(vc, "render_width", self.render_width))

        # debug.py passes np.array([0,1,2]) without dtype cast
        enabled_geom_groups = np.array(getattr(vc, "enabled_geom_groups", [0, 1, 2]))

        return BatchRenderer(
            self._mjx_model,
            int(getattr(vc, "gpu_id", 0)),
            int(num_worlds),
            int(view_w),
            int(view_w),  # debug.py passes width twice
            enabled_geom_groups,
            None,  # enabled_cameras
            bool(getattr(vc, "add_cam_debug_geo", False)),
            bool(getattr(vc, "use_rasterizer", False)),
            None,  # viz_gpu_hdls
        )

    # -------------------------
    # Physics (single-world)
    # -------------------------

    def reset(self, rng: jax.Array) -> State:
        m = self._mjx_model
        rng, rng_box, rng_target = jax.random.split(rng, 3)

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
        data = data.replace(
            qpos=data.qpos.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)
        )

        # Mocap target
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
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=jp.float32),
            "truncation": jp.array(0.0, dtype=jp.float32),
        }

        obs = jp.zeros((1,), dtype=jp.float32)  # pixels handled in wrapper
        reward = jp.array(0.0, dtype=jp.float32)
        done = jp.array(False)

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action_scale = float(self._config.action_scale)
        delta = action * action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = _body_xpos(data, self._obj_body)
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)

        new_metrics = dict(state.metrics)
        new_metrics["out_of_bounds"] = out_of_bounds.astype(jp.float32)
        for k, v in raw_rewards.items():
            new_metrics[k] = v

        done = out_of_bounds | jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        return state.replace(
            data=data,
            obs=state.obs,
            reward=total_reward,
            done=done.astype(jp.float32),
            metrics=new_metrics,
            info=info,
        )

    # -------------------------
    # Rewards / model tweaks
    # -------------------------

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]):
        box_pos = _body_xpos(data, self._obj_body)
        target_pos = info["target_pos"]

        gripper_pos = _body_xpos(data, self._hand_body)
        d_gripper_box = jp.linalg.norm(gripper_pos - box_pos)
        d_box_target = jp.linalg.norm(box_pos - target_pos)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05), 1.0)
        info = {**info, "reached_box": reached_box}

        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        floor_coll = jp.array(0.0, dtype=jp.float32)
        for sensor_id in self._floor_hand_found_sensor:
            floor_coll = floor_coll + (
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            ).astype(jp.float32)
        no_floor_collision = jp.where(floor_coll > 0, 0.0, 1.0)

        if self._box_hand_found_sensor >= 0:
            hand_box = data.sensordata[self._mj_model.sensor_adr[self._box_hand_found_sensor]] > 0
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
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [i for i, data_id in enumerate(mj_model.geom_dataid) if data_id == mesh_id]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        return mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
