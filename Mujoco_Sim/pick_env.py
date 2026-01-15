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
        use_rasterizer=False,  # False => raytracer in madrona_mjx
        enabled_geom_groups=[0, 1, 2],
        enabled_cameras=[0, 1],
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

        # ---------------------------------------------------------------------
        # 1) Build MuJoCo model
        # ---------------------------------------------------------------------
        mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(),
            assets=self._model_assets,
        )
        mj_model = self.modify_model(mj_model)
        mj_model.opt.timestep = self._config.sim_dt
        self._mj_model: mujoco.MjModel = mj_model

        # ---------------------------------------------------------------------
        # 2) IMPORTANT: PandaBase post-init must run BEFORE we read any PandaBase fields
        # ---------------------------------------------------------------------
        self._post_init(obj_name="box", keyframe="low_home")

        # ---------------------------------------------------------------------
        # 3) Freeze the MJX model from the final MuJoCo model
        # ---------------------------------------------------------------------
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # ---------------------------------------------------------------------
        # 4) Now it is SAFE to read PandaBase-derived fields. Build ids/indices.
        # ---------------------------------------------------------------------
        self._floor_hand_geom_ids = [
            self._mj_model.geom(geom).id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]
        self._floor_geom_id = self._mj_model.geom("floor").id

        # Cache sensor_adr indices (jit-friendly)
        # These are created by PandaBase._post_init(...)
        self._floor_hand_sensor_adrs = jp.asarray(
            [int(self._mj_model.sensor_adr[sid]) for sid in self._floor_hand_found_sensor],
            dtype=jp.int32,
        )
        self._box_hand_sensor_adr = jp.int32(int(self._mj_model.sensor_adr[self._box_hand_found_sensor]))

        # ---------------------------------------------------------------------
        # 5) Create renderer (micro-repro style)
        # ---------------------------------------------------------------------
        self.renderer: BatchRenderer = self._create_renderer()
        self._render_token: Optional[jax.Array] = None

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._sample_orientation = False

    def _create_renderer(self) -> BatchRenderer:
        vc = self._config.vision_config

        enabled_geom_groups = np.asarray(vc.enabled_geom_groups, dtype=np.int32, order="C")
        cams = getattr(vc, "enabled_cameras", None)
        if cams is None:
            cams = [0, 1]
        enabled_cameras = np.asarray(cams, dtype=np.int32, order="C")

        if self._mj_model.ncam <= int(np.max(enabled_cameras)):
            raise ValueError(
                f"enabled_cameras={enabled_cameras.tolist()} but mj_model.ncam={self._mj_model.ncam}. "
                "Your XML likely has fewer cameras than expected."
            )

        return BatchRenderer(
            m=self._mjx_model,
            gpu_id=int(vc.gpu_id),
            num_worlds=int(self.render_batch_size),
            batch_render_view_width=int(self.render_width),
            batch_render_view_height=int(self.render_height),
            enabled_geom_groups=enabled_geom_groups,
            enabled_cameras=enabled_cameras,
            add_cam_debug_geo=False,
            use_rasterizer=bool(vc.use_rasterizer),
            viz_gpu_hdls=None,
        )

    # -------------------------------------------------------------------------
    # Micro-repro style: make B independent Data objects
    # -------------------------------------------------------------------------
    def _make_batched_data_safe(self, B: int) -> mjx.Data:
        m = self._mjx_model
        return jax.vmap(lambda _: mjx.make_data(m))(jp.arange(B))

    def _apply_per_world_initial_state(
        self,
        data: mjx.Data,
        box_pos: jax.Array,        # (B, 3)
        target_pos: jax.Array,     # (B, 3)
        rng_robot: jax.Array,      # (B, 2) keys
    ) -> mjx.Data:
        m = self._mjx_model
        B = int(box_pos.shape[0])

        nq = int(m.nq)
        nv = int(m.nv)

        init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[:nq]
        qpos = jp.broadcast_to(init_q0[None, :], (B, nq)).astype(jp.float32)
        qvel = jp.zeros((B, nv), dtype=jp.float32)

        ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
        ctrl = jp.broadcast_to(ctrl0[None, ...], (B,) + ctrl0.shape).astype(jp.float32)

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        eps = jp.float32(0.01)
        qpos_noise = jax.vmap(
            lambda k: jax.random.uniform(k, (nq,), minval=-eps, maxval=eps)
        )(rng_robot).astype(jp.float32)

        data = data.replace(
            qpos=data.qpos.at[:, : self._obj_qposadr].add(qpos_noise[:, : self._obj_qposadr])
        )

        data = data.replace(
            qpos=data.qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(
                box_pos.astype(jp.float32)
            )
        )

        target_quat0 = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        target_quat = jp.broadcast_to(target_quat0[None, :], (B, 4))

        mpos = data.mocap_pos
        mquat = data.mocap_quat

        if mpos.ndim == 2:
            mpos = jp.broadcast_to(mpos[None, ...], (B,) + mpos.shape)
        if mquat.ndim == 2:
            mquat = jp.broadcast_to(mquat[None, ...], (B,) + mquat.shape)

        mpos = mpos.at[:, 0, :].set(target_pos.astype(jp.float32))
        mquat = mquat.at[:, 0, :].set(target_quat)

        return data.replace(mocap_pos=mpos, mocap_quat=mquat)

    def _maybe_init_renderer(self, data_for_init: mjx.Data, debug: bool):
        if self._render_token is not None:
            return

        B = int(data_for_init.qpos.shape[0]) if data_for_init.qpos.ndim >= 2 else 1
        if B != self.render_batch_size:
            raise ValueError(
                f"Renderer num_worlds={self.render_batch_size} but init data batch B={B}. "
                "Keep them equal."
            )

        if debug:
            backend = jax.lib.xla_bridge.get_backend()
            print("[diag] jax/jaxlib:", jax.__version__, jaxlib.__version__)
            print("[diag] backend:", backend.platform, "|", getattr(backend, "platform_version", None))
            print("[diag] JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))

        m = self._mjx_model
        data_for_init = jax.vmap(lambda d: mjx.forward(m, d))(data_for_init)

        if debug:
            print("[diag] calling renderer.init ...")
        tok, _, _ = self.renderer.init(data_for_init, self._mj_model)
        jax.block_until_ready(tok)
        self._render_token = tok

        if debug:
            _, rgb, _ = self.renderer.render(tok, data_for_init, self._mj_model)
            jax.block_until_ready(rgb)
            print("[diag] smoke render OK:", tuple(rgb.shape), rgb.dtype)

    # -------------------------------------------------------------------------
    # Rendering (keep OUTSIDE jit)
    # -------------------------------------------------------------------------
    def render_pixels(self, render_token: jax.Array, data_batched: mjx.Data) -> jax.Array:
        _, rgb, _ = self.renderer.render(render_token, data_batched, self._mj_model)

        B = int(data_batched.qpos.shape[0]) if data_batched.qpos.ndim >= 2 else 1

        if rgb.ndim == 5:
            left = rgb[:, 0]
            right = rgb[:, 1]
        elif rgb.ndim == 4:
            # single-world stereo
            if rgb.shape[0] == 2 and B == 1:
                left = rgb[0:1]
                right = rgb[1:2]
            else:
                raise ValueError(f"Unrecognized rgb shape {tuple(rgb.shape)} for B={B}")
        else:
            raise ValueError(f"Unrecognized rgb ndim={rgb.ndim}, shape={tuple(rgb.shape)}")

        left = left[..., :3].astype(jp.float32) / 255.0
        right = right[..., :3].astype(jp.float32) / 255.0
        pixels = jp.concatenate([left, right], axis=2)
        return pixels

    def compute_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        pixels = self.render_pixels(info["render_token"], data)
        if "brightness" in info:
            pixels = adjust_brightness(pixels, info["brightness"])
        return pixels

    # -------------------------------------------------------------------------
    # Physics-only APIs (safe to jit)
    # -------------------------------------------------------------------------
    def reset_physics(self, rng: jax.Array) -> State:
        if rng.ndim == 1:
            rng = rng[None, :]
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        dev = next(iter(rng.devices())) if hasattr(rng, "devices") else None

        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            m = self._mjx_model
            B = int(rng.shape[0])

            if debug:
                print(f"[pick_env] reset_physics: B={B} render_batch_size={self.render_batch_size}")
                if dev is not None:
                    print(f"[pick_env] reset device: {dev}")

            keys = jax.vmap(lambda k: jax.random.split(k, 5))(rng)
            rng_main = keys[:, 0, :]
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]
            rng_brightness = keys[:, 3, :]
            rng_robot = keys[:, 4, :]

            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)

            data = self._make_batched_data_safe(B)
            self._maybe_init_renderer(data, debug=debug)

            data = self._apply_per_world_initial_state(
                data=data,
                box_pos=box_pos,
                target_pos=target_pos,
                rng_robot=rng_robot,
            )
            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            bmin, bmax = self._config.obs_noise.brightness
            brightness = jax.vmap(
                lambda k: jax.random.uniform(k, (1,), minval=bmin, maxval=bmax)
            )(rng_brightness).reshape((B, 1, 1, 1))

            metrics = {
                "out_of_bounds": jp.zeros((B,), dtype=jp.float32),
                **{k: jp.zeros((B,), dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
            }
            info = {
                "rng": rng_main,
                "target_pos": target_pos,
                "reached_box": jp.zeros((B,), dtype=jp.float32),
                "truncation": jp.zeros((B,), dtype=jp.float32),
                "render_token": self._render_token,
                "brightness": brightness,
            }

            # placeholder obs, filled outside jit via compute_obs()
            obs0 = jp.zeros((B, self.render_height, 2 * self.render_width, 3), dtype=jp.float32)
            reward = jp.zeros((B,), dtype=jp.float32)
            done = jp.zeros((B,), dtype=jp.float32)
            return State(data, obs0, reward, done, metrics, info)

    def step_physics(self, state: State, action: jax.Array) -> State:
        """
        Physics-only step.
        Fixes the crash by stepping per-world so mjx.step sees single-world Data.
        """
        delta = action * self._config.action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        def _one(d: mjx.Data, u: jax.Array) -> mjx.Data:
            return mjx_env.step(self._mjx_model, d, u, self.n_substeps)

        data = jax.vmap(_one, in_axes=(0, 0))(state.data, ctrl)

        info, raw_rewards = self._get_reward_batched(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[:, self._obj_body, :]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0, axis=-1) | (box_pos[:, 2] < 0.0)

        metrics = {
            **state.metrics,
            "out_of_bounds": out_of_bounds.astype(jp.float32),
            **{k: v for k, v in raw_rewards.items()},
        }

        done = out_of_bounds | jp.isnan(data.qpos).any(axis=-1) | jp.isnan(data.qvel).any(axis=-1)

        return state.replace(
            data=data,
            obs=state.obs,
            reward=total_reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    # Keep old API for non-wrapper callers
    def reset(self, rng: jax.Array) -> State:
        st = self.reset_physics(rng)
        obs = self.compute_obs(st.data, st.info)
        return st.replace(obs=obs)

    def step(self, state: State, action: jax.Array) -> State:
        st = self.step_physics(state, action)
        obs = self.compute_obs(st.data, st.info)
        return st.replace(obs=obs)

    # -------------------------------------------------------------------------
    # Batch-safe reward
    # -------------------------------------------------------------------------
    def _get_reward_batched(self, data: mjx.Data, info: dict[str, Any]):
        box_pos = data.xpos[:, self._obj_body, :]
        target_pos = info["target_pos"]

        gripper_pos = data.xpos[:, self._hand_body, :]
        d_gripper_box = jp.linalg.norm(gripper_pos - box_pos, axis=-1)
        d_box_target = jp.linalg.norm(box_pos - target_pos, axis=-1)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05), 1.0)
        info = {**info, "reached_box": reached_box}

        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        # sensordata: (B, nsensordata)
        floor_vals = jp.take(data.sensordata, self._floor_hand_sensor_adrs, axis=-1)
        floor_coll = jp.sum((floor_vals > 0).astype(jp.float32), axis=-1)
        no_floor_collision = jp.where(floor_coll > 0, 0.0, 1.0)

        hand_box_val = jp.take(data.sensordata, self._box_hand_sensor_adr, axis=-1)
        hand_box = hand_box_val > 0
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
