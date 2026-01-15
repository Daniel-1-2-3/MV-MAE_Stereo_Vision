# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0
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
        use_rasterizer=False,  # False => raytracer
        enabled_geom_groups=[0, 1, 2],
        enabled_cameras=[0, 1],
    )


def default_config():
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
        obs_noise=config_dict.create(brightness=[1.0, 1.0]),
        impl="jax",
        nconmax=24 * 2048,
        njmax=128,
    )


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

        # Build MuJoCo model ourselves (your PandaBase __init__ builds its own model;
        # we are not calling PandaBase.__init__ directly in your original pick_env either.)
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

        # MuJoCo model
        mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(),
            assets=self._model_assets,
        )
        mj_model = self.modify_model(mj_model)
        mj_model.opt.timestep = self._config.sim_dt
        self._mj_model: mujoco.MjModel = mj_model

        # IMPORTANT: this populates ONLY the vars you showed in panda.py
        self._post_init(obj_name="box", keyframe="low_home")

        # MJX model from final MuJoCo model
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Cache geom ids (all exist per your PandaBase._post_init)
        self._hand_geoms = jp.asarray(
            [int(self._left_finger_geom), int(self._right_finger_geom), int(self._hand_geom)],
            dtype=jp.int32,
        )
        self._floor_geom_id = jp.int32(int(self._floor_geom))

        # Renderer
        self.renderer: BatchRenderer = self._create_renderer()
        self._render_token: Optional[jax.Array] = None

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._sample_orientation = False

    def _create_renderer(self) -> BatchRenderer:
        vc = self._config.vision_config
        enabled_geom_groups = np.asarray(vc.enabled_geom_groups, dtype=np.int32, order="C")
        enabled_cameras = np.asarray(getattr(vc, "enabled_cameras", [0, 1]), dtype=np.int32, order="C")

        if self._mj_model.ncam <= int(np.max(enabled_cameras)):
            raise ValueError(
                f"enabled_cameras={enabled_cameras.tolist()} but mj_model.ncam={self._mj_model.ncam}."
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

    # ---------------- micro-style batched Data ----------------
    def _make_batched_data_safe(self, B: int) -> mjx.Data:
        m = self._mjx_model
        return jax.vmap(lambda _: mjx.make_data(m))(jp.arange(B))

    def _maybe_init_renderer(self, data_for_init: mjx.Data, debug: bool):
        if self._render_token is not None:
            return

        B = int(data_for_init.qpos.shape[0]) if data_for_init.qpos.ndim >= 2 else 1
        if B != self.render_batch_size:
            raise ValueError(f"Renderer num_worlds={self.render_batch_size} but init data B={B}.")

        # Micro-repro does forward() before init
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

    # ---------------- render (must stay OUTSIDE jit for no-warning path) ----------------
    def render_pixels(self, render_token: jax.Array, data_batched: mjx.Data) -> jax.Array:
        _, rgb, _ = self.renderer.render(render_token, data_batched, self._mj_model)

        B = int(data_batched.qpos.shape[0]) if data_batched.qpos.ndim >= 2 else 1

        if rgb.ndim == 5:
            left = rgb[:, 0]
            right = rgb[:, 1]
        elif rgb.ndim == 4 and rgb.shape[0] == 2 and B == 1:
            left = rgb[0:1]
            right = rgb[1:2]
        else:
            raise ValueError(f"Unrecognized rgb shape {tuple(rgb.shape)} for B={B}")

        left = left[..., :3].astype(jp.float32) / 255.0
        right = right[..., :3].astype(jp.float32) / 255.0
        return jp.concatenate([left, right], axis=2)

    def compute_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        pixels = self.render_pixels(info["render_token"], data)
        if "brightness" in info:
            pixels = adjust_brightness(pixels, info["brightness"])
        return pixels

    # ---------------- physics-only (jit-safe) ----------------
    def reset_physics(self, rng: jax.Array) -> State:
        if rng.ndim == 1:
            rng = rng[None, :]
        if rng.shape[0] != self.render_batch_size:
            rng = jax.random.split(rng[0], self.render_batch_size)

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        dev = next(iter(rng.devices())) if hasattr(rng, "devices") else None

        with (jax.default_device(dev) if dev is not None else contextlib.nullcontext()):
            B = int(rng.shape[0])
            m = self._mjx_model

            keys = jax.vmap(lambda k: jax.random.split(k, 4))(rng)
            rng_main = keys[:, 0, :]
            rng_box = keys[:, 1, :]
            rng_target = keys[:, 2, :]
            rng_brightness = keys[:, 3, :]

            # sample box/target
            min_box = jp.array([-0.2, -0.2, 0.0], dtype=jp.float32)
            max_box = jp.array([0.2, 0.2, 0.0], dtype=jp.float32)
            min_tgt = jp.array([-0.2, -0.2, 0.2], dtype=jp.float32)
            max_tgt = jp.array([0.2, 0.2, 0.4], dtype=jp.float32)
            base = jp.asarray(self._init_obj_pos, dtype=jp.float32)

            def _sample_pos(key, mn, mx):
                return jax.random.uniform(key, (3,), minval=mn, maxval=mx) + base

            box_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_box, min_box, max_box)
            target_pos = jax.vmap(_sample_pos, in_axes=(0, None, None))(rng_target, min_tgt, max_tgt)

            # make safe batched Data
            data = self._make_batched_data_safe(B)
            self._maybe_init_renderer(data, debug=debug)

            # write init qpos/ctrl
            nq = int(m.nq)
            nv = int(m.nv)

            init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[:nq]
            qpos = jp.broadcast_to(init_q0[None, :], (B, nq)).astype(jp.float32)
            qvel = jp.zeros((B, nv), dtype=jp.float32)
            ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
            ctrl = jp.broadcast_to(ctrl0[None, ...], (B,) + ctrl0.shape).astype(jp.float32)

            data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

            # set box position into qpos slice defined by PandaBase
            data = data.replace(
                qpos=data.qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(box_pos.astype(jp.float32))
            )

            # set mocap target position (use PandaBase._mocap_target)
            mpos = data.mocap_pos
            if mpos.ndim == 2:
                mpos = jp.broadcast_to(mpos[None, ...], (B,) + mpos.shape)
            mpos = mpos.at[:, int(self._mocap_target), :].set(target_pos.astype(jp.float32))
            data = data.replace(mocap_pos=mpos)

            # forward
            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            # brightness
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

            obs0 = jp.zeros((B, self.render_height, 2 * self.render_width, 3), dtype=jp.float32)
            reward = jp.zeros((B,), dtype=jp.float32)
            done = jp.zeros((B,), dtype=jp.float32)
            return State(data, obs0, reward, done, metrics, info)

    def step_physics(self, state: State, action: jax.Array) -> State:
        # ctrl update
        delta = action * self._config.action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        # FIX: vmap per-world so mjx.step sees single-world Data
        def _one(d: mjx.Data, u: jax.Array) -> mjx.Data:
            return mjx_env.step(self._mjx_model, d, u, self.n_substeps)

        data = jax.vmap(_one, in_axes=(0, 0))(state.data, ctrl)

        info, raw = self._get_reward_distance(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[:, int(self._obj_body), :]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0, axis=-1) | (box_pos[:, 2] < 0.0)

        metrics = {
            **state.metrics,
            "out_of_bounds": out_of_bounds.astype(jp.float32),
            **raw,
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

    # Legacy API (used by older code). Wrapper will prefer *_physics + compute_obs.
    def reset(self, rng: jax.Array) -> State:
        st = self.reset_physics(rng)
        obs = self.compute_obs(st.data, st.info)
        return st.replace(obs=obs)

    def step(self, state: State, action: jax.Array) -> State:
        st = self.step_physics(state, action)
        obs = self.compute_obs(st.data, st.info)
        return st.replace(obs=obs)

    # ---------------- distance-based reward ----------------
    def _get_reward_distance(self, data: mjx.Data, info: dict[str, Any]):
        B = int(data.qpos.shape[0])
        obj_id = int(self._obj_body)

        box_pos = data.xpos[:, obj_id, :]
        target_pos = info["target_pos"]

        # gripper position via site (PandaBase gives _gripper_site but not used elsewhere;
        # mjx.Data has site_xpos in newer mjx; if missing, fall back to hand geom.)
        if hasattr(data, "site_xpos"):
            grip = data.site_xpos[:, int(self._gripper_site), :]
        else:
            # fallback: use hand geom position if geom_xpos exists
            if hasattr(data, "geom_xpos"):
                grip = data.geom_xpos[:, int(self._hand_geom), :]
            else:
                grip = box_pos  # last resort; keeps shapes valid

        d_gripper_box = jp.linalg.norm(grip - box_pos, axis=-1)
        d_box_target = jp.linalg.norm(box_pos - target_pos, axis=-1)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05), 1.0)
        info = {**info, "reached_box": reached_box}

        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        # --- floor collision: check hand geoms vs floor geom height ---
        if hasattr(data, "geom_xpos"):
            floor_z = data.geom_xpos[:, int(self._floor_geom_id), 2]
            hand_z = jp.take(data.geom_xpos[..., 2], self._hand_geoms, axis=1)  # (B,3)
            floor_clearance = jp.float32(0.004)
            floor_coll = jp.any(hand_z < (floor_z[:, None] + floor_clearance), axis=1)
        else:
            floor_coll = jp.zeros((B,), dtype=jp.bool_)

        no_floor_collision = jp.where(floor_coll, 0.0, 1.0)

        # --- box collision: use distance threshold (minimal, sensorless) ---
        box_touch_dist = jp.float32(0.030)
        box_touch = d_gripper_box < box_touch_dist
        no_box_collision = jp.where(box_touch, 0.0, 1.0)

        raw = dict(
            gripper_box=gripper_box,
            box_target=box_target,
            no_floor_collision=no_floor_collision,
            no_box_collision=no_box_collision,
            robot_target_qpos=jp.zeros_like(gripper_box),
        )
        return info, raw

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size so Madrona renders it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]
        # Make the finger pads white for visibility
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