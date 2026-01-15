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
        # IMPORTANT: make cameras explicit like the micro-repro.
        # If your XML has two cameras, they are usually indices 0 and 1.
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
        # 2) PandaBase post-init may mutate the MuJoCo model. Do it BEFORE mjx.put_model
        # ---------------------------------------------------------------------
        # NOTE: PandaBase._post_init often sets up _init_q, _init_ctrl, _obj_qposadr, etc.
        # It can also mutate the model. So we call it now, THEN create mjx_model afterwards.
        self._post_init(obj_name="box", keyframe="low_home")

        # ---------------------------------------------------------------------
        # 3) Now freeze the MJX model from the final MuJoCo model
        # ---------------------------------------------------------------------
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Convenience ids
        self._floor_hand_geom_ids = [
            self._mj_model.geom(geom).id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._floor_clearance_margin = jp.float32(0.01)  # "touching floor" if within 1cm
        self._hand_box_margin = jp.float32(0.05)         # "touching box" if within 5cm
        # ---------------------------------------------------------------------
        # 4) Create renderer from the FINAL mjx_model and mj_model (micro-repro style)
        # ---------------------------------------------------------------------
        self.renderer: BatchRenderer = self._create_renderer()
        self._render_token: Optional[jax.Array] = None
        self._render_fn = lambda token, data: self.renderer.render(token, data, self._mj_model)

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._sample_orientation = False

    def _create_renderer(self) -> BatchRenderer:
        vc = self._config.vision_config

        enabled_geom_groups = np.asarray(vc.enabled_geom_groups, dtype=np.int32, order="C")

        # Match micro-repro: explicit enabled_cameras int32 C-contiguous.
        cams = getattr(vc, "enabled_cameras", None)
        if cams is None:
            # Fallback to [0,1] if user forgot to set it
            cams = [0, 1]
        enabled_cameras = np.asarray(cams, dtype=np.int32, order="C")

        # Sanity: the MuJoCo model must actually have these camera indices.
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
    # Batched mjx.Data creation that matches micro-repro exactly.
    # -------------------------------------------------------------------------
    def _make_batched_data_safe(self, B: int) -> mjx.Data:
        """Create B independent Data objects (no broadcast views), like the micro-repro."""
        m = self._mjx_model
        data = jax.vmap(lambda _: mjx.make_data(m))(jp.arange(B))
        return data

    def _apply_per_world_initial_state(
        self,
        data: mjx.Data,
        box_pos: jax.Array,        # (B, 3)
        target_pos: jax.Array,     # (B, 3)
        rng_robot: jax.Array,      # (B, 2) keys
    ) -> mjx.Data:
        """Write qpos/qvel/ctrl + box + mocap into an existing batched data."""
        m = self._mjx_model
        B = int(box_pos.shape[0])

        nq = int(m.nq)
        nv = int(m.nv)

        # Base init state (float32)
        init_q0 = jp.asarray(self._init_q, dtype=jp.float32)[:nq]  # (nq,)
        qpos = jp.broadcast_to(init_q0[None, :], (B, nq)).astype(jp.float32)

        qvel = jp.zeros((B, nv), dtype=jp.float32)

        ctrl0 = jp.asarray(self._init_ctrl, dtype=jp.float32)
        ctrl = jp.broadcast_to(ctrl0[None, ...], (B,) + ctrl0.shape).astype(jp.float32)

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        # Per-world robot joint noise (only affects robot portion)
        eps = jp.float32(0.01)
        qpos_noise = jax.vmap(
            lambda k: jax.random.uniform(k, (nq,), minval=-eps, maxval=eps)
        )(rng_robot).astype(jp.float32)

        data = data.replace(
            qpos=data.qpos.at[:, : self._obj_qposadr].add(qpos_noise[:, : self._obj_qposadr])
        )

        # Set object position into qpos (batched)
        data = data.replace(
            qpos=data.qpos.at[:, self._obj_qposadr : self._obj_qposadr + 3].set(
                box_pos.astype(jp.float32)
            )
        )

        # Set target mocap (batched)
        target_quat0 = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
        target_quat = jp.broadcast_to(target_quat0[None, :], (B, 4))

        mpos = data.mocap_pos
        mquat = data.mocap_quat

        # Ensure batched mocap arrays
        if mpos.ndim == 2:
            mpos = jp.broadcast_to(mpos[None, ...], (B,) + mpos.shape)
        if mquat.ndim == 2:
            mquat = jp.broadcast_to(mquat[None, ...], (B,) + mquat.shape)

        mpos = mpos.at[:, 0, :].set(target_pos.astype(jp.float32))
        mquat = mquat.at[:, 0, :].set(target_quat)

        data = data.replace(mocap_pos=mpos, mocap_quat=mquat)
        return data

    # -------------------------------------------------------------------------
    # Renderer init: mirror micro-repro (forward before init; explicit cameras)
    # -------------------------------------------------------------------------
    def _maybe_init_renderer(self, data_for_init: mjx.Data, debug: bool):
        if self._render_token is not None:
            return

        # Hard asserts BEFORE calling the extension.
        # These should fail loudly in Python rather than corrupting backend_config.
        if data_for_init.qpos.dtype != jp.float32:
            data_for_init = data_for_init.replace(qpos=jp.asarray(data_for_init.qpos, jp.float32))
        if data_for_init.qvel.dtype != jp.float32:
            data_for_init = data_for_init.replace(qvel=jp.asarray(data_for_init.qvel, jp.float32))
        if data_for_init.ctrl.dtype != jp.float32:
            data_for_init = data_for_init.replace(ctrl=jp.asarray(data_for_init.ctrl, jp.float32))

        B = int(data_for_init.qpos.shape[0]) if data_for_init.qpos.ndim >= 2 else 1
        if B != self.render_batch_size:
            # Renderer is built for num_worlds=self.render_batch_size.
            # If you run B != render_batch_size, you must rebuild renderer or always reset at that size.
            raise ValueError(
                f"Renderer num_worlds={self.render_batch_size} but init data batch B={B}. "
                "Keep them equal."
            )

        if debug:
            backend = jax.lib.xla_bridge.get_backend()
            print("[diag] jax/jaxlib:", jax.__version__, jaxlib.__version__)
            print("[diag] backend:", backend.platform, "|", getattr(backend, "platform_version", None))
            print("[diag] JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "<unset>"))

            for name in [
                "madrona_mjx._madrona_mjx_batch_renderer",
                "madrona_mjx._madrona_mjx_visualizer",
                "_madrona_mjx_batch_renderer",
                "_madrona_mjx_visualizer",
            ]:
                spec = importlib.util.find_spec(name)
                print("[diag]", name, "->", getattr(spec, "origin", None))

            eg = np.asarray(self._config.vision_config.enabled_geom_groups, dtype=np.int32, order="C")
            cams = np.asarray(self._config.vision_config.enabled_cameras, dtype=np.int32, order="C")
            print("[diag] enabled_geom_groups:", eg, "dtype:", eg.dtype, "C_CONTIG:", eg.flags["C_CONTIGUOUS"])
            print("[diag] enabled_cameras:", cams, "dtype:", cams.dtype, "C_CONTIG:", cams.flags["C_CONTIGUOUS"])
            print("[diag] mj_model.ncam:", self._mj_model.ncam)
            print("[diag] init data qpos:", tuple(data_for_init.qpos.shape), data_for_init.qpos.dtype)

        # Micro-repro does forward() before init; do the same.
        m = self._mjx_model
        data_for_init = jax.vmap(lambda d: mjx.forward(m, d))(data_for_init)

        # Init renderer token
        if debug:
            print("[diag] calling renderer.init ...")
        tok, _, _ = self.renderer.init(data_for_init, self._mj_model)
        jax.block_until_ready(tok)
        self._render_token = tok

        if debug:
            print("[diag] renderer.init OK; token:", getattr(tok, "shape", None), getattr(tok, "dtype", None))
            # Smoke render once
            _, rgb, _ = self.renderer.render(tok, data_for_init, self._mj_model)
            jax.block_until_ready(rgb)
            print("[diag] smoke render OK:", tuple(rgb.shape), rgb.dtype)

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

            # Safe reference-like batched data (micro-repro style)
            data = self._make_batched_data_safe(B)

            # IMPORTANT: init renderer token using safe data + forward (micro style)
            self._maybe_init_renderer(data, debug=debug)

            # Now apply per-world state and forward for the actual start state
            data = self._apply_per_world_initial_state(
                data=data,
                box_pos=box_pos,
                target_pos=target_pos,
                rng_robot=rng_robot,
            )
            data = jax.vmap(lambda d: mjx.forward(m, d))(data)

            # Brightness noise
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

            info, obs = self._get_obs(data, info)

            reward = jp.zeros((B,), dtype=jp.float32)
            done = jp.zeros((B,), dtype=jp.float32)
            return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action_scale = self._config.action_scale
        delta = action * action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = self._physics_step(state.data, ctrl, self.n_substeps)

        info, raw_rewards = self._get_reward(data, state.info)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        box_pos = data.xpos[:, self._obj_body, :]  # (B, 3)

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
        
    def _physics_step(self, data: mjx.Data, ctrl: jax.Array, n_substeps: int) -> mjx.Data:
        """Steps physics for either batched [B,...] or unbatched [...] mjx.Data."""
        m = self._mjx_model

        def step_one(d_single: mjx.Data, c_single: jax.Array) -> mjx.Data:
            # scan expects unbatched mjx.Data
            def one_substep(d, _):
                d = d.replace(ctrl=c_single)
                d = mjx.step(m, d)
                return d, ()
            return jax.lax.scan(one_substep, d_single, (), length=n_substeps)[0]

        # If batched, vmap over worlds
        if data.qpos.ndim == 2:
            return jax.vmap(step_one)(data, ctrl)
        else:
            # tolerate ctrl coming in as [1,nu]
            if ctrl.ndim == 2:
                ctrl = ctrl[0]
            return step_one(data, ctrl)

    def _get_reward(self, data: mjx.Data, info: dict[str, Any]):
        # NOTE: with batched MJX, these are (B, 3) when indexed by body id
        box_pos = data.xpos[:, self._obj_body, :]  # (B, 3)
        
        assert box_pos.ndim == 2 and box_pos.shape[1] == 3, box_pos.shape
        target_pos = info["target_pos"]              # (B, 3)

        # Use the hand geom centers as the gripper position (no PandaBase _hand_body needed)
        pad_ids = jp.array(self._floor_hand_geom_ids[:2], dtype=jp.int32)  # left/right pads only
        pad_pos = data.geom_xpos[:, pad_ids, :]                            # (B, 2, 3)
        gripper_pos = jp.mean(pad_pos, axis=1)    
        d_gripper_box = jp.linalg.norm(gripper_pos - box_pos, axis=-1)   # (B,)
        d_box_target = jp.linalg.norm(box_pos - target_pos, axis=-1)     # (B,)

        reached_box = jp.minimum(info["reached_box"] + (d_gripper_box < 0.05), 1.0)
        info = {**info, "reached_box": reached_box}

        gripper_box = 1.0 - jp.tanh(5.0 * d_gripper_box)
        box_target = reached_box * (1.0 - jp.tanh(5.0 * d_box_target))

        # ------------------------------------------------------------
        # Distance-based floor + box collision (NO sensors)
        # ------------------------------------------------------------
        # data.geom_xpos is (B, ngeom, 3) for batched data
        geom_xpos = data.geom_xpos

        # Floor reference z
        floor_z = geom_xpos[:, self._floor_geom_id, 2]  # (B,)

        # Hand geoms z positions
        hand_geom_ids = jp.array(self._floor_hand_geom_ids, dtype=jp.int32)  # (Nh,)
        hand_z = geom_xpos[:, hand_geom_ids, 2]  # (B, Nh)
        min_hand_z = jp.min(hand_z, axis=1)      # (B,)

        # Consider a "floor collision" if the lowest hand geom gets too close to the floor plane
        floor_coll = (min_hand_z - floor_z) < self._floor_clearance_margin
        no_floor_collision = jp.where(floor_coll, 0.0, 1.0).astype(jp.float32)

        # Hand-box collision: any hand geom center within threshold of box center
        hand_pos = geom_xpos[:, hand_geom_ids, :]                 # (B, Nh, 3)
        dists = jp.linalg.norm(hand_pos - box_pos[:, None, :], axis=-1)  # (B, Nh)
        hand_box = jp.any(dists < self._hand_box_margin, axis=1)          # (B,)
        no_box_collision = jp.where(hand_box, 0.0, 1.0).astype(jp.float32)

        raw_rewards = dict(
            gripper_box=gripper_box,
            box_target=box_target,
            no_floor_collision=no_floor_collision,
            no_box_collision=no_box_collision,
            robot_target_qpos=jp.zeros_like(gripper_box),
        )
        return info, raw_rewards

    def render_pixels(self, render_token: jax.Array, data_batched: mjx.Data) -> jax.Array:
        """
        Robustly handle renderer RGB output shapes.

        Expected (most common for stereo):
          - (B, 2, H, W, 4)  -> multiworld, two cameras
          - (2, H, W, 4)     -> single world, two cameras

        We explicitly avoid `if rgb.shape[0] == 2` because it breaks when B==2.
        """
        _, rgb, _ = self._render_fn(render_token, data_batched)

        try:
            B = int(data_batched.qpos.shape[0]) if data_batched.qpos.ndim >= 2 else 1
        except Exception:
            B = 1

        if rgb.ndim == 5:
            # (B, C, H, W, 4)
            if rgb.shape[1] < 2:
                raise ValueError(
                    f"Renderer returned rgb with {rgb.shape[1]} cameras, expected >=2 for stereo."
                )
            left = rgb[:, 0]
            right = rgb[:, 1]
        elif rgb.ndim == 4:
            # Either (2, H, W, 4) for single-world stereo OR (B, H, W, 4) for mono.
            if rgb.shape[0] == 2 and B == 1:
                left = rgb[0:1]
                right = rgb[1:2]
            elif rgb.shape[0] == B:
                raise ValueError(
                    f"Renderer returned rgb shape {tuple(rgb.shape)} which looks mono (B,H,W,C). "
                    "Expected stereo (B,2,H,W,C). Check camera config."
                )
            else:
                raise ValueError(
                    f"Unrecognized rgb shape {tuple(rgb.shape)} with inferred B={B}. "
                    "Expected (B,2,H,W,C) or (2,H,W,C)."
                )
        else:
            raise ValueError(f"Unrecognized rgb ndim={rgb.ndim}, shape={tuple(rgb.shape)}")

        left = left[..., :3].astype(jp.float32) / 255.0
        right = right[..., :3].astype(jp.float32) / 255.0
        pixels = jp.concatenate([left, right], axis=2)  # concat along width
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
