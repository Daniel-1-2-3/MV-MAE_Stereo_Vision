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

This version keeps the *same task/reward structure* as the prior StereoPickCube,
but uses a single-world physics core. Any batching is done via `jax.vmap` in
`reset` / `step` (or by external wrappers), while rendering is *always* done by
calling the renderer on a single world at a time (no batched renderer outputs).

Key points:
- `_reset_single` / `_step_single` are single-world and JIT-friendly.
- `reset` / `step` accept either single or batched inputs; batched uses `jax.vmap`.
- `init_render` / `render_pixels` are Python-side helpers (do NOT JIT these).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math

from Custom_Mujoco_Playground._src import mjx_env
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda import panda_kinematics

# Renderer is optional (vision-only)
try:
    from madrona_mjx.renderer import BatchRenderer # type: ignore
except Exception:  # pragma: no cover
    BatchRenderer = None


def _add_assets(dst: Dict[str, Any], src: Dict[str, Any], base_path) -> None:
    """Recursively merge `src` into `dst` while prefixing filesystem paths."""
    for k, v in src.items():
        if isinstance(v, dict):
            dst.setdefault(k, {})
            _add_assets(dst[k], v, base_path)
        else:
            if isinstance(v, str) and (v.endswith(".png") or v.endswith(".jpg") or v.endswith(".xml") or v.endswith(".stl") or v.endswith(".obj")):
                dst[k] = str(base_path / v)
            else:
                dst[k] = v


# ---------------------------
# Default configs (unchanged)
# ---------------------------

def default_vision_config() -> mjx_env.VisionConfig:
    return mjx_env.VisionConfig(
        width=64,
        height=64,
        fov=60.0,
        camera_names=("view_0", "view_1"),
        render_batch_size=1,  # single-world
        enabled_geom_groups=(0, 1, 2),
        use_raytracer=True,
        brightness=(1.0, 1.0),
    )


def default_config() -> mjx_env.Config:
    return mjx_env.Config(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=500,
        action_repeat=1,
        vision=True,
        vision_config=default_vision_config(),
        # physics buffer sizes
        nconmax=512,
        njmax=2048,
        # task-specific knobs (kept)
        action_scale=0.05,
    )


class StereoPickCube(panda.PandaBase):
    """Single-world core env; supports batched reset/step via vmap."""

    def __init__(
        self,
        config: Optional[mjx_env.Config] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = default_config()

        # Force single-world renderer usage
        if config.vision and config.vision_config is not None:
            vc = config.vision_config
            if getattr(vc, "render_batch_size", 1) != 1:
                vc = vc.replace(render_batch_size=1)
                config = config.replace(vision_config=vc)

        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        # Renderer setup (single-world only)
        self.renderer = None
        if self._config.vision:
            if BatchRenderer is None:
                raise ImportError(
                    "madrona_mjx is required for vision rendering but could not be imported."
                )
            vc = self._config.vision_config
            self.renderer = BatchRenderer(
                mj_model=self._mj_model,
                gpu_id=0,
                num_worlds=1,
                batch_render_view_width=vc.width,
                batch_render_view_height=vc.height,
                enabled_geom_groups=jp.asarray(vc.enabled_geom_groups, dtype=jp.int32),
                use_rasterizer=not vc.use_raytracer,
            )

        self._post_init()
        
    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._guide_q = self._mj_model.keyframe('picked').qpos
        self._guide_ctrl = self._mj_model.keyframe('picked').ctrl
        # Use forward kinematics to init cartesian control
        self._start_tip_transform = panda_kinematics.compute_franka_fk(
            self._init_ctrl[:7]
        )
        self._sample_orientation = False
    # ---------------------------
    # Model / XML (unchanged)
    # ---------------------------

    @property
    def xml_path(self) -> str:
        return str(
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )

    def modify_model(self, mj_model):
        """Hook for future MJCF tweaks (kept for compatibility)."""
        return mj_model

    # ---------------------------
    # Single-world physics core
    # ---------------------------

    @property
    def observation_size(self) -> int:
        """Flattened observation size used by some wrappers (pixel obs)."""
        vc = self._config.vision_config
        if vc is None:
            return 0
        return int(vc.height) * int(vc.width) * 2 * 3

    def _reset_single(self, rng: jax.Array) -> mjx_env.State:
        """Reset one world (rng shape: (2,))."""
        rng_box, rng_target = jax.random.split(rng, 2)

        # Sample object + target positions (same ranges as before)
        box_pos = self._init_obj_pos + jax.random.uniform(
            rng_box, (3,), minval=-self._r_range, maxval=self._r_range
        )
        target_pos = self._init_target_pos + jax.random.uniform(
            rng_target, (3,), minval=-self._r_range, maxval=self._r_range
        )

        # Initial qpos / qvel (single world)
        init_q = jp.asarray(self._init_q)
        init_q = init_q.at[self._obj_qposadr : self._obj_qposadr + 3].set(box_pos)

        data = mjx_env.make_data(
            self._mjx_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=jp.float32),
            ctrl=jp.asarray(self._init_ctrl),
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # Set mocap target pose
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target].set(self._init_target_quat),
        )

        # Placeholder obs (wrapper will overwrite with rendered pixels)
        vc = self._config.vision_config
        obs = jp.zeros((vc.height, vc.width * 2, 3), dtype=jp.uint8) if (vc is not None) else jp.zeros((0,), dtype=jp.float32)

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=jp.float32),
            "gripper_box": jp.array(0.0, dtype=jp.float32),
            "box_target": jp.array(0.0, dtype=jp.float32),
            "no_floor_collision": jp.array(0.0, dtype=jp.float32),
            "robot_target_qpos": jp.array(0.0, dtype=jp.float32),
        }

        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=jp.float32),
        }

        return mjx_env.State(data, obs, jp.array(0.0, jp.float32), jp.array(0.0, jp.float32), metrics, info)

    def _step_single(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step one world (action shape: (action_size,))."""
        delta = action * self._action_scale
        ctrl = state.data.ctrl.at[:].add(delta)

        data = mjx_env.step(
            self._mjx_model, state.data, ctrl, self._n_substeps
        )

        # Reward + info updates
        reward, info, metrics = self._get_reward(data, state.info)

        # Termination conditions (same as before)
        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        nan_qpos = jp.any(jp.isnan(data.qpos))
        done = jp.logical_or(out_of_bounds, nan_qpos).astype(jp.float32)

        metrics = dict(metrics)
        metrics["out_of_bounds"] = out_of_bounds.astype(jp.float32)

        # Keep obs unchanged (wrapper overwrites with rendered pixels each step)
        return state.replace(data=data, reward=reward, done=done, metrics=metrics, info=info)

    # Public API: accept either single or batched inputs
    def reset(self, rng: jax.Array) -> mjx_env.State:
        if rng.ndim == 1:
            return self._reset_single(rng)
        return jax.vmap(self._reset_single)(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if action.ndim == 1:
            return self._step_single(state, action)
        return jax.vmap(self._step_single, in_axes=(0, 0))(state, action)

    # ---------------------------
    # Reward (single-world)
    # ---------------------------

    def _has_contact_with_floor(self, data: mjx.Data, geom: int) -> jax.Array:
        """Returns bool scalar: whether `geom` is in contact with the floor geom."""
        g1 = data.contact.geom1
        g2 = data.contact.geom2
        # valid contacts are [0, ncon)
        idx = jp.arange(g1.shape[0])
        valid = idx < data.ncon
        pair = (jp.minimum(g1, g2) == self._floor_geom) & (jp.maximum(g1, g2) == geom)
        return jp.any(pair & valid)

    def _get_reward(
        self, data: mjx.Data, info: Dict[str, Any]
    ) -> Tuple[jax.Array, Dict[str, Any], Dict[str, Any]]:
        # positions
        box_pos = data.xpos[self._obj_body]
        gripper_pos = data.site_xpos[self._gripper_site]
        target_pos = info["target_pos"]

        box_gripper_dist = jp.linalg.norm(box_pos - gripper_pos)
        box_target_dist = jp.linalg.norm(box_pos - target_pos)

        # orientation error
        box_mat = data.xmat[self._obj_body]
        box_mat = box_mat.reshape(3, 3)

        target_quat = data.mocap_quat[self._mocap_target]
        target_mat = math.quat_to_mat(target_quat)

        orient_err = jp.linalg.norm(box_mat - target_mat)

        # Reach box once
        reached_box_prev = info.get("reached_box", jp.array(0.0, jp.float32))
        reached_box = jp.maximum(reached_box_prev, (box_gripper_dist < self._threshold).astype(jp.float32))
        info = dict(info)
        info["reached_box"] = reached_box

        # reward terms (same structure)
        r_gripper_box = reached_box
        r_box_target = reached_box * jp.exp(-self._box_target_scale * (box_target_dist + orient_err))

        # floor collision penalty (same structure)
        hand_floor = jp.stack([
            self._has_contact_with_floor(data, g) for g in self._hand_geom
        ])
        floor_collision = jp.any(hand_floor).astype(jp.float32)
        r_no_floor_collision = 1.0 - floor_collision

        # guide / target qpos term (same structure)
        qpos = data.qpos
        qpos_arm = qpos[self._robot_arm_qposadr]
        qpos_guide = jp.asarray(self._guide_qpos)
        robot_target_qpos = jp.exp(-jp.linalg.norm(qpos_arm - qpos_guide))
        r_robot_target_qpos = 0.1 * robot_target_qpos

        reward = (
            self._r_gripper_box_scale * r_gripper_box
            + self._r_box_target_scale * r_box_target
            + self._r_no_floor_collision_scale * r_no_floor_collision
            + self._r_robot_target_qpos_scale * r_robot_target_qpos
        )

        metrics = {
            "gripper_box": r_gripper_box,
            "box_target": r_box_target,
            "no_floor_collision": r_no_floor_collision,
            "robot_target_qpos": r_robot_target_qpos,
        }

        return reward, info, metrics

    # ---------------------------
    # Rendering helpers (single-world renderer calls ONLY)
    # ---------------------------

    def init_render(self, data: mjx.Data):
        """Initialize renderer token using the first world in `data` if batched."""
        if self.renderer is None:
            raise RuntimeError("Vision is disabled; renderer is not initialized.")
        data0 = self._slice_first_world(data)
        try:
            render_token, _rgb, _depth = self.renderer.init(data0, self._mjx_model)
        except TypeError:
            render_token, _rgb, _depth = self.renderer.init(data0)
        return render_token

    def render_pixels(self, render_token, data: mjx.Data) -> jax.Array:
        """Render fused stereo pixels.

        Accepts batched or single `data`. Always renders *one world at a time*
        and stacks outputs into shape (B, H, 2W, 3) uint8.
        """
        if self.renderer is None:
            raise RuntimeError("Vision is disabled; renderer is not initialized.")

        vc = self._config.vision_config
        H, W = int(vc.height), int(vc.width)

        # Determine batch size from qpos
        if data.qpos.ndim == 1:
            # single world
            fused = self._render_one(render_token, data)
            return fused[None, ...]
        else:
            B = data.qpos.shape[0]
            imgs = []
            for i in range(int(B)):
                di = self._index_world(data, i, int(B))
                imgs.append(self._render_one(render_token, di))
            return jp.stack(imgs, axis=0)

    def _render_one(self, render_token, data1: mjx.Data) -> jax.Array:
        """Render exactly one world -> fused (H, 2W, 3) uint8."""
        try:
            _tok2, rgb, _depth = self.renderer.render(render_token, data1, self._mjx_model)
        except TypeError:
            _tok2, rgb, _depth = self.renderer.render(render_token, data1)

        # Common rgb layouts seen in madrona_mjx:
        #   (B, C, H, W, 4)   or   (C, B, H, W, 4)   or   (C, H, W, 4)
        if rgb.ndim == 5:
            # Prefer squeezing the dimension that equals 1 (single-world)
            if rgb.shape[0] == 1:
                rgb = rgb[0]              # (C, H, W, 4)
            elif rgb.shape[1] == 1:
                rgb = rgb[:, 0]           # (C, H, W, 4)
            else:
                raise ValueError(f"Unexpected 5D rgb (no singleton world dim): {rgb.shape}")
        if rgb.ndim != 4:
            raise ValueError(f"Unexpected rgb shape from renderer: {rgb.shape}")

        # Take first two cams (view_0, view_1); fall back if only one
        left = rgb[0, :, :, :3]
        right = rgb[1, :, :, :3] if rgb.shape[0] > 1 else left

        # Fuse side-by-side along width
        fused = jp.concatenate([left, right], axis=1)
        return fused



    @staticmethod
    def _slice_first_world(data: mjx.Data) -> mjx.Data:
        """If `data` is batched, slice world 0; else return unchanged."""
        if hasattr(data, "qpos") and getattr(data.qpos, "ndim", 0) >= 2:
            B = data.qpos.shape[0]
            return StereoPickCube._index_world(data, 0, int(B))
        return data

    @staticmethod
    def _index_world(tree: Any, i: int, B: int) -> Any:
        """Index the leading batch dimension for all leaves that have it."""
        def _idx(x):
            if hasattr(x, "shape") and len(x.shape) >= 1 and x.shape[0] == B:
                return x[i]
            return x
        return jax.tree_util.tree_map(_idx, tree)
