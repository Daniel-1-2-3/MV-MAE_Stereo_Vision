"""Vectorised stereo-image environment around the Isaac Lab needle-on-tissue task.

Import this module only after the simulator has been launched (see sim_app.py).

Isaac Lab 3.0: asset / sensor ``.data.*`` fields and camera outputs are ProxyArrays, read as
tensors with ``.torch``. Isaac Lab quaternions are (x, y, z, w); they are converted to this
project's (w, x, y, z) with ``wxyz_from_xyzw`` as soon as they are read, so every value this
wrapper returns (robot state, poses for the scripted controller) is the same as on Isaac Lab 2.3.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import mdp
from .env_cfg import EE_BODY, NeedleTissueEnvCfg, make_env_cfg
from .rotations import subtract_frame_transforms, wxyz_from_xyzw
from .stereo import StereoRig


class NeedleTissueRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that remembers whether each episode ended in success.

    Isaac Lab resets finished environments inside ``step()``, before returning,
    so the final state of an episode is otherwise lost. ``_reset_idx`` runs before
    any state is reset, so success is measured there on the true final state.
    """

    cfg: NeedleTissueEnvCfg
    BLOWUP_TERMS = ("blowup_nonfinite", "blowup_joint_speed", "blowup_needle_speed")

    # Set to True (e.g. by diagnose_blowups.py) to keep a description of the state at every glitch.
    record_blowups: bool = False

    def _reset_idx(self, env_ids: Sequence[int]):
        if not hasattr(self, "final_success"):
            self.final_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.final_goal_distance = torch.zeros(self.num_envs, device=self.device)
            self.blowup_records: list[dict] = []
        if hasattr(self, "command_manager"):
            self.final_success[env_ids] = self.success()[env_ids]
            self.final_goal_distance[env_ids] = mdp.needle_goal_distance(self)[env_ids]
            if self.record_blowups:
                self._record_blowups(env_ids)
        super()._reset_idx(env_ids)

    @torch.no_grad()
    def _record_blowups(self, env_ids) -> None:
        """Describe the state of every env that is being reset because of a glitch (before it is reset)."""
        ids = torch.as_tensor(env_ids, device=self.device).long().reshape(-1)
        causes = {name: self.termination_manager.get_term(name)[ids] for name in self.BLOWUP_TERMS}
        hit = torch.zeros_like(ids, dtype=torch.bool)
        for c in causes.values():
            hit |= c
        if not bool(hit.any()):
            return
        robot, needle = self.scene["robot"], self.scene["object"]
        top = self.cfg.tissue_top_w
        jv = robot.data.joint_vel.torch[ids].abs()
        tip = self.scene["ee_frame"].data.target_pos_w.torch[ids, 0, :]
        npos = needle.data.root_pos_w.torch[ids]
        if self.cfg.tissue_deformable:
            tissue = self.scene["tissue"]
            dent = (tissue.data.default_nodal_state_w.torch[ids, :, 2] - tissue.data.nodal_pos_w.torch[ids, :, 2]).amax(dim=1)
        else:
            dent = torch.zeros(ids.numel(), device=self.device)  # rigid pad cannot dent
        g = robot.find_joints(["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"])[0]
        opening = robot.data.joint_pos.torch[ids][:, g[1]] - robot.data.joint_pos.torch[ids][:, g[0]]
        action = self.action_manager.action[ids]
        rows = {
            "episode_step": self.episode_length_buf[ids].float(),
            "max_joint_speed": jv.amax(dim=1),
            "fastest_joint": jv.argmax(dim=1).float(),
            "needle_speed": torch.linalg.vector_norm(needle.data.root_lin_vel_w.torch[ids], dim=1),
            "needle_ang_speed": torch.linalg.vector_norm(needle.data.root_ang_vel_w.torch[ids], dim=1),
            "needle_height_mm": (npos[:, 2] - top) * 1000.0,
            "tip_height_mm": (tip[:, 2] - top) * 1000.0,
            "tip_to_needle_mm": torch.linalg.vector_norm(tip - npos, dim=1) * 1000.0,
            "gripper_opening_rad": opening,
            "gripper_command": action[:, -1],
            "arm_action_norm": torch.linalg.vector_norm(action[:, :-1], dim=1),
            "tissue_dent_mm": dent * 1000.0,
        }
        rows = {k: v[hit].cpu().tolist() for k, v in rows.items()}
        names = [next(n for n, c in causes.items() if bool(c[i])) for i in hit.nonzero().flatten().tolist()]
        for i, cause in enumerate(names):
            rec = {k: v[i] for k, v in rows.items()}
            rec["cause"] = cause
            rec["env_id"] = int(ids[hit][i])
            rec["fastest_joint"] = robot.joint_names[int(rec["fastest_joint"])]
            self.blowup_records.append(rec)

    def success(self) -> torch.Tensor:
        return mdp.needle_success(self, self.cfg.lift_height_w, self.cfg.success_threshold)


class StereoNeedleEnv:
    """Returns stereo images as uint8 tensors on the simulation device.

    obs: (num_envs, 2, 3, H, W) -- view 0 = left eye, view 1 = right eye.
    Actions: (num_envs, 7) in [-1, 1]: tool-tip translation (3) and rotation (3),
    relative to its current pose, then the gripper (< 0 closes, >= 0 opens).
    """

    def __init__(self, env: NeedleTissueRLEnv, rig: StereoRig, cfg=None):
        self.env = env
        self.rig = rig
        self.cfg = cfg  # the experiment Config (needed for the robustness suite's scene changes)
        self.hires_prim: str | None = None  # env_0's render-only paper camera (add_hires_camera)
        self._hires_checked = False
        self.num_envs = env.num_envs
        self.device = env.device
        self.action_dim = env.action_manager.total_action_dim
        self.max_episode_steps = int(env.max_episode_length)
        self.step_dt = env.step_dt
        self.left = env.scene["stereo_left"]
        self.right = env.scene["stereo_right"]
        self._ee_idx = env.scene["robot"].find_bodies(EE_BODY)[0][0]
        self._jaw_ids = env.scene["robot"].find_joints(["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"])[0]

    proprio_dim = 9

    def proprio(self) -> torch.Tensor:
        """(N, 9) robot state from its own kinematics: tool-tip position in the robot base frame (x10, so
        decimetres), tool-tip orientation quaternion (w >= 0) and the two jaw angles (x2). No needle state."""
        ee_pos, ee_quat, *_ = self.poses_in_base()
        quat = torch.where(ee_quat[:, :1] < 0, -ee_quat, ee_quat)
        jaws = self.env.scene["robot"].data.joint_pos.torch[:, self._jaw_ids]
        return torch.cat([ee_pos * 10.0, quat, jaws * 2.0], dim=-1).float()

    @property
    def obs_shape(self) -> tuple[int, ...]:
        h, w = self.left.cfg.height, self.left.cfg.width
        return (2, 3, h, w)

    def images(self) -> torch.Tensor:
        left = self.left.data.output["rgb"].torch  # (N, H, W, 3) uint8, a view into the RGBA buffer
        right = self.right.data.output["rgb"].torch
        return torch.stack([left, right], dim=1).permute(0, 1, 4, 2, 3).contiguous()

    def depths(self) -> torch.Tensor:
        """(N, 2, H, W) float metres; only available when the env was built with depth."""
        left = self.left.data.output["distance_to_image_plane"].torch[..., 0]
        right = self.right.data.output["distance_to_image_plane"].torch[..., 0]
        return torch.stack([left, right], dim=1)

    def reset(self) -> torch.Tensor:
        self.env.reset()
        return self.images()

    def step(self, action: torch.Tensor):
        _, reward, terminated, truncated, extras = self.env.step(action.to(self.device).clamp(-1.0, 1.0))
        done = terminated | truncated
        info = {
            # valid only where done: outcome of the episode that just ended
            "final_success": self.env.final_success.clone(),
            "final_goal_distance": self.env.final_goal_distance.clone(),
            # valid only where not done: state of the running episode
            "success_now": self.env.success(),
            "needle_lifted": self.env.scene["object"].data.root_pos_w.torch[:, 2] > self.env.cfg.lift_height_w,
            # Isaac Lab episode statistics (per reward term etc.), present when something reset
            "log": dict(extras.get("log", {})) if bool(done.any()) else {},
        }
        return self.images(), reward.clone(), terminated.clone(), truncated.clone(), info

    # -------------------------------------------------------------- state access
    def poses_in_base(self):
        """(ee_pos, ee_quat, needle_pos, needle_quat, goal_pos, goal_quat), robot base frame, quaternions (w, x, y, z)."""
        robot = self.env.scene["robot"]
        needle = self.env.scene["object"]
        root_pos, root_quat = robot.data.root_pos_w.torch, wxyz_from_xyzw(robot.data.root_quat_w.torch)
        ee_pos, ee_quat = subtract_frame_transforms(
            root_pos, root_quat, robot.data.body_pos_w.torch[:, self._ee_idx],
            wxyz_from_xyzw(robot.data.body_quat_w.torch[:, self._ee_idx]),
        )
        obj_pos, obj_quat = subtract_frame_transforms(root_pos, root_quat, needle.data.root_pos_w.torch,
                                                      wxyz_from_xyzw(needle.data.root_quat_w.torch))
        goal = self.env.command_manager.get_command("object_pose")  # (x, y, z, qx, qy, qz, qw) in Isaac Lab 3.0
        return ee_pos, ee_quat, obj_pos, obj_quat, goal[:, :3], wxyz_from_xyzw(goal[:, 3:7])

    def needle_xy_local(self) -> torch.Tensor:
        """(N, 2) needle position on the pad, metres from the pad centre."""
        pos = self.env.scene["object"].data.root_pos_w.torch - self.env.scene.env_origins
        return pos[:, :2]

    def close(self) -> None:
        self.env.close()

    # -------------------------------------------------------------- evaluation-time tools
    def training_state(self) -> dict:
        """Counters and reward weights that extra evaluation episodes would advance (Isaac Lab's reward curriculum
        reads common_step_counter and changes reward weights); the robustness suite puts them back afterwards."""
        rm = self.env.reward_manager
        return {"common_step_counter": self.env.common_step_counter, "sim_step_counter": self.env._sim_step_counter,
                "reward_weights": {name: rm.get_term_cfg(name).weight for name in rm.active_terms}}

    def restore_training_state(self, state: dict) -> None:
        self.env.common_step_counter = state["common_step_counter"]
        self.env._sim_step_counter = state["sim_step_counter"]
        rm = self.env.reward_manager
        for name, weight in state["reward_weights"].items():
            term = rm.get_term_cfg(name)
            term.weight = weight
            rm.set_term_cfg(name, term)

    def sim_perturbation(self, family: str, severity: float):
        """Apply one scene change of the robustness suite; returns the function that restores the exact previous
        state. Raises (with the reason) if it cannot be applied and restored exactly."""
        from . import eval_scene

        if self.cfg is None:
            raise RuntimeError("the env was built without its config")
        if family == "camera_azimuth":
            return eval_scene.orbit_stereo_rig(self.env.scene.env_prim_paths, self.rig, self.cfg.camera, severity, 0.0)
        if family == "camera_elevation":
            return eval_scene.orbit_stereo_rig(self.env.scene.env_prim_paths, self.rig, self.cfg.camera, 0.0, severity)
        if family == "light_intensity":
            return eval_scene.scale_dome_light(1.0 + severity)
        if family == "shadow_light":
            return eval_scene.add_shadow_light(self.cfg.env)
        raise NotImplementedError(f"no scene perturbation '{family}'")

    def add_hires_camera(self, hcfg, env_cfg) -> None:
        """Create env_0's render-only paper camera (hires.* settings); no render product until an evaluation."""
        from . import eval_scene
        from .stereo import view_camera_pose

        top = env_cfg.tissue_size[2]
        look_at = (env_cfg.goal_xy[0], env_cfg.goal_xy[1], top + hcfg.look_at_height)
        pos, quat = view_camera_pose(look_at, hcfg.distance, hcfg.azimuth_deg, hcfg.elevation_deg)
        prim = f"{self.env.scene.env_prim_paths[0]}/HiResCam"
        eval_scene.spawn_view_camera(prim, pos, quat, hcfg.width, hcfg.height, hcfg.focal_length,
                                     hcfg.horizontal_aperture)
        self.hires_prim = prim

    def _rendered_stereo(self) -> torch.Tensor:
        """Render (no physics step) and read the stereo cameras again."""
        self.env.sim.render()
        self.left.update(0.0, force_recompute=True)
        self.right.update(0.0, force_recompute=True)
        return self.images()

    def hires_recorder(self, hcfg, video_dir, env_step: int, fps: float):
        """Recorder for env_0's paper camera (and the left stereo camera at hires.left_eye_size) for one evaluation:
        videos/hires_eval_<step>.mp4 and hires_eval_left_<step>.mp4."""
        from . import eval_scene

        if self.hires_prim is None:
            raise RuntimeError("no hi-res camera: build the env with make_env(..., hires_camera=True)")
        video_dir = Path(video_dir)
        cams = [("main", self.hires_prim, (hcfg.width, hcfg.height), video_dir / f"hires_eval_{env_step}.mp4")]
        if hcfg.left_eye_size > 0:
            s = hcfg.left_eye_size
            cams.append(("left", f"{self.env.scene.env_prim_paths[0]}/StereoLeft", (s, s),
                         video_dir / f"hires_eval_left_{env_step}.mp4"))
        check = None if self._hires_checked else self._rendered_stereo  # verify once that the policy's images stay
        self._hires_checked = True
        return eval_scene.HiResRecorder(cams, fps, hcfg.crf, hcfg.preset, check=check)


def make_env(cfg, device: str = "cuda:0", with_depth: bool = False, track_camera_pose: bool = False,
             hires_camera: bool = False) -> StereoNeedleEnv:
    env_cfg, rig = make_env_cfg(cfg.env, cfg.camera, with_depth=with_depth, track_camera_pose=track_camera_pose)
    env_cfg.seed = cfg.train.seed
    env_cfg.sim.device = device
    env = NeedleTissueRLEnv(cfg=env_cfg)
    senv = StereoNeedleEnv(env, rig, cfg)
    if hires_camera:
        try:
            senv.add_hires_camera(cfg.hires, cfg.env)
        except Exception as e:  # the paper video is optional: never stop training for it
            print(f"WARNING: could not create the hi-res camera ({type(e).__name__}: {e}); no hi-res videos")
    return senv
