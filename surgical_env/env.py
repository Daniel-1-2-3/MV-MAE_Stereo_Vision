"""Vectorised stereo-image environment around the Isaac Lab needle-on-tissue task.

Import this module only after the simulator has been launched (see sim_app.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import mdp
from .env_cfg import EE_BODY, NeedleTissueEnvCfg, make_env_cfg
from .rotations import subtract_frame_transforms
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
        jv = robot.data.joint_vel[ids].abs()
        tip = self.scene["ee_frame"].data.target_pos_w[ids, 0, :]
        npos = needle.data.root_pos_w[ids]
        if self.cfg.tissue_deformable:
            tissue = self.scene["tissue"]
            dent = (tissue.data.default_nodal_state_w[ids, :, 2] - tissue.data.nodal_pos_w[ids, :, 2]).amax(dim=1)
        else:
            dent = torch.zeros(ids.numel(), device=self.device)  # rigid pad cannot dent
        g = robot.find_joints(["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"])[0]
        opening = robot.data.joint_pos[ids][:, g[1]] - robot.data.joint_pos[ids][:, g[0]]
        action = self.action_manager.action[ids]
        rows = {
            "episode_step": self.episode_length_buf[ids].float(),
            "max_joint_speed": jv.amax(dim=1),
            "fastest_joint": jv.argmax(dim=1).float(),
            "needle_speed": torch.linalg.vector_norm(needle.data.root_lin_vel_w[ids], dim=1),
            "needle_ang_speed": torch.linalg.vector_norm(needle.data.root_ang_vel_w[ids], dim=1),
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

    def __init__(self, env: NeedleTissueRLEnv, rig: StereoRig):
        self.env = env
        self.rig = rig
        self.num_envs = env.num_envs
        self.device = env.device
        self.action_dim = env.action_manager.total_action_dim
        self.max_episode_steps = int(env.max_episode_length)
        self.step_dt = env.step_dt
        self.left = env.scene["stereo_left"]
        self.right = env.scene["stereo_right"]
        self._ee_idx = env.scene["robot"].find_bodies(EE_BODY)[0][0]

    @property
    def obs_shape(self) -> tuple[int, ...]:
        h, w = self.left.cfg.height, self.left.cfg.width
        return (2, 3, h, w)

    def images(self) -> torch.Tensor:
        left = self.left.data.output["rgb"]  # (N, H, W, 3) uint8, a view into the RGBA buffer
        right = self.right.data.output["rgb"]
        return torch.stack([left, right], dim=1).permute(0, 1, 4, 2, 3).contiguous()

    def depths(self) -> torch.Tensor:
        """(N, 2, H, W) float metres; only available when the env was built with depth."""
        left = self.left.data.output["distance_to_image_plane"][..., 0]
        right = self.right.data.output["distance_to_image_plane"][..., 0]
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
            "needle_lifted": self.env.scene["object"].data.root_pos_w[:, 2] > self.env.cfg.lift_height_w,
            # Isaac Lab episode statistics (per reward term etc.), present when something reset
            "log": dict(extras.get("log", {})) if bool(done.any()) else {},
        }
        return self.images(), reward.clone(), terminated.clone(), truncated.clone(), info

    # -------------------------------------------------------------- state access
    def poses_in_base(self):
        """(ee_pos, ee_quat, needle_pos, needle_quat, goal_pos, goal_quat), robot base frame."""
        robot = self.env.scene["robot"]
        needle = self.env.scene["object"]
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        ee_pos, ee_quat = subtract_frame_transforms(
            root_pos, root_quat, robot.data.body_pos_w[:, self._ee_idx], robot.data.body_quat_w[:, self._ee_idx]
        )
        obj_pos, obj_quat = subtract_frame_transforms(root_pos, root_quat, needle.data.root_pos_w, needle.data.root_quat_w)
        goal = self.env.command_manager.get_command("object_pose")
        return ee_pos, ee_quat, obj_pos, obj_quat, goal[:, :3], goal[:, 3:7]

    def close(self) -> None:
        self.env.close()


def make_env(cfg, device: str = "cuda:0", with_depth: bool = False, track_camera_pose: bool = False) -> StereoNeedleEnv:
    env_cfg, rig = make_env_cfg(cfg.env, cfg.camera, with_depth=with_depth, track_camera_pose=track_camera_pose)
    env_cfg.seed = cfg.train.seed
    env_cfg.sim.device = device
    env = NeedleTissueRLEnv(cfg=env_cfg)
    return StereoNeedleEnv(env, rig)
