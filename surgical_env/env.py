"""Vectorised stereo-image environment around the Isaac Lab needle-on-tissue task.

Import this module only after the simulator has been launched (see sim_app.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import mdp
from .env_cfg import EE_BODY, NeedleTissueEnvCfg, make_env_cfg
from .handover_cfg import ROBOTS as HANDOVER_ROBOTS
from .handover_cfg import NeedleHandoverEnvCfg, make_handover_env_cfg
from .handover_geom import arc_points_local, update_handover_stages
from .rotations import quat_apply, subtract_frame_transforms
from .stereo import StereoRig

JAW_JOINTS = ["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"]


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

    def needle_goal_distance(self) -> torch.Tensor:
        return mdp.needle_goal_distance(self)


class HandoverRLEnv(ManagerBasedRLEnv):
    """Two-arm needle handover (see handover_cfg.py). Like NeedleTissueRLEnv, it records each
    episode's outcome in ``_reset_idx`` before the reset, and it latches the handover stages:

      picked       arm 1 has lifted the needle on its own (arm 2 not near it)
      handed_over  after that, arm 2 held the lifted needle while arm 1 was clear of it
    """

    cfg: NeedleHandoverEnvCfg
    ROBOTS = HANDOVER_ROBOTS

    def _ensure_handover(self) -> None:
        if hasattr(self, "picked"):
            return
        n, dev = self.num_envs, self.device
        self.picked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.handed_over = torch.zeros(n, dtype=torch.bool, device=dev)
        self.final_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.final_goal_distance = torch.zeros(n, device=dev)
        self.final_handed_over = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arc = arc_points_local(self.cfg.needle_scale).to(dev)  # (K, 3) along the needle, its root frame
        self._ee_ids = [self.scene[r].find_bodies(EE_BODY)[0][0] for r in self.ROBOTS]
        self._handover_point = torch.tensor(self.cfg.handover_point, device=dev)
        self._goal_point = torch.tensor(self.cfg.goal_point, device=dev)
        self._state, self._state_step = None, -1

    def tip_needle_distance(self, tip_w: torch.Tensor) -> torch.Tensor:
        """(N,) distance from a tool tip to the closest point of the needle."""
        needle = self.scene["object"]
        k = self._arc.shape[0]
        q = needle.data.root_quat_w[:, None, :].expand(-1, k, -1)
        arc_w = needle.data.root_pos_w[:, None, :] + quat_apply(q, self._arc[None].expand(self.num_envs, -1, -1))
        return torch.linalg.vector_norm(arc_w - tip_w[:, None, :], dim=-1).amin(dim=1)

    def handover_state(self) -> dict[str, torch.Tensor]:
        """Distances, lifted flag and stages for this control step (computed once per step, cached)."""
        self._ensure_handover()
        if self._state is not None and self._state_step == self.common_step_counter:
            return self._state
        cfg, needle, origins = self.cfg, self.scene["object"], self.scene.env_origins
        tips = [self.scene[r].data.body_pos_w[:, i] for r, i in zip(self.ROBOTS, self._ee_ids)]
        d1, d2 = self.tip_needle_distance(tips[0]), self.tip_needle_distance(tips[1])
        pos = needle.data.root_pos_w
        lifted = pos[:, 2] > cfg.lift_height_w
        self.picked, self.handed_over, held_by_2 = update_handover_stages(
            self.picked, self.handed_over, lifted, d1, d2, cfg.hold_distance, cfg.release_distance)
        goal_distance = torch.linalg.vector_norm(pos - (origins + self._goal_point), dim=1)
        self._state = {
            "d1": d1,
            "d2": d2,
            "lifted": lifted,
            "picked": self.picked.clone(),
            "handed_over": self.handed_over.clone(),
            "held_by_2": held_by_2,
            "handover_point_distance": torch.linalg.vector_norm(pos - (origins + self._handover_point), dim=1),
            "goal_distance": goal_distance,
            "success": held_by_2 & (goal_distance < cfg.success_threshold),
        }
        self._state_step = self.common_step_counter
        return self._state

    def _reset_idx(self, env_ids: Sequence[int]):
        self._ensure_handover()
        if hasattr(self, "reward_manager"):
            s = self.handover_state()
            self.final_success[env_ids] = s["success"][env_ids]
            self.final_goal_distance[env_ids] = s["goal_distance"][env_ids]
            self.final_handed_over[env_ids] = self.handed_over[env_ids]
        self.picked[env_ids] = False
        self.handed_over[env_ids] = False
        self._state = None  # the latches changed
        super()._reset_idx(env_ids)

    def success(self) -> torch.Tensor:
        return self.handover_state()["success"]

    def needle_goal_distance(self) -> torch.Tensor:
        return self.handover_state()["goal_distance"]


class StereoNeedleEnv:
    """Returns stereo images as uint8 tensors on the simulation device.

    obs: (num_envs, 2, 3, H, W) -- view 0 = left eye, view 1 = right eye.
    Actions: (num_envs, 7 per arm) in [-1, 1]: tool-tip translation (3) and rotation (3),
    relative to its current pose, then the gripper (< 0 closes, >= 0 opens); for the
    handover task arm 1's 7 numbers come first, then arm 2's.
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
        self.robot_names = tuple(getattr(env, "ROBOTS", ("robot",)))
        self.num_arms = len(self.robot_names)
        self.robots = [env.scene[r] for r in self.robot_names]
        self._ee_ids = [r.find_bodies(EE_BODY)[0][0] for r in self.robots]
        self._jaw_id_lists = [r.find_joints(JAW_JOINTS)[0] for r in self.robots]
        self._ee_idx, self._jaw_ids = self._ee_ids[0], self._jaw_id_lists[0]
        self.proprio_dim = 9 * self.num_arms

    def proprio(self) -> torch.Tensor:
        """(N, 9 per arm) robot state from its own kinematics: tool-tip position in the robot base frame (x10,
        so decimetres), tool-tip orientation quaternion (w >= 0) and the two jaw angles (x2). No needle state."""
        parts = []
        for i, robot in enumerate(self.robots):
            ee_pos, ee_quat = self.tool_pose_in_base(i)
            quat = torch.where(ee_quat[:, :1] < 0, -ee_quat, ee_quat)
            jaws = robot.data.joint_pos[:, self._jaw_id_lists[i]]
            parts.append(torch.cat([ee_pos * 10.0, quat, jaws * 2.0], dim=-1))
        return torch.cat(parts, dim=-1).float()

    def tool_pose_in_base(self, arm: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos, quat) of an arm's tool tip in that arm's base frame."""
        robot, i = self.robots[arm], self._ee_ids[arm]
        return subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                         robot.data.body_pos_w[:, i], robot.data.body_quat_w[:, i])

    def tool_tip_w(self, arm: int = 0) -> torch.Tensor:
        return self.robots[arm].data.body_pos_w[:, self._ee_ids[arm]]

    def base_pos_local(self, arm: int = 0) -> torch.Tensor:
        """(N, 3) an arm's base position relative to the environment origin."""
        return self.robots[arm].data.root_pos_w - self.env.scene.env_origins

    def single_arm_action(self, arm_action: torch.Tensor, arm: int = 0) -> torch.Tensor:
        """Full action that moves one arm by `arm_action` (N, 7) and keeps the others still with open grippers."""
        full = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        full[:, 6::7] = 1.0
        full[:, 7 * arm:7 * arm + 7] = arm_action
        return full

    def handover_poses(self) -> dict:
        """What the scripted handover controller needs, positions relative to the environment origin (world axes)."""
        origins = self.env.scene.env_origins
        needle = self.env.scene["object"]
        return {
            "ee_pos": [self.tool_tip_w(i) - origins for i in range(self.num_arms)],
            "ee_quat": [r.data.body_quat_w[:, i] for r, i in zip(self.robots, self._ee_ids)],
            "base_pos": [r.data.root_pos_w - origins for r in self.robots],
            "base_quat": [r.data.root_quat_w for r in self.robots],
            "needle_pos": needle.data.root_pos_w - origins,
            "needle_quat": needle.data.root_quat_w,
            "handover_point": self.env._handover_point.expand(self.num_envs, -1),
            "goal": self.env._goal_point.expand(self.num_envs, -1),
        }

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
        if hasattr(self.env, "handed_over"):  # handover task: did arm 2 take the needle over (in this episode)
            info["handed_over"] = torch.where(done, self.env.final_handed_over, self.env.handed_over)
        return self.images(), reward.clone(), terminated.clone(), truncated.clone(), info

    # -------------------------------------------------------------- state access
    def poses_in_base(self):
        """(ee_pos, ee_quat, needle_pos, needle_quat, goal_pos, goal_quat), robot base frame (lift task)."""
        robot = self.env.scene["robot"]
        needle = self.env.scene["object"]
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        ee_pos, ee_quat = subtract_frame_transforms(
            root_pos, root_quat, robot.data.body_pos_w[:, self._ee_idx], robot.data.body_quat_w[:, self._ee_idx]
        )
        obj_pos, obj_quat = subtract_frame_transforms(root_pos, root_quat, needle.data.root_pos_w, needle.data.root_quat_w)
        goal = self.env.command_manager.get_command("object_pose")
        return ee_pos, ee_quat, obj_pos, obj_quat, goal[:, :3], goal[:, 3:7]

    def needle_xy_local(self) -> torch.Tensor:
        """(N, 2) needle position on the pad, metres from the centre of where it is dropped
        (the pad centre for the lift task, arm 1's half of the pad for the handover)."""
        pos = self.env.scene["object"].data.root_pos_w - self.env.scene.env_origins
        spawn = torch.tensor(self.env.cfg.scene.object.init_state.pos[:2], device=pos.device)
        return pos[:, :2] - spawn

    def close(self) -> None:
        self.env.close()


def make_env(cfg, device: str = "cuda:0", with_depth: bool = False, track_camera_pose: bool = False) -> StereoNeedleEnv:
    make, env_cls = (make_handover_env_cfg, HandoverRLEnv) if cfg.env.task == "handover" else (make_env_cfg, NeedleTissueRLEnv)
    env_cfg, rig = make(cfg.env, cfg.camera, with_depth=with_depth, track_camera_pose=track_camera_pose)
    env_cfg.seed = cfg.train.seed
    env_cfg.sim.device = device
    env = env_cls(cfg=env_cfg)
    return StereoNeedleEnv(env, rig)
