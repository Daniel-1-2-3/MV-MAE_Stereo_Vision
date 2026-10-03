# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Pick-and-lift state machine ported from Isaac for Healthcare v0.4.0
# (workflows/robotic_surgery/scripts/simulation/scripts/environments/state_machine/lift_needle_sm.py),
# rewritten in plain PyTorch and adapted to output *relative* IK actions (the
# action space the RL agent uses) instead of absolute end-effector poses.

"""Scripted needle-lift controller, used to record demonstrations and in test_stereo.py."""

from __future__ import annotations

import torch

from .rotations import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul, subtract_frame_transforms

REST, APPROACH_ABOVE, APPROACH, GRASP, LIFT = range(5)
GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0


def relative_action(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    des_pos: torch.Tensor,
    des_quat: torch.Tensor,
    gripper: torch.Tensor,
    pos_scale: float,
    rot_scale: float,
) -> torch.Tensor:
    """Raw action in [-1, 1] that moves the tool tip toward a desired pose (all in the robot base frame).

    Inverse of Isaac Lab's relative pose IK command: target = (ee_pos + d_pos,
    quat(d_rot) * ee_quat), with (d_pos, d_rot) = raw_action[:6] * scale.
    """
    d_pos = des_pos - ee_pos
    d_rot = axis_angle_from_quat(quat_mul(des_quat, quat_conjugate(ee_quat)))
    arm = torch.cat([d_pos / pos_scale, d_rot / rot_scale], dim=-1).clamp(-1.0, 1.0)
    return torch.cat([arm, gripper[:, None]], dim=-1)


class ScriptedNeedleLifter:
    """REST -> APPROACH_ABOVE -> APPROACH -> GRASP -> LIFT, advancing on timers like the original.

    Wait times (seconds) match the original: rest 0.5, both approaches 0.7,
    grasp 0.5, then lift until the episode ends.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device,
        pos_scale: float,
        rot_scale: float,
        above_offset: float = 0.05,
        wait_rest: float = 0.5,
        wait_approach: float = 0.7,
        wait_grasp: float = 0.5,
    ):
        self.dt = float(dt)
        self.pos_scale, self.rot_scale = pos_scale, rot_scale
        self.above_offset = above_offset
        self.waits = (wait_rest, wait_approach, wait_approach, wait_grasp)
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait_time = torch.zeros(num_envs, device=device)

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.state[:] = REST
            self.wait_time[:] = 0.0
        else:
            self.state[env_ids] = REST
            self.wait_time[env_ids] = 0.0

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        obj_pos: torch.Tensor,
        obj_quat: torch.Tensor,
        goal_pos: torch.Tensor,
        goal_quat: torch.Tensor,
    ) -> torch.Tensor:
        """All poses in the robot base frame, quaternions (w, x, y, z). Returns raw actions (N, 7)."""
        s = self.state
        above = obj_pos.clone()
        above[:, 2] += self.above_offset

        des_pos = ee_pos.clone()
        des_quat = ee_quat.clone()
        m = s == APPROACH_ABOVE
        des_pos[m], des_quat[m] = above[m], obj_quat[m]
        m = (s == APPROACH) | (s == GRASP)
        des_pos[m], des_quat[m] = obj_pos[m], obj_quat[m]
        m = s == LIFT
        des_pos[m], des_quat[m] = goal_pos[m], goal_quat[m]
        gripper = torch.where((s == GRASP) | (s == LIFT), GRIPPER_CLOSE, GRIPPER_OPEN).to(ee_pos.dtype)

        # Advance on timers (same order as the original kernel: decide, then add dt).
        wait = torch.tensor(self.waits + (float("inf"),), device=s.device)[s]
        advance = (self.wait_time >= wait) & (s < LIFT)
        self.state = torch.where(advance, s + 1, s)
        self.wait_time = torch.where(advance, torch.zeros_like(self.wait_time), self.wait_time) + self.dt

        return relative_action(ee_pos, ee_quat, des_pos, des_quat, gripper, self.pos_scale, self.rot_scale)

    def act(self, env) -> torch.Tensor:
        """Raw actions (N, 7) for a StereoNeedleEnv running the lift task."""
        return self.compute(*env.poses_in_base())


# ----------------------------------------------------------------------------- two-arm handover
# Arm 1 stages
A_REST, A_ABOVE, A_APPROACH, A_GRASP, A_LIFT, A_HOLD, A_RELEASE, A_RETREAT = range(8)
# Arm 2 stages
B_WAIT, B_ABOVE, B_APPROACH, B_GRASP, B_HOLD, B_CARRY = range(6)


class ScriptedNeedleHandover:
    """Two state machines in step, advancing on timers like ScriptedNeedleLifter.

    Arm 1: REST -> ABOVE (5 cm over the needle) -> APPROACH -> GRASP (close) -> LIFT (carry the needle
    to the handover point) -> HOLD (until arm 2 has closed on it) -> RELEASE (open) -> RETREAT.
    Arm 2: WAIT (until arm 1 holds the needle at the handover point) -> ABOVE (2 cm over its grasp
    point on the arc) -> APPROACH -> GRASP (close) -> HOLD (while arm 1 lets go) -> CARRY (to the goal).

    Carrying moves the tool by the needle's own error (target tool position = tool + (point - needle)),
    so it works wherever on the needle the tool holds it. Targets are computed in env-local world
    coordinates and turned into each arm's base frame for the relative IK action.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device,
        pos_scale: float,
        rot_scale: float,
        needle_scale: float,
        grasp_deg: float,
        above_offset: float = 0.05,
        above_offset_2: float = 0.02,
        retreat_offset: tuple[float, float, float] = (-0.03, 0.0, 0.03),
    ):
        from .handover_geom import arc_point_local, grasp_yaw_deg, quat_about_z

        self.dt = float(dt)
        self.pos_scale, self.rot_scale = pos_scale, rot_scale
        self.above_offset, self.above_offset_2 = above_offset, above_offset_2
        self.retreat_offset = torch.tensor(retreat_offset, device=device)
        self.grasp_local = torch.tensor(arc_point_local(grasp_deg, needle_scale), device=device)
        self.grasp_rot = torch.tensor(quat_about_z(grasp_yaw_deg(grasp_deg)), device=device)
        # seconds in each stage before moving on (inf = until the other arm says so / the episode ends)
        inf = float("inf")
        self.waits_1 = torch.tensor([0.3, 0.7, 0.7, 0.5, 1.2, inf, 0.5, inf], device=device)
        self.waits_2 = torch.tensor([inf, 0.8, 0.7, 0.5, inf, inf], device=device)
        self.state_1 = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state_2 = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_1 = torch.zeros(num_envs, device=device)
        self.time_2 = torch.zeros(num_envs, device=device)
        self.retreat_point = torch.zeros(num_envs, 3, device=device)

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.state_1[ids], self.state_2[ids] = A_REST, B_WAIT
        self.time_1[ids], self.time_2[ids] = 0.0, 0.0

    def act(self, env) -> torch.Tensor:
        """Raw actions (N, 14) for a StereoNeedleEnv running the handover task."""
        p = env.handover_poses()
        (ee1, ee2), (q1, q2) = p["ee_pos"], p["ee_quat"]
        n_pos, n_quat = p["needle_pos"], p["needle_quat"]
        s1, s2 = self.state_1, self.state_2

        # ---- arm 1 targets
        des1, dq1 = ee1.clone(), q1.clone()
        m = s1 == A_ABOVE
        des1[m] = n_pos[m] + torch.tensor([0.0, 0.0, self.above_offset], device=ee1.device)
        dq1[m] = n_quat[m]
        m = (s1 == A_APPROACH) | (s1 == A_GRASP)
        des1[m], dq1[m] = n_pos[m], n_quat[m]
        m = (s1 == A_LIFT) | (s1 == A_HOLD)
        des1[m] = ee1[m] + (p["handover_point"][m] - n_pos[m])
        m = s1 == A_RETREAT
        des1[m] = self.retreat_point[m]
        grip1 = torch.where((s1 >= A_GRASP) & (s1 <= A_HOLD), GRIPPER_CLOSE, GRIPPER_OPEN).to(ee1.dtype)

        # ---- arm 2 targets
        grasp_pos = n_pos + quat_apply(n_quat, self.grasp_local.expand_as(n_pos))
        grasp_quat = quat_mul(n_quat, self.grasp_rot.expand_as(n_quat))
        des2, dq2 = ee2.clone(), q2.clone()
        m = s2 == B_ABOVE
        des2[m] = grasp_pos[m] + torch.tensor([0.0, 0.0, self.above_offset_2], device=ee2.device)
        dq2[m] = grasp_quat[m]
        m = (s2 == B_APPROACH) | (s2 == B_GRASP)
        des2[m], dq2[m] = grasp_pos[m], grasp_quat[m]
        m = s2 == B_CARRY
        des2[m] = ee2[m] + (p["goal"][m] - n_pos[m])
        grip2 = torch.where(s2 >= B_GRASP, GRIPPER_CLOSE, GRIPPER_OPEN).to(ee2.dtype)

        # ---- advance: timers, plus the hand-shakes between the arms
        done_1 = (self.time_1 >= self.waits_1[s1]) & (s1 < A_RETREAT)
        done_1 |= (s1 == A_HOLD) & (s2 == B_HOLD)  # arm 2 has closed on the needle: let go
        done_2 = (self.time_2 >= self.waits_2[s2]) & (s2 < B_CARRY)
        done_2 |= (s2 == B_WAIT) & (s1 == A_HOLD)  # needle waiting at the handover point
        done_2 |= (s2 == B_HOLD) & (s1 == A_RETREAT)  # arm 1 has let go
        entering_retreat = done_1 & (s1 == A_RELEASE)
        self.retreat_point[entering_retreat] = ee1[entering_retreat] + self.retreat_offset
        self.state_1 = torch.where(done_1, s1 + 1, s1)
        self.state_2 = torch.where(done_2, s2 + 1, s2)
        self.time_1 = torch.where(done_1, torch.zeros_like(self.time_1), self.time_1) + self.dt
        self.time_2 = torch.where(done_2, torch.zeros_like(self.time_2), self.time_2) + self.dt

        actions = []
        for ee, q, des, dq, grip, base, bq in ((ee1, q1, des1, dq1, grip1, p["base_pos"][0], p["base_quat"][0]),
                                               (ee2, q2, des2, dq2, grip2, p["base_pos"][1], p["base_quat"][1])):
            ee_b, q_b = subtract_frame_transforms(base, bq, ee, q)
            des_b, dq_b = subtract_frame_transforms(base, bq, des, dq)
            actions.append(relative_action(ee_b, q_b, des_b, dq_b, grip, self.pos_scale, self.rot_scale))
        return torch.cat(actions, dim=-1)


def make_scripted_controller(env, cfg):
    """The scripted demonstrator for cfg.env.task, for a StereoNeedleEnv. Use .reset_idx(ids) and .act(env)."""
    e = cfg.env
    if e.task == "handover":
        from .handover_geom import NEEDLE_SCALE

        return ScriptedNeedleHandover(env.num_envs, env.step_dt, env.device, e.ik_pos_scale, e.ik_rot_scale,
                                      NEEDLE_SCALE, e.handover_grasp_deg)
    return ScriptedNeedleLifter(env.num_envs, env.step_dt, env.device, e.ik_pos_scale, e.ik_rot_scale)
