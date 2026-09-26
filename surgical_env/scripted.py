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

from .rotations import axis_angle_from_quat, quat_conjugate, quat_mul

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
