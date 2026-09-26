# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Reward / observation functions from Isaac for Healthcare v0.4.0
# (robotic.surgery.tasks/.../surgical/lift/mdp), unchanged. `pin_tissue_bottom`,
# `needle_success` and the `blowup_*` terms are additions for the soft-tissue version of the task.

"""MDP terms for needle lifting on soft tissue."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ----------------------------------------------------------------------------- observations (original)
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


# ----------------------------------------------------------------------------- rewards (original)
def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


# ----------------------------------------------------------------------------- additions
def needle_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """World-space distance between the needle and the goal point, (num_envs,)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3])
    return torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)


def needle_success(env: ManagerBasedRLEnv, minimal_height: float, threshold: float) -> torch.Tensor:
    """Needle lifted above `minimal_height` (world z) and within `threshold` metres of the goal, (num_envs,) bool."""
    object: RigidObject = env.scene["object"]
    lifted = object.data.root_pos_w[:, 2] > minimal_height
    return lifted & (needle_goal_distance(env) < threshold)


def blowup_nonfinite(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination: NaN/inf in the robot or needle state, (num_envs,) bool."""
    robot = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ok = torch.isfinite(robot.data.joint_pos).all(dim=1) & torch.isfinite(robot.data.joint_vel).all(dim=1)
    ok &= torch.isfinite(object.data.root_state_w).all(dim=1)
    return ~ok


def blowup_joint_speed(env: ManagerBasedRLEnv, max_joint_vel: float, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Termination: a robot joint moves faster than any real motion could (rad/s or m/s), (num_envs,) bool."""
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_vel.abs().amax(dim=1) > max_joint_vel


def blowup_needle_speed(
    env: ManagerBasedRLEnv, max_object_speed: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Termination: the needle moves faster than `max_object_speed` m/s (e.g. squeezed out of a contact), (num_envs,) bool."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.linalg.vector_norm(object.data.root_lin_vel_w, dim=1) > max_object_speed


def pin_tissue_bottom(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("tissue"),
    height_tolerance: float = 1.0e-4,
) -> None:
    """Reset event: hold the bottom layer of the tissue's simulation mesh at its rest position.

    The pad then behaves like tissue attached to the structure underneath: it can
    be pressed and deformed but not slid around or lifted off. Kinematic-target
    flag convention (Isaac Lab): 0 = driven to the target, 1 = free.
    """
    tissue: DeformableObject = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    rest = tissue.data.default_nodal_state_w[env_ids, :, :3]
    targets = tissue.data.nodal_kinematic_target[env_ids].clone()
    targets[..., :3] = rest
    z = rest[..., 2]
    bottom = z <= z.min(dim=1, keepdim=True).values + height_tolerance
    targets[..., 3] = torch.where(bottom, 0.0, 1.0)
    tissue.write_nodal_kinematic_target_to_sim(targets, env_ids=env_ids)
