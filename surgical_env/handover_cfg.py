"""Isaac Lab configuration for the two-arm needle handover with the stereo camera rig.

Scene layout from Isaac for Healthcare v0.4.0's dual-PSM needle handover
(robotic.surgery.tasks/.../surgical/handover/config/needle/ik_rel_env_cfg.py: two PSMs, bases
7 cm either side of the centre, same orientation, relative-pose IK). That task ships without
rewards; the stages, rewards and success test here are new. Everything else (tissue pad,
needle, lights, stereo rig, solver settings, speed caps, glitch cut-offs, 25 Hz control) is
the same as the single-arm lift task in env_cfg.py.

Stages (see HandoverRLEnv.handover_state):
  1. arm 1 reaches the needle on the pad and lifts it          -> "picked" (latched)
  2. arm 1 brings it to the handover point, arm 2 grasps it, arm 1 lets go -> "handed_over" (latched)
  3. arm 2 carries it to the goal                               -> success
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import (
    PSM_ARM_JOINTS,
    NeedleTissueSceneCfg,
    gripper_action_cfg,
    ik_arm_action_cfg,
    needle_cfg,
    setup_lights_and_cameras,
    setup_timing,
    tissue_cfg,
    tuned_robot_cfg,
)
from .handover_geom import NEEDLE_SCALE
from .psm import PSM_BASE_POS, PSM_HIGH_PD_CFG
from .stereo import StereoRig

ROBOTS = ("robot_1", "robot_2")


@configclass
class HandoverSceneCfg(NeedleTissueSceneCfg):
    # The single-arm entries are switched off (Isaac Lab skips None entries).
    robot: ArticulationCfg | None = None
    ee_frame: FrameTransformerCfg | None = None
    robot_1: ArticulationCfg = MISSING  # passes the needle (at -x)
    robot_2: ArticulationCfg = MISSING  # receives it (at +x)


@configclass
class HandoverActionsCfg:
    """14 numbers in [-1, 1]: arm 1 pose change (6), arm 1 gripper, arm 2 pose change (6), arm 2 gripper."""

    arm_1: DifferentialInverseKinematicsActionCfg = MISSING
    gripper_1: mdp.BinaryJointPositionActionCfg = MISSING
    arm_2: DifferentialInverseKinematicsActionCfg = MISSING
    gripper_2: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class HandoverObservationsCfg:
    """Not used by the agent (it sees the stereo images and, with agent.proprio, the robot state)."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HandoverEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": MISSING, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object", body_names="Object")},
    )
    pin_tissue = EventTerm(func=mdp.pin_tissue_bottom, mode="reset")


@configclass
class HandoverRewardsCfg:
    # Stage 1: arm 1 reaches and lifts (reach only counts until the needle has been picked up).
    reaching_needle_1 = RewTerm(func=mdp.handover_reach_1, params={"std": 0.1}, weight=1.0)
    lifting_needle = RewTerm(func=mdp.handover_lifted, weight=5.0)
    # Stage 2: arm 1 brings it to the handover point while arm 2 comes to it.
    to_handover_point = RewTerm(func=mdp.handover_to_point, params={"std": 0.05}, weight=8.0)
    reaching_needle_2 = RewTerm(func=mdp.handover_reach_2, params={"std": 0.05}, weight=4.0)
    # Stage 3: held by arm 2 alone (after a real handover), carried to the goal.
    held_by_arm_2 = RewTerm(func=mdp.handover_held_by_2, weight=10.0)
    goal_tracking = RewTerm(func=mdp.handover_goal, params={"std": 0.1}, weight=16.0)
    goal_tracking_fine_grained = RewTerm(func=mdp.handover_goal, params={"std": 0.02}, weight=5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    # Arm joints only (jaw contact chatter is not motion worth penalising), one term per arm.
    joint_vel_1 = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=PSM_ARM_JOINTS)}
    )
    joint_vel_2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=PSM_ARM_JOINTS)}
    )


@configclass
class HandoverTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    # Simulation glitches end the episode as a time-out (never learned as a failure), as in the lift task.
    blowup_nonfinite_1 = DoneTerm(func=mdp.blowup_nonfinite, params={"robot_cfg": SceneEntityCfg("robot_1")}, time_out=True)
    blowup_nonfinite_2 = DoneTerm(func=mdp.blowup_nonfinite, params={"robot_cfg": SceneEntityCfg("robot_2")}, time_out=True)
    blowup_joint_speed_1 = DoneTerm(func=mdp.blowup_joint_speed, params=MISSING, time_out=True)
    blowup_joint_speed_2 = DoneTerm(func=mdp.blowup_joint_speed, params=MISSING, time_out=True)
    blowup_needle_speed = DoneTerm(func=mdp.blowup_needle_speed, params=MISSING, time_out=True)


@configclass
class HandoverCurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel_1 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel_1", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel_2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel_2", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class NeedleHandoverEnvCfg(ManagerBasedRLEnvCfg):
    scene: HandoverSceneCfg = HandoverSceneCfg(num_envs=32, env_spacing=2.5, replicate_physics=False)
    observations: HandoverObservationsCfg = HandoverObservationsCfg()
    actions: HandoverActionsCfg = HandoverActionsCfg()
    # no commands: the goal is a fixed point (goal_point below), not a command
    rewards: HandoverRewardsCfg = HandoverRewardsCfg()
    terminations: HandoverTerminationsCfg = HandoverTerminationsCfg()
    events: HandoverEventCfg = HandoverEventCfg()
    curriculum: HandoverCurriculumCfg = HandoverCurriculumCfg()
    # Task settings read by HandoverRLEnv (positions relative to the environment origin).
    lift_height_w: float = MISSING
    success_threshold: float = MISSING
    tissue_top_w: float = MISSING
    tissue_deformable: bool = MISSING
    handover_point: tuple[float, float, float] = MISSING
    goal_point: tuple[float, float, float] = MISSING
    hold_distance: float = MISSING
    release_distance: float = MISSING
    needle_scale: float = MISSING


def robot_base_positions(env) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Base positions of arm 1 (-x) and arm 2 (+x), relative to the environment origin."""
    z = PSM_BASE_POS[2]
    return (-env.handover_base_x, 0.0, z), (env.handover_base_x, 0.0, z)


def make_handover_env_cfg(env, cam, with_depth: bool = False, track_camera_pose: bool = False) -> tuple[NeedleHandoverEnvCfg, StereoRig]:
    """Build the Isaac Lab config of the handover task from our EnvConfig / CameraConfig (see config.py)."""
    cfg = NeedleHandoverEnvCfg()
    tissue_top = env.tissue_size[2]

    # ---- scene: same pad, needle, lights and stereo rig as the lift task, two arms
    cfg.scene.num_envs = env.num_envs
    cfg.scene.env_spacing = env.env_spacing
    cfg.scene.replicate_physics = False
    cfg.scene.tissue = tissue_cfg(env)
    cfg.tissue_deformable = env.tissue_deformable
    cfg.scene.object = needle_cfg(env, spawn_xy=(env.handover_needle_x, 0.0))
    # The rig looks at the middle of the workspace: just right of the handover point above the pad centre.
    rig = setup_lights_and_cameras(cfg.scene, env, cam, (env.handover_look_at_x, 0.0), with_depth, track_camera_pose)
    for name, pos in zip(ROBOTS, robot_base_positions(env)):
        robot = tuned_robot_cfg(PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/" + name.capitalize()), env)
        robot = robot.replace(init_state=robot.init_state.replace(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)))
        setattr(cfg.scene, name, robot)

    # ---- actions
    cfg.actions.arm_1 = ik_arm_action_cfg("robot_1", env)
    cfg.actions.gripper_1 = gripper_action_cfg("robot_1")
    cfg.actions.arm_2 = ik_arm_action_cfg("robot_2", env)
    cfg.actions.gripper_2 = gripper_action_cfg("robot_2")

    # ---- glitch cut-offs
    for i, name in enumerate(ROBOTS, start=1):
        getattr(cfg.terminations, f"blowup_joint_speed_{i}").params = {
            "max_joint_vel": env.blowup_joint_vel,
            "robot_cfg": SceneEntityCfg(name, joint_names=PSM_ARM_JOINTS),
        }
    cfg.terminations.blowup_needle_speed.params = {"max_object_speed": env.blowup_needle_speed}

    # ---- needle placement, task points and thresholds
    xy = env.needle_xy_range
    cfg.events.reset_object_position.params["pose_range"] = {"x": (-xy, xy), "y": (-xy, xy), "z": (0.0, 0.0)}
    if not env.penalty_curriculum:
        cfg.curriculum = None
    if not (env.tissue_deformable and env.pin_tissue_bottom):
        cfg.events.pin_tissue = None
    cfg.lift_height_w = tissue_top + env.lift_height
    cfg.success_threshold = env.success_threshold
    cfg.tissue_top_w = tissue_top
    cfg.handover_point = (0.0, 0.0, tissue_top + env.goal_height)
    cfg.goal_point = (env.handover_goal_x, 0.0, tissue_top + env.goal_height)
    cfg.hold_distance = env.hold_distance
    cfg.release_distance = env.release_distance
    cfg.needle_scale = NEEDLE_SCALE

    setup_timing(cfg, env, cam)
    cfg.viewer.eye = (0.0, -0.3, 0.15)
    cfg.viewer.lookat = (0.0, 0.0, 0.04)
    return cfg, rig
