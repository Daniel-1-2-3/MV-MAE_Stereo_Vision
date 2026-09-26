# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Based on the needle-lift task of Isaac for Healthcare v0.4.0
# (robotic.surgery.tasks/.../surgical/lift/lift_env_cfg.py and
# lift/config/needle/ik_rel_env_cfg.py). Changes, all marked "CHANGED":
#   * the needle lies on a tissue pad (rigid by default, soft FEM optional) on a platform instead of a table;
#   * two tiled cameras form a stereo endoscope looking at the workspace;
#   * the goal is a fixed point above the tissue (images cannot show a moving goal);
#   * heights of the reward/success thresholds are measured from the tissue top;
#   * 25 Hz control, 5 s episodes, IK action scale sized for millimetre-level motion;
#   * joint speed caps that are actually applied (see EnvConfig.arm_joint_vel_limit) and
#     `blowup_*` time-outs that end an episode if the simulation glitches.
# Reward terms, weights, the other terminations and the curriculum are the original ones.

"""Isaac Lab configuration for needle lifting from a tissue pad with a stereo camera rig."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import robotic_surgery_assets
from .psm import PSM_BASE_POS, PSM_HIGH_PD_CFG
from .stereo import StereoRig, make_stereo_rig

PSM_ARM_JOINTS = [
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
]
EE_BODY = "psm_tool_tip_link"


@configclass
class NeedleTissueSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
        debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/" + EE_BODY, name="end_effector")],
    )
    object: RigidObjectCfg = MISSING  # the suture needle
    tissue: DeformableObjectCfg | AssetBaseCfg = MISSING  # soft pad, or a static rigid one
    # CHANGED: rigid platform (top at z = 0) under the tissue, replaces the table.
    platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(0.40, 0.40, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.35), roughness=0.8),
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.95)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    stereo_left: TiledCameraCfg = MISSING
    stereo_right: TiledCameraCfg = MISSING


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY,
        resampling_time_range=(1.0e6, 1.0e6),  # CHANGED: never resampled within an episode
        debug_vis=False,
        ranges=MISSING,
    )


@configclass
class ActionsCfg:
    body_joint_pos: DifferentialInverseKinematicsActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["psm_tool_gripper.*_joint"],
        open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
        close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
    )


@configclass
class ObservationsCfg:
    """Proprioceptive/privileged state. Not used by the agent (it only sees the stereo images)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # Order matters: reset everything, place the needle, then (re)pin the tissue.
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": MISSING,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
    pin_tissue = EventTerm(func=mdp.pin_tissue_bottom, mode="reset")


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": MISSING}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": MISSING, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": MISSING, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    # CHANGED: added. Simulation glitches, split by what went wrong. Time-outs (not failures):
    # the episode is cut and never bootstrapped through. Isaac Lab logs each one as
    # Episode_Termination/<name>.
    blowup_nonfinite = DoneTerm(func=mdp.blowup_nonfinite, time_out=True)
    blowup_joint_speed = DoneTerm(func=mdp.blowup_joint_speed, params=MISSING, time_out=True)
    blowup_needle_speed = DoneTerm(func=mdp.blowup_needle_speed, params=MISSING, time_out=True)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class NeedleTissueEnvCfg(ManagerBasedRLEnvCfg):
    scene: NeedleTissueSceneCfg = NeedleTissueSceneCfg(num_envs=32, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Extra (task-specific) settings read by the environment wrapper.
    lift_height_w: float = MISSING
    success_threshold: float = MISSING
    tissue_top_w: float = MISSING
    tissue_deformable: bool = MISSING


def _soft_tissue_cfg(env, visual) -> DeformableObjectCfg:
    """FEM soft-tissue pad (env.tissue_deformable=True), lying on the platform."""
    return DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tissue",
        spawn=sim_utils.MeshCuboidCfg(
            size=tuple(env.tissue_size),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=env.tissue_hex_resolution,
            ),
            visual_material=visual,
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=env.tissue_youngs_modulus,
                poissons_ratio=env.tissue_poissons_ratio,
                dynamic_friction=env.tissue_friction,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * env.tissue_size[2])),
        debug_vis=False,
    )


def _camera_cfg(prim_name: str, pos, rig: StereoRig, cam, with_depth: bool, track_pose: bool) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rig.quat_ros, convention="ros"),
        data_types=["rgb", "distance_to_image_plane"] if with_depth else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam.focal_length,
            focus_distance=400.0,
            horizontal_aperture=cam.horizontal_aperture,
            clipping_range=(cam.near_clip, cam.far_clip),
        ),
        width=cam.width,
        height=cam.height,
        update_latest_camera_pose=track_pose,
    )


def make_env_cfg(env, cam, with_depth: bool = False, track_camera_pose: bool = False) -> tuple[NeedleTissueEnvCfg, StereoRig]:
    """Build the Isaac Lab config from our EnvConfig / CameraConfig (see config.py)."""
    cfg = NeedleTissueEnvCfg()
    sx, sy, sz = env.tissue_size
    tissue_top = sz  # the pad's bottom rests on the platform top (z = 0)

    # ---- scene
    cfg.scene.num_envs = env.num_envs
    cfg.scene.env_spacing = env.env_spacing
    # Deformable bodies do not work with replicated physics (same as Isaac Lab's own deformable lift task);
    # kept off for the rigid pad too, so both variants build the same scene.
    cfg.scene.replicate_physics = False
    tissue_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=env.tissue_color, roughness=0.6)
    if env.tissue_deformable:
        cfg.scene.tissue = _soft_tissue_cfg(env, tissue_visual)
    else:
        # Static collider (no rigid-body properties): same size, colour and place as the soft pad.
        cfg.scene.tissue = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Tissue",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * sz)),
            spawn=sim_utils.CuboidCfg(
                size=(sx, sy, sz),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=tissue_visual,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=env.tissue_friction, dynamic_friction=env.tissue_friction
                ),
            ),
        )
    cfg.tissue_deformable = env.tissue_deformable
    needle_usd = robotic_surgery_assets.Needle_SDF if env.needle_asset == "sdf" else robotic_surgery_assets.Needle
    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, tissue_top + env.needle_spawn_height), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=needle_usd,
            scale=(0.4, 0.4, 0.4),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200,
                max_linear_velocity=200,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        ),
    )
    look_at = (env.goal_xy[0], env.goal_xy[1], tissue_top + cam.look_at_height)
    rig = make_stereo_rig(cam, look_at)
    cfg.scene.stereo_left = _camera_cfg("StereoLeft", rig.left_pos, rig, cam, with_depth, track_camera_pose)
    cfg.scene.stereo_right = _camera_cfg("StereoRight", rig.right_pos, rig, cam, with_depth, track_camera_pose)

    # ---- robot joint speed caps (CHANGED): `velocity_limit_sim` is what Isaac Lab >= 2.0 applies
    act = cfg.scene.robot.actuators
    cfg.scene.robot = cfg.scene.robot.replace(
        actuators={
            "psm": act["psm"].replace(velocity_limit=None, velocity_limit_sim=env.arm_joint_vel_limit),
            "psm_tool": act["psm_tool"].replace(velocity_limit=None, velocity_limit_sim=env.gripper_joint_vel_limit),
        }
    )
    cfg.terminations.blowup_joint_speed.params = {"max_joint_vel": env.blowup_joint_vel}
    cfg.terminations.blowup_needle_speed.params = {"max_object_speed": env.blowup_needle_speed}

    # ---- actions: relative end-effector pose (IK) + binary gripper, 7 numbers in [-1, 1]
    p, r = env.ik_pos_scale, env.ik_rot_scale
    cfg.actions.body_joint_pos = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=PSM_ARM_JOINTS,
        body_name=EE_BODY,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=(p, p, p, r, r, r),
    )

    # ---- fixed goal, expressed in the robot base frame (base has no rotation)
    gx, gy = env.goal_xy
    gz = tissue_top + env.goal_height
    bx, by, bz = PSM_BASE_POS
    cfg.commands.object_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(gx - bx, gx - bx),
        pos_y=(gy - by, gy - by),
        pos_z=(gz - bz, gz - bz),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )

    # ---- needle placement and height thresholds relative to the tissue top
    xy = env.needle_xy_range
    cfg.events.reset_object_position.params["pose_range"] = {"x": (-xy, xy), "y": (-xy, xy), "z": (0.0, 0.0)}
    if not (env.tissue_deformable and env.pin_tissue_bottom):
        cfg.events.pin_tissue = None
    lift_h = tissue_top + env.lift_height
    cfg.rewards.lifting_object.params["minimal_height"] = lift_h
    cfg.rewards.object_goal_tracking.params["minimal_height"] = lift_h
    cfg.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = lift_h
    cfg.lift_height_w = lift_h
    cfg.success_threshold = env.success_threshold
    cfg.tissue_top_w = tissue_top

    # ---- timing
    cfg.decimation = env.decimation
    cfg.sim.dt = env.sim_dt
    cfg.sim.render_interval = env.decimation
    cfg.sim.render.antialiasing_mode = cam.antialiasing
    cfg.episode_length_s = env.episode_length_s
    # Re-render after resets so the first image of a new episode shows the reset scene.
    cfg.rerender_on_reset = True
    cfg.viewer.eye = (0.2, 0.2, 0.1)
    cfg.viewer.lookat = (0.0, 0.0, 0.04)
    return cfg, rig
