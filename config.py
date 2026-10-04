"""Experiment configuration.

Everything lives in one nested dataclass tree. Defaults come from here, a YAML
file (``configs/needle_tissue.yaml``) can override any of them, and so can
``section.key=value`` arguments on the command line, e.g.::

    python train.py env.num_envs=16 agent.batch_size=128 log.wandb_mode=offline

This module must not import Isaac Sim / Isaac Lab: it is loaded before the
simulator is launched.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    num_envs: int = 32
    env_spacing: float = 2.5
    # Physics runs at 1 / sim_dt Hz. The policy acts every `decimation` physics
    # steps: 0.005 s * 8 = 0.04 s -> 25 Hz control.
    sim_dt: float = 0.005
    decimation: int = 8
    episode_length_s: float = 5.0

    # Tissue pad on a rigid platform whose top is at z = 0. Rigid by default: with the
    # soft (FEM) pad, tool-needle-tissue contacts made the simulation glitch often
    # (needle shot out of the contact). tissue_deformable=True brings the soft pad back;
    # its bottom layer of nodes is then pinned to the platform. The youngs_modulus /
    # poissons_ratio / hex_resolution / pin settings only apply to the soft pad.
    tissue_deformable: bool = False
    tissue_size: tuple[float, float, float] = (0.10, 0.10, 0.01)
    tissue_youngs_modulus: float = 5.0e4  # Pa
    tissue_poissons_ratio: float = 0.45
    tissue_friction: float = 1.0
    tissue_hex_resolution: int = 12  # simulation voxels along the longest side
    tissue_color: tuple[float, float, float] = (0.78, 0.38, 0.38)
    pin_tissue_bottom: bool = True

    # Suture needle. "sdf" = the SDF-collision needle used by the original task,
    # "mesh" = the plain-mesh needle (use if SDF contact with the tissue misbehaves).
    needle_asset: str = "sdf"
    needle_spawn_height: float = 0.015  # above the tissue top, same drop as the original task
    needle_xy_range: float = 0.03  # uniform +- range of the needle position at reset

    # Task. The original task's goal moves every second, which images cannot show,
    # so the goal is a fixed point: centred over the tissue, `goal_height` above it.
    goal_xy: tuple[float, float] = (0.0, 0.0)
    goal_height: float = 0.04
    lift_height: float = 0.02  # "lifted" = needle this far above the tissue top (original: 2 cm above table)
    success_threshold: float = 0.02  # needle-to-goal distance for success

    # Lighting. The default dome light casts no visible shadows. shadow_light=True adds a distant light
    # tilted shadow_light_tilt_deg from overhead, coming from shadow_light_azimuth_deg (camera rig is
    # at 45 deg, so -45 deg lights from the side and the tool's shadow falls where the cameras see it).
    dome_light_intensity: float = 3000.0
    shadow_light: bool = False
    shadow_light_intensity: float = 2000.0
    shadow_light_tilt_deg: float = 35.0
    shadow_light_azimuth_deg: float = -45.0

    # Relative IK action scaling: an action of 1.0 moves the tool tip by
    # ik_pos_scale metres / rotates it by ik_rot_scale radians in one control step.
    ik_pos_scale: float = 0.005
    ik_rot_scale: float = 0.05
    # The original task's curriculum: after 10k simulation steps the action_rate and joint_vel
    # penalties grow to -0.1. False keeps them at their small starting weights.
    penalty_curriculum: bool = True

    # Joint speed caps for the robot, applied by PhysX. The vendored dVRK config sets
    # `velocity_limit`, which Isaac Lab >= 2.0 ignores for implicit actuators, so without
    # these the joints have no speed limit and contact glitches can spin them at 100s of rad/s.
    # Units: rad/s (m/s for the insertion joint). Normal motion stays well below them
    # (IK moves the tip <= 0.125 m/s and turns it <= 1.25 rad/s).
    arm_joint_vel_limit: float = 3.0
    # PhysX solver iterations for the robot (the original PSM config uses 4 / 0). With 0 velocity
    # iterations the jaws chatter at tens of rad/s while squeezing the needle.
    robot_solver_position_iterations: int = 16
    robot_solver_velocity_iterations: int = 4
    gripper_joint_vel_limit: float = 3.0
    # Safety net: end an episode (as a time-out, so it is not learned as a failure) if the
    # physics blows up: non-finite state, a robot joint faster than this (rad/s or m/s),
    # or the needle moving faster than `blowup_needle_speed` (m/s).
    blowup_joint_vel: float = 10.0  # arm joints only: the gripper jaws chatter in contact without anything blowing up
    blowup_needle_speed: float = 2.0


@dataclass
class CameraConfig:
    width: int = 96
    height: int = 96
    baseline: float = 0.005  # distance between the two eyes, metres (da Vinci endoscope ~ 5 mm)
    # Rig centre = look-at point + distance * (direction given by azimuth/elevation).
    # Framing (checked in tests/test_geometry.py): the tool tip's start position (~8 cm up),
    # the goal and the whole needle area all fall inside both images, and the rig stays
    # below the robot base (z = 0.15) on the side the original task's viewer looks from.
    look_at_height: float = 0.035  # look-at point is this far above the tissue top, at goal_xy
    distance: float = 0.17
    azimuth_deg: float = 45.0  # measured from +x towards +y
    elevation_deg: float = 35.0  # above the horizontal
    focal_length: float = 24.0  # mm (USD pinhole units)
    horizontal_aperture: float = 20.955  # mm -> ~47 deg horizontal field of view
    near_clip: float = 0.005
    far_clip: float = 5.0
    # Isaac Sim's default (DLSS) upscales from a lower internal resolution and needs >= 300 px;
    # at 96 px it blurs thin objects like the needle and mixes in previous frames. FXAA works at
    # the native resolution. Options: Off | FXAA | TAA | DLSS | DLAA.
    antialiasing: str = "FXAA"


@dataclass
class MVMAEConfig:
    frame_stack: int = 3
    patch_size: int = 16  # conv stem downsampling factor: 96 px -> 6 x 6 tokens per view
    embed_dim: int = 256
    encoder_depth: int = 8
    encoder_heads: int = 4
    decoder_dim: int = 256
    decoder_depth: int = 6
    decoder_heads: int = 4
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.95  # fraction of all tokens hidden (one whole view per frame + most of the rest)
    loss_on_masked_only: bool = True
    reward_prediction: bool = True


@dataclass
class AgentConfig:
    # Image input. "mvmae": the MV-MAE transformer (trained by its reconstruction / reward-prediction
    # losses plus the critic). "pixels": baseline without MV-MAE and without any learned image encoder:
    # the raw frames, shrunk by averaging pixel_downsample x pixel_downsample blocks, flattened and fed
    # straight to the actor's and critic's first layer. Everything else (actor, critic, augmentation,
    # demos, robot state) is identical; the mvmae.* settings other than frame_stack and the MV-MAE
    # losses / pre-training / encoder freezing do not apply to "pixels".
    encoder: str = "mvmae"
    pixel_downsample: int = 2  # "pixels": 96 px -> 48 px, 3 frames x 2 views x 3 colours x 48 x 48 = 41,472 numbers
    lr: float = 1e-4  # actor and critic
    encoder_lr: float = 1e-4  # MV-MAE encoder + decoder
    encoder_warmup_updates: int = 2500
    feature_dim: int = 50
    hidden_dim: int = 1024
    critic_tau: float = 0.01
    gamma: float = 0.99
    nstep: int = 3
    batch_size: int = 256
    # Exploration noise added to the actor's actions (actions are in [-1, 1]), in environment
    # transitions. DrQ-v2 starts at 1.0, which is close to random and makes a millimetre-scale
    # grasp almost impossible to execute while exploring; with demonstrations guiding the actor
    # a smaller, faster-shrinking noise works better for this task.
    stddev_schedule: str = "linear(0.5,0.1,500000)"
    stddev_clip: float = 0.3
    aug_pad: int = 4  # random-shift augmentation, pixels
    mae_coef: float = 1.0  # weight of the MV-MAE loss in the encoder update
    reward_pred_coef: float = 1.0
    mae_every: int = 1  # run the MV-MAE loss every N updates
    # Give the actor and critic the robot's own state as well as the images: tool-tip position and
    # orientation (robot base frame) and the two jaw angles, all from the robot's kinematics, as a real
    # da Vinci knows them. Nothing about the needle. Stereo disparity at 96 px shows ~0.1 px per cm of
    # height, so without this the policy cannot tell how far its jaws are above the pad.
    proprio: bool = False
    critic_grad_to_encoder: bool = True  # DrQ-v2 style; False = encoder learns from MV-MAE only
    # Scales the critic's gradient into the encoder (forward pass unchanged): a constant ("0.1")
    # or a schedule over environment steps ("linear(1.0,0.0,400000)"). Full strength (1.0) early
    # helps the encoder pick out the needle quickly; runs that kept it at 1.0 peaked and slid once
    # Q values grew, and runs without it (critic_grad_to_encoder=false) reached but never grasped.
    # Default 1.0 together with train.freeze_encoder_*: full critic shaping until the encoder is frozen.
    critic_encoder_grad_scale: str = "1.0"
    bc_coef: float = 0.4  # behaviour cloning on demo samples (TD3+BC, alpha=2.5); 0 disables
    max_grad_norm: float = 10.0
    amp: bool = True  # bfloat16 autocast for the transformer on CUDA


@dataclass
class TrainConfig:
    seed: int = 1
    total_env_steps: int = 2_000_000  # transitions summed over all parallel envs
    seed_env_steps: int = 10_000  # uniform random actions before learning starts
    mae_pretrain_updates: int = 2_000  # MV-MAE-only updates right after the random phase
    updates_per_env_step: float = 0.25  # gradient updates per collected transition
    replay_capacity: int = 100_000  # transitions; ~55 KB each at 2 x 96 x 96 RGB
    replay_device: str = "cuda"
    demo_path: str = ""  # .pt file from record_demos.py; empty = no demos
    demo_ratio: float = 0.25  # fraction of every batch drawn from the demos
    eval_every_env_steps: int = 50_000
    checkpoint_every_env_steps: int = 200_000
    # Freeze the encoder (no more MV-MAE or critic updates to it) once the robot has shown it can
    # do the task: eval success_any >= freeze_encoder_success in freeze_encoder_evals evaluations
    # in a row, or at freeze_encoder_at env steps at the latest. The actor and critic keep learning
    # on the fixed features. freeze_encoder_success=0 and freeze_encoder_at=0 disable it.
    freeze_encoder_success: float = 0.1
    freeze_encoder_evals: int = 2
    freeze_encoder_at: int = 450_000
    run_dir: str = "runs"


@dataclass
class LogConfig:
    wandb_project: str = "mvmae-drqv2-surgery"
    wandb_entity: str = ""
    wandb_mode: str = "online"  # online | offline | disabled
    run_name: str = ""
    log_every_updates: int = 250
    recon_every_updates: int = 5_000
    video: bool = True
    video_scale: int = 2  # nearest-neighbour upscaling of the 96 px stereo frames
    # Copy ckpt_latest / best / frozen / final to wandb Artifacts (at every checkpoint, at the
    # freeze and at the end) so they survive losing the machine.
    upload_checkpoints: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    mvmae: MVMAEConfig = field(default_factory=MVMAEConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(current: Any, value: Any, key: str) -> Any:
    """Convert a YAML/CLI value to the type of the default it replaces."""
    if isinstance(value, str) and isinstance(current, (int, float)) and not isinstance(current, bool):
        # YAML reads "1e6" as a string; accept numeric strings for numeric keys.
        try:
            value = float(value)
        except ValueError:
            raise TypeError(f"{key}: expected a number, got {value!r}") from None
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        raise TypeError(f"{key}: expected a bool, got {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise TypeError(f"{key}: expected an int, got {value!r}")
        return int(value)
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key}: expected a number, got {value!r}")
        return float(value)
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(current):
            raise TypeError(f"{key}: expected a list of {len(current)} values, got {value!r}")
        return tuple(_coerce(c, v, key) for c, v in zip(current, value))
    if isinstance(current, str):
        return str(value)
    return value


def _apply(cfg: Config, section: str, key: str, value: Any) -> None:
    if not hasattr(cfg, section):
        raise KeyError(f"unknown config section '{section}'")
    sub = getattr(cfg, section)
    if not hasattr(sub, key):
        raise KeyError(f"unknown config key '{section}.{key}'")
    setattr(sub, key, _coerce(getattr(sub, key), value, f"{section}.{key}"))


def load_config(path: str | None = None, overrides: list[str] | None = None, base: dict | None = None) -> Config:
    """Defaults <- `base` dict (e.g. the config stored in a checkpoint) <- YAML file <- overrides."""
    cfg = Config()
    sources = [base or {}]
    if path:
        sources.append(yaml.safe_load(Path(path).read_text()) or {})
    for data in sources:
        for section, values in data.items():
            for key, value in (values or {}).items():
                _apply(cfg, section, key, value)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        if dotted.count(".") != 1:
            raise ValueError(f"override '{item}' must look like section.key=value")
        section, key = dotted.split(".")
        _apply(cfg, section, key, yaml.safe_load(raw))
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    c, m, a, t = cfg.camera, cfg.mvmae, cfg.agent, cfg.train
    if c.height % m.patch_size or c.width % m.patch_size:
        raise ValueError("camera width/height must be divisible by mvmae.patch_size")
    if m.patch_size & (m.patch_size - 1) or m.patch_size < 2:
        raise ValueError("mvmae.patch_size must be a power of two >= 2")
    if m.embed_dim % m.encoder_heads or m.decoder_dim % m.decoder_heads:
        raise ValueError("embedding dims must be divisible by the number of heads")
    if not 0.5 <= m.mask_ratio < 1.0:
        raise ValueError("mvmae.mask_ratio must be in [0.5, 1) with two views (one view is always fully hidden)")
    if a.encoder not in ("mvmae", "pixels"):
        raise ValueError("agent.encoder must be 'mvmae' or 'pixels'")
    if a.pixel_downsample < 1 or c.height % a.pixel_downsample or c.width % a.pixel_downsample:
        raise ValueError("agent.pixel_downsample must be >= 1 and divide the image width and height")
    if not 0.0 <= t.demo_ratio < 1.0:
        raise ValueError("train.demo_ratio must be in [0, 1)")
    if a.nstep < 1 or m.frame_stack < 1:
        raise ValueError("agent.nstep and mvmae.frame_stack must be >= 1")
    if cfg.log.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError("log.wandb_mode must be online, offline or disabled")
    if cfg.camera.antialiasing not in ("Off", "FXAA", "TAA", "DLSS", "DLAA"):
        raise ValueError("camera.antialiasing must be Off, FXAA, TAA, DLSS or DLAA")
    if cfg.env.needle_asset not in ("sdf", "mesh"):
        raise ValueError("env.needle_asset must be 'sdf' or 'mesh'")
