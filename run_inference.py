import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MUJOCO_PLATFORM", "glfw")
os.environ.setdefault("PYOPENGL_PLATFORM", "glfw")
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import argparse
import numpy as np
import torch
from pathlib import Path
from gymnasium.spaces import Box

import DrQv2_Architecture.utils as utils
from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.video import VideoRecorder
from DrQv2_Architecture.env_wrappers import (
    ExtendedTimeStepWrapper,
    ActionRepeatWrapper,
    FrameStackWrapper,
)
from Sawyer_Sim.sawyer_stereo_env import SawyerReachEnvV3

def make_agent(args, action_space, device) -> DrQV2Agent:
    return DrQV2Agent(
        action_shape=(action_space.shape[0],),
        device=device,
        lr=args.lr,
        nviews=args.nviews,
        mvmae_patch_size=args.mvmae_patch_size,
        mvmae_encoder_embed_dim=args.mvmae_encoder_embed_dim,
        mvmae_decoder_embed_dim=args.mvmae_decoder_embed_dim,
        mvmae_encoder_heads=args.mvmae_encoder_heads,
        mvmae_decoder_heads=args.mvmae_decoder_heads,
        in_channels=args.in_channels,
        img_h_size=args.img_h_size,
        img_w_size=args.img_w_size,
        masking_ratio=args.masking_ratio,
        coef_mvmae=args.coef_mvmae,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        critic_target_tau=args.critic_target_tau,
        num_expl_steps=0, # no exploration during inference
        update_every_steps=args.update_every_steps,
        update_mvmae_every_steps=args.update_mvmae_every_steps,
        stddev_schedule=args.stddev_schedule,
        stddev_clip=args.stddev_clip,
        use_tb=False,
    )

def make_env(args, obs_space, action_space, render_mode="rgb_array"):
    sawyer_obs_space = Box(
        low=np.float32(-4.0),
        high=np.float32(4.0),
        shape=(args.img_h_size, args.nviews * args.img_w_size, args.in_channels // 3),
        dtype=np.float32,
    )
    env = SawyerReachEnvV3(
        render_mode=render_mode,
        img_height=args.img_h_size,
        img_width=args.img_w_size,
        max_path_length=args.episode_horizon,
        obs_space=sawyer_obs_space,
        action_space=action_space,
        discount=args.discount,
    )
    env = FrameStackWrapper(env, num_frames=args.in_channels // 3)
    env = ActionRepeatWrapper(env, num_repeats=args.action_repeat)
    env = ExtendedTimeStepWrapper(env)
    return env

def load_agent(agent: DrQV2Agent, path: str):
    checkpoint = torch.load(path, map_location=agent.device)
    agent.mvmae.load_state_dict(checkpoint["mvmae"])
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.critic_target.load_state_dict(checkpoint["critic_target"])
    print(f"Loaded checkpoint from '{path}'")

def run_inference(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    action_space = Box(np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]), dtype=np.float32)
    overall_obs_space = Box(
        low=np.float32(-4.0),
        high=np.float32(4.0),
        shape=(args.img_h_size, args.nviews * args.img_w_size, args.in_channels),
        dtype=np.float32,
    )

    agent = make_agent(args, action_space, device)
    load_agent(agent, args.checkpoint)

    env = make_env(args, overall_obs_space, action_space, render_mode=args.render_mode)
    video_recorder = VideoRecorder(Path(args.save_dir)) if args.save_video else None

    total_reward = 0.0
    total_success = 0
    step_counts = []

    for episode in range(args.num_episodes):
        time_step = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_success = False

        if video_recorder is not None:
            video_recorder.init(env, enabled=True)

        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step=0, eval_mode=True)

            time_step = env.step(action)
            ep_reward += float(time_step.reward[0])
            ep_steps += 1

            if video_recorder is not None:
                video_recorder.record(env)

            # SawyerReachEnvV3.evaluate_state sets success when reach_dist <= 0.05
            if hasattr(env, '_env') and hasattr(env._env, '_env'):
                info = getattr(env._env._env._env, '_last_info', {})
                if info.get("success", False):
                    ep_success = True

        if video_recorder is not None:
            video_recorder.save(f"episode_{episode}.mp4")
            print(f"  Saved video: episode_{episode}.mp4")

        total_reward += ep_reward
        total_success += int(ep_success)
        step_counts.append(ep_steps)

        print(
            f"Episode {episode + 1:>3}/{args.num_episodes} | "
            f"Reward: {ep_reward:>8.3f} | "
            f"Steps: {ep_steps:>4} | "
            f"Success: {ep_success}"
        )

    print(f"Episodes        : {args.num_episodes}")
    print(f"Mean reward     : {total_reward / args.num_episodes:.3f}")
    print(f"Success rate    : {total_success / args.num_episodes * 100:.1f}%")
    print(f"Mean ep. length : {np.mean(step_counts) * args.action_repeat:.1f} frames")

def get_args():
    parser = argparse.ArgumentParser(description="DrQV2 inference / evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved agent_weights.pt")

    # Eval settings
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--render_mode", type=str, default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument("--save_video", action="store_true", help="Save mp4 of the first episode in --save_dir")
    parser.add_argument("--save_dir", type=str, default="inference_results")
    parser.add_argument("--device", type=str, default=None)

    # Must match training hyperparameters exactly
    parser.add_argument("--episode_horizon", type=int, default=300)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    # MV-MAE
    parser.add_argument("--nviews", type=int, default=2)
    parser.add_argument("--mvmae_patch_size", type=int, default=8)
    parser.add_argument("--mvmae_encoder_embed_dim", type=int, default=256)
    parser.add_argument("--mvmae_decoder_embed_dim", type=int, default=128)
    parser.add_argument("--mvmae_encoder_heads", type=int, default=16)
    parser.add_argument("--mvmae_decoder_heads", type=int, default=16)
    parser.add_argument("--masking_ratio", type=float, default=0.75)
    parser.add_argument("--coef_mvmae", type=float, default=0.005)
    # Actor + Critic
    parser.add_argument("--feature_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--critic_target_tau", type=float, default=0.001)
    parser.add_argument("--update_every_steps", type=int, default=2)
    parser.add_argument("--update_mvmae_every_steps", type=int, default=10)
    parser.add_argument("--stddev_schedule", type=str, default="linear(1.0,0.1,500000)")
    parser.add_argument("--stddev_clip", type=float, default=0.3)
    # Image
    parser.add_argument("--in_channels", type=int, default=9)
    parser.add_argument("--img_h_size", type=int, default=64)
    parser.add_argument("--img_w_size", type=int, default=64)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    run_inference(args)