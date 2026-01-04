import os

# Match debug.py env setup as closely as possible (must be set before JAX creates a client)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")
xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in xla_flags:
    xla_flags = (xla_flags + " --xla_gpu_triton_gemm_any=True").strip()
os.environ["XLA_FLAGS"] = xla_flags

import torch
import jax
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio

from Mujoco_Sim.pick_env import StereoPickCube
from Mujoco_Sim.brax_wrapper import RSLRLBraxWrapper
from DrQv2_Architecture.drqv2 import DrQv2Agent


def _resize_uint8_hwc(frame_uint8_hwc: torch.Tensor, out_size: int) -> torch.Tensor:
    """frame: [H,W,3] uint8 -> resized [out_size,out_size,3] uint8 (Torch-only)."""
    x = frame_uint8_hwc.permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    x = torch.nn.functional.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.clamp(0, 255).byte()
    return x.squeeze(0).permute(1, 2, 0)  # [out,out,3] uint8


def eval_and_record(env, agent, num_steps: int = 10_000, video_root: Path | None = None):
    agent.train(training=False)

    if video_root is None:
        video_root = Path(".")
    out_dir = video_root / "eval_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "eval.gif"

    render_size = 256
    fps = 20
    frames = []

    obs_td = env.reset()
    obs = obs_td["state"]  # [B,H,2W,3] uint8 (from wrapper)

    ep_rew = torch.zeros((agent.num_envs,), device=agent.device, dtype=torch.float32)
    ep_len = torch.zeros((agent.num_envs,), device=agent.device, dtype=torch.int32)

    for t in range(num_steps):
        with torch.no_grad():
            action = agent.act(obs, step=t, eval_mode=True)

        next_obs_td, rew_t, done_t, info = env.step(action)
        next_obs = next_obs_td["state"]

        frame0 = next_obs[0]  # [H,2W,3]
        if frame0.dtype != torch.uint8:
            frame0 = (frame0 * 255.0).clamp(0, 255).to(torch.uint8)
        frame0 = _resize_uint8_hwc(frame0, render_size)
        frames.append(frame0.detach().cpu().numpy())

        rew_t = rew_t.reshape(-1).to(torch.float32)
        done_t = done_t.reshape(-1).to(torch.bool)

        ep_rew += rew_t
        ep_len += 1

        if done_t.any():
            finished = done_t
            mean_finished_rew = ep_rew[finished].mean().item() if finished.any() else float("nan")
            mean_finished_len = ep_len[finished].float().mean().item() if finished.any() else float("nan")
            print(
                f"[EVAL] done->reset at step {t}: "
                f"mean_ep_rew={mean_finished_rew:.3f}, mean_ep_len={mean_finished_len:.1f}"
            )

            obs_td = env.reset()
            obs = obs_td["state"]
            ep_rew.zero_()
            ep_len.zero_()
        else:
            obs = next_obs

    imageio.mimsave(gif_path.as_posix(), frames, fps=fps)
    print(f"[EVAL] Saved {gif_path}")


def main():
    num_envs = 32
    episode_length = 200
    total_timesteps = 1_000_000
    learning_starts = 50_000
    seed = 1

    render_trajectory = []
    def render_callback(_, state):
        render_trajectory.append(state)

    # Keep your GPU detection, but JAX env vars are already set above
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    device_rank = 0 if has_gpu else None

    raw_env = StereoPickCube()
    print("Loaded env")

    brax_env = RSLRLBraxWrapper(
        raw_env,
        num_envs,
        seed,
        episode_length,
        1,
        render_callback=render_callback,
        randomization_fn=None,
        device_rank=device_rank,
    )
    print("wrapped")

    runner = DrQv2Agent(brax_env, num_envs=num_envs, episode_length=episode_length)
    print("made agent")

    start_time = datetime.now()
    runner.learn(total_timesteps=total_timesteps, learning_starts=learning_starts)
    end_time = datetime.now()
    elapsed_s = (end_time - start_time).total_seconds()

    total_steps = total_timesteps * num_envs
    fps = total_steps / elapsed_s
    print(f"[FPS] {fps:.2f} env-steps/sec")
    print(f"[Timing] {elapsed_s:.2f}s for {total_steps:,} steps")

    video_root = Path(".")
    eval_and_record(brax_env, runner, num_steps=10_000, video_root=video_root)


if __name__ == "__main__":
    main()
