import torch
import jax
from datetime import datetime
from pathlib import Path
from Mujoco_Sim.pick_env import StereoPickCube
from Mujoco_Sim.brax_wrapper import RSLRLBraxWrapper
from DrQv2_Architecture.drqv2 import DrQv2Agent
from DrQv2_Architecture.video import VideoRecorder

def eval_and_record(env, agent, num_steps: int = 10_000, video_root: Path | None = None):
    """
    Runs an evaluation rollout for num_steps env-steps (vectorized),
    resets when any env is done, and records a GIF from env_id=0.
    """
    # Agent eval mode
    agent.train(training=False)

    recorder = VideoRecorder(video_root, render_size=256, fps=20)
    recorder.init(env, enabled=True)

    obs_td = env.reset()
    obs = obs_td["state"]

    # Track episodic reward for logging (vectorized)
    ep_rew = torch.zeros((agent.num_envs,), device=agent.device, dtype=torch.float32)
    ep_len = torch.zeros((agent.num_envs,), device=agent.device, dtype=torch.int32)

    for t in range(num_steps):
        with torch.no_grad():
            action = agent.act(obs, step=t, eval_mode=True)

        next_obs_td, rew_t, done_t, info = env.step(action)
        next_obs = next_obs_td["state"]

        # Record one frame per step (env_id=0)
        recorder.record(env)

        # bookkeeping
        rew_t = rew_t.reshape(-1).to(torch.float32)
        done_t = done_t.reshape(-1).to(torch.bool)

        ep_rew += rew_t
        ep_len += 1

        # Terminate/reset logic:
        # Your wrapper only has reset() for the full batch, not per-env,
        # so if ANY env is done we reset the whole batch.
        if done_t.any():
            finished = done_t
            mean_finished_rew = ep_rew[finished].mean().item() if finished.any() else float("nan")
            mean_finished_len = ep_len[finished].float().mean().item() if finished.any() else float("nan")
            print(f"[EVAL] done->reset at step {t}: mean_ep_rew={mean_finished_rew:.3f}, mean_ep_len={mean_finished_len:.1f}")

            obs_td = env.reset()
            obs = obs_td["state"]
            ep_rew.zero_()
            ep_len.zero_()
        else:
            obs = next_obs

    recorder.save("eval")
    print("[EVAL] Saved eval_video/eval.gif")

def main():
    num_envs = 32
    episode_length = 200
    total_timesteps = 1_000_000
    learning_starts = 50_000
    seed = 1

    # Store environment states during rendering (optional)
    render_trajectory = []
    def render_callback(_, state):
        render_trajectory.append(state)

    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    device = "cuda:0" if has_gpu else "cpu"
    device_rank = 0 if has_gpu else None

    raw_env = StereoPickCube(render_batch_size=num_envs)
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
    print("wrappeed")

    runner = DrQv2Agent(brax_env, num_envs=num_envs)
    print("made agent")

    start_time = datetime.now()
    runner.learn(total_timesteps=total_timesteps, learning_starts=learning_starts)
    end_time = datetime.now()
    elapsed_s = (end_time - start_time).total_seconds()

    total_steps = total_timesteps * num_envs
    fps = total_steps / elapsed_s
    print(f"[FPS] {fps:.2f} env-steps/sec")
    print(f"[Timing] {elapsed_s:.2f}s for {total_steps:,} steps")

    # 10_000 vectorized steps of eval
    video_root = Path(".") # will create ./eval_video/
    eval_and_record(brax_env, runner, num_steps=10_000, video_root=video_root)

if __name__ == "__main__":
    main()