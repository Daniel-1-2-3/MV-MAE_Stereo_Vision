from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.replay_buffer import rb_init, rb_add, rb_sample_jit
from DrQv2_Architecture.env_wrappers import FrameStackWrapper, ActionRepeatWrapper

from Mujoco_Sim.mujoco_pick_env import StereoPickCube

@dataclass
class TrainConfig:
    seed: int = 1

    # env
    img_h_size: int = 64
    img_w_size: int = 64
    nviews: int = 2
    action_repeat: int = 2
    episode_horizon: int = 300
    discount: float = 0.99

    # obs stacking (in_channels = 3 * K)
    in_channels: int = 9  # e.g., K=3 frames

    # replay
    buffer_size: int = 100_000
    batch_size: int = 64
    nstep: int = 3
    gamma: float = 0.99

    # training schedule
    total_steps: int = 200_000
    learning_starts: int = 2_000
    num_expl_steps: int = 2_000

    # update cadence
    update_every_steps: int = 2
    update_mvmae_every_steps: int = 10

    # actor noise schedule (DrQV2Agent.act uses linear schedule)
    std_start: float = 1.0
    std_end: float = 0.1
    std_duration: int = 500_000
    stddev_clip: float = 0.3

    # losses / target update
    coef_mvmae: float = 0.005
    critic_target_tau: float = 0.001

    # eval
    eval_every_steps: int = 10_000
    num_eval_episodes: int = 10

def make_env(cfg: TrainConfig, render_mode: str):
    env = StereoPickCube(
        render_mode=render_mode,
        img_h_size=cfg.img_h_size,
        img_w_size=cfg.img_w_size,
        discount=cfg.discount,
        max_path_length=cfg.episode_horizon,
    )
    # apply ActionRepeat first, then FrameStack on top.
    env = ActionRepeatWrapper(env, num_repeats=cfg.action_repeat)
    env = FrameStackWrapper(env, num_frames=cfg.in_channels // 3)
    return env

def main():
    cfg = TrainConfig()

    # Create train and eval envs
    train_env = make_env(cfg, render_mode="rgb_array")
    eval_env = make_env(cfg, render_mode="rgb_array")

    act_dim = train_env.action_size
    obs_shape = (cfg.img_h_size, cfg.nviews * cfg.img_w_size, cfg.in_channels)

    rb = rb_init(
        capacity=cfg.buffer_size,
        obs_shape=obs_shape,
        action_shape=(act_dim,),
        reward_shape=(1,),
        discount_shape=(1,),
    )

    agent = DrQV2Agent(
        action_shape=(act_dim,),
        nviews=cfg.nviews,
        mvmae_patch_size=8,
        mvmae_encoder_embed_dim=256,
        mvmae_decoder_embed_dim=128,
        mvmae_encoder_heads=16,
        mvmae_decoder_heads=16,
        in_channels=cfg.in_channels,
        img_h_size=cfg.img_h_size,
        img_w_size=cfg.img_w_size,
        masking_ratio=0.75,
        feature_dim=100,
        hidden_dim=1024,
        lr=1e-4,
    )
    agent_state = agent.init_state(cfg.seed)

    # JIT wrapper for agent.act
    act_jit = jax.jit(
        lambda state, obs, step, eval_mode: DrQV2Agent.act(
            agent=agent,
            state=state,
            obs=obs,
            step=step,
            eval_mode=eval_mode,
            num_expl_steps=cfg.num_expl_steps,
            std_start=cfg.std_start,
            std_end=cfg.std_end,
            std_duration=cfg.std_duration,
            clip_pre_tanh=cfg.stddev_clip,
            deterministic_encoder=False,
        )
    )

    # Update step - JIT-able (agent captured; python ints are static via closure)
    update_jit = jax.jit(
        lambda state, batch, step, stddev: DrQV2Agent.update_step(
            agent=agent,
            state=state,
            batch=batch,
            step=step,
            update_every_steps=cfg.update_every_steps,
            update_mvmae_every_steps=cfg.update_mvmae_every_steps,
            coef_mvmae=cfg.coef_mvmae,
            critic_target_tau=cfg.critic_target_tau,
            stddev=stddev,
            stddev_clip=cfg.stddev_clip,
        )
    )

    # training state
    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_key = jax.random.split(rng)
    env_state = train_env.reset(reset_key)

    ep_return = 0.0
    ep_len = 0
    episode = 0

    t0 = time.time()
    next_eval_at = cfg.eval_every_steps

    def do_eval(agent_state_in):
        nonlocal rng
        returns = []
        for _ in range(cfg.num_eval_episodes):
            rng, k = jax.random.split(rng)
            st = eval_env.reset(k)
            done = False
            R = 0.0
            steps = 0
            while not bool(done) and steps < cfg.episode_horizon:
                step_j = jnp.asarray(0, jnp.int32)  # not used for schedule in eval (but required)
                action, agent_state_tmp, _std = act_jit(
                    agent_state_in, st.obs, step_j, jnp.asarray(True)
                )
                # keep agent_state_in unchanged in eval (don’t advance RNG permanently)
                st = eval_env.step(st, action)
                R += float(st.reward)
                done = bool(st.done > 0.5)
                steps += 1
            returns.append(R)
        return sum(returns) / max(1, len(returns))

    # train / eval alternating 
    for step in range(cfg.total_steps):
        step_j = jnp.asarray(step, jnp.int32)

        action, agent_state, stddev = act_jit(agent_state, env_state.obs, step_j, jnp.asarray(False)) # Take action
        next_env_state = train_env.step(env_state, action)

        # store transition
        # replay expects obs: [B,H,W2,C], action: [B,A], reward/discount: [B,1], done: [B]
        obs_b = env_state.obs  # already (1,H,W2,C)
        act_b = action[None, ...]
        rew_b = jnp.asarray(next_env_state.reward, jnp.float32).reshape(1, 1)
        done_b = jnp.asarray(next_env_state.done > 0.5).reshape(1,)
        disc_b = (1.0 - done_b.astype(jnp.float32)).reshape(1, 1)
        rb = rb_add(rb, obs_b, act_b, rew_b, disc_b, done_b)

        ep_return += float(next_env_state.reward)
        ep_len += 1

        # update if past learning_starts
        if step >= cfg.learning_starts:
            rng, k = jax.random.split(rng)
            batch = rb_sample_jit(rb, k, batch_size=cfg.batch_size, nstep=cfg.nstep, gamma=cfg.gamma)
            agent_state, metrics = update_jit(agent_state, batch, step_j, stddev)

        # --- episode handling ---
        if bool(next_env_state.done > 0.5) or ep_len >= cfg.episode_horizon:
            episode += 1
            print(f"[train] episode={episode} step={step} ep_len={ep_len} ep_return={ep_return:.3f}")
            ep_return = 0.0
            ep_len = 0
            rng, reset_key = jax.random.split(rng)
            next_env_state = train_env.reset(reset_key)

        env_state = next_env_state

        # eval chunk
        if (step + 1) >= next_eval_at:
            avg_R = do_eval(agent_state)
            elapsed = time.time() - t0
            sps = (step + 1) / max(1e-9, elapsed)
            print(f"[eval] step={step+1} avg_return={avg_R:.3f} steps_per_sec={sps:.1f}")
            next_eval_at += cfg.eval_every_steps

    print("Done training")

if __name__ == "__main__":
    main()
