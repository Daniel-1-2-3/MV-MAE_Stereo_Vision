from __future__ import annotations

import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Optional, Callable, Any

import jax
import jax.numpy as jnp

from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.replay_buffer import rb_init, rb_add, rb_sample_jit
from DrQv2_Architecture.env_wrappers import FrameStackWrapper, ActionRepeatWrapper
from Mujoco_Sim.mujoco_pick_env import StereoPickCube


# -------------------------
# Timing helpers
# -------------------------
@contextmanager
def timed(name: str, stats: Dict[str, float], sync: Optional[Callable[[], None]] = None):
    t0 = time.monotonic_ns()
    yield
    if sync is not None:
        sync()
    dt_ms = (time.monotonic_ns() - t0) / 1e6
    stats[name] = stats.get(name, 0.0) + dt_ms


def sync_tree(x: Any):
    """Force device completion for a pytree."""
    def _sync_one(y):
        return y.block_until_ready() if hasattr(y, "block_until_ready") else y

    jax.tree_util.tree_map(_sync_one, x)


# -------------------------
# Config
# -------------------------
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
    batch_size: int = 32
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

    # debugging / timing
    print_every_steps: int = 3          # print timing breakdown every N steps
    warmup_steps: int = 2               # do not print timing for first N steps
    sync_timings: bool = True           # True => accurate timings (forces device sync)
    heartbeat_every_steps: int = 50     # prints progress without syncing huge arrays


def make_env(cfg: TrainConfig, render_mode: str):
    env = StereoPickCube()
    env = ActionRepeatWrapper(env, num_repeats=cfg.action_repeat)           # repeat first
    env = FrameStackWrapper(env, num_frames=cfg.in_channels // 3)           # then stack
    return env


def main():
    cfg = TrainConfig()

    # Create train env (also used for eval)
    train_env = make_env(cfg, render_mode="rgb_array")

    act_dim = train_env.action_size
    obs_shape = (cfg.img_h_size, cfg.nviews * cfg.img_w_size, cfg.in_channels)

    rb = rb_init(
        capacity=cfg.buffer_size,
        obs_shape=obs_shape,
        action_shape=(act_dim,),
        reward_shape=(1,),
        discount_shape=(1,),
    )

    agent = DrQV2Agent.create(
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

    # Update step - JIT-able
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
    eval_rng = jax.random.PRNGKey(cfg.seed + 12345)
    rng, reset_key = jax.random.split(rng)
    env_state = train_env.reset(reset_key)

    # Keep episode stats ON DEVICE to avoid per-step device->host sync.
    ep_return = jnp.asarray(0.0, jnp.float32)
    ep_len = 0
    episode = 0

    wall_t0 = time.time()
    next_eval_at = cfg.eval_every_steps

    def do_eval(agent_state_in):
        nonlocal eval_rng, env_state
        saved_env_state = env_state

        returns = []
        for _ in range(cfg.num_eval_episodes):
            eval_rng, k = jax.random.split(eval_rng)
            st = train_env.reset(k)

            R = 0.0
            steps = 0

            while steps < cfg.episode_horizon:
                step_j = jnp.asarray(0, jnp.int32)

                action, _unused_state, _std = act_jit(
                    agent_state_in, st.obs, step_j, jnp.asarray(True)
                )
                st = train_env.step(st, action)

                # Eval is allowed to sync; it's infrequent.
                R += float(st.reward)
                if bool(st.done > 0.5):
                    break
                steps += 1

            returns.append(R)

        env_state = saved_env_state
        return sum(returns) / max(1, len(returns))

    # -------------------------
    # Warmup (compile once)
    # -------------------------
    # Warmup act_jit
    step0 = jnp.asarray(0, jnp.int32)
    _a, _st, _sd = act_jit(agent_state, env_state.obs, step0, jnp.asarray(False))
    if cfg.sync_timings:
        sync_tree((_a, _st, _sd))

    # Warmup env step (compiles renderer/custom call path)
    dummy_action = jnp.zeros((act_dim,), dtype=jnp.float32)
    st1 = train_env.step(env_state, dummy_action)
    if cfg.sync_timings:
        sync_tree(st1)

    # -------------------------
    # Train loop
    # -------------------------
    for step in range(cfg.total_steps):
        step_stats: Dict[str, float] = {}
        did_update = False

        if cfg.heartbeat_every_steps > 0 and (step % cfg.heartbeat_every_steps == 0):
            # Heartbeat: prints without forcing extra device sync beyond Python itself.
            print(f"progress step={step}")

        with timed("make_step_j", step_stats):
            step_j = jnp.asarray(step, jnp.int32)

        # ACT
        with timed(
            "act_jit",
            step_stats,
            sync=(lambda: sync_tree((action, stddev)) if cfg.sync_timings else None),
        ):
            action, agent_state, stddev = act_jit(
                agent_state, env_state.obs, step_j, jnp.asarray(False)
            )

        # ENV STEP
        with timed(
            "env_step",
            step_stats,
            sync=(lambda: sync_tree(next_env_state) if cfg.sync_timings else None),
        ):
            next_env_state = train_env.step(env_state, action)

        # REPLAY ADD
        with timed(
            "rb_add",
            step_stats,
            sync=(lambda: sync_tree((rb.ptr, rb.size)) if cfg.sync_timings else None),
        ):
            obs_b = env_state.obs  # (1,H,W2,C)
            act_b = action[None, ...]
            rew_b = jnp.asarray(next_env_state.reward, jnp.float32).reshape(1, 1)
            done_b = jnp.asarray(next_env_state.done > 0.5).reshape(1,)
            disc_b = (1.0 - done_b.astype(jnp.float32)).reshape(1, 1)
            rb = rb_add(rb, obs_b, act_b, rew_b, disc_b, done_b)

        # Episode bookkeeping: stay on device
        with timed("ep_bookkeeping", step_stats):
            ep_return = ep_return + jnp.asarray(next_env_state.reward, jnp.float32)
            ep_len += 1
            done_dev = (next_env_state.done > 0.5)  # device bool

        # UPDATE
        if step >= cfg.learning_starts:
            did_update = True
            with timed("rng_split", step_stats):
                rng, k = jax.random.split(rng)

            with timed(
                "rb_sample_jit",
                step_stats,
                sync=(lambda: None),
            ):
                batch = rb_sample_jit(
                    rb, k, batch_size=cfg.batch_size, nstep=cfg.nstep, gamma=cfg.gamma
                )
                if cfg.sync_timings:
                    sync_tree(batch)

            with timed(
                "update_jit",
                step_stats,
                sync=(lambda: sync_tree((action, stddev)) if cfg.sync_timings else None),
            ):
                agent_state, metrics = update_jit(agent_state, batch, step_j, stddev)

        # EPISODE END CHECK
        # Only one host sync here (bool()), unavoidable if you need Python control flow.
        with timed("episode_check", step_stats):
            done_host = bool(done_dev)

        if done_host or (ep_len >= cfg.episode_horizon):
            episode += 1
            # Only sync ep_return when you print (infrequent).
            ep_ret_host = float(ep_return)
            print(
                f"[train] episode={episode} step={step} ep_len={ep_len} ep_return={ep_ret_host:.3f}"
            )
            ep_return = jnp.asarray(0.0, jnp.float32)
            ep_len = 0
            with timed("reset_rng", step_stats):
                rng, reset_key = jax.random.split(rng)
            with timed(
                "env_reset",
                step_stats,
                sync=(lambda: sync_tree(next_env_state) if cfg.sync_timings else None),
            ):
                next_env_state = train_env.reset(reset_key)

        env_state = next_env_state

        # EVAL (infrequent)
        if (step + 1) >= next_eval_at:
            with timed("eval_total", step_stats):
                avg_R = do_eval(agent_state)

            elapsed = time.time() - wall_t0
            sps = (step + 1) / max(1e-9, elapsed)
            print(f"[eval] step={step+1} avg_return={avg_R:.3f} steps_per_sec={sps:.1f}")
            next_eval_at += cfg.eval_every_steps

        # Print timing breakdown
        if step >= cfg.warmup_steps and (cfg.print_every_steps > 0) and (step % cfg.print_every_steps == 0):
            order = [
                "make_step_j",
                "act_jit",
                "env_step",
                "rb_add",
                "ep_bookkeeping",
                "rng_split",
                "rb_sample_jit",
                "update_jit",
                "episode_check",
                "reset_rng",
                "env_reset",
                "eval_total",
            ]
            parts = [f"{k}={step_stats[k]:.2f}ms" for k in order if k in step_stats]
            total = sum(step_stats.values())
            print("timing:", "  ".join(parts), f"  TOTAL={total:.2f}ms", f"  updated={did_update}")

    print("Done training")


if __name__ == "__main__":
    main()
