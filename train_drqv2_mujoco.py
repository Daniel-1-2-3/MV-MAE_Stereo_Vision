# train_sac_like_ppo_scan.py
# Matches the structure of train_drqv2_mujoco_2.py (PPO reference) but runs SAC/DrQv2.
# Key: env.step (and renderer.render inside it) executes inside jax.lax.scan under jit.

from __future__ import annotations

import datetime
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from absl import app, flags, logging

import jax
import jax.numpy as jnp
from etils import epath

import Custom_Mujoco_Playground
from Custom_Mujoco_Playground import registry

# Your SAC/DrQv2 + replay + wrappers
from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.replay_buffer import rb_init, rb_add, rb_sample_jit
from DrQv2_Architecture.env_wrappers import FrameStackWrapper, ActionRepeatWrapper


# -------------------------
# Env + run flags (mirror PPO reference style)
# -------------------------
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "PandaPickCubeCartesian",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_VISION = flags.DEFINE_boolean("vision", True, "Use vision input")
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Total env steps")


# -------------------------
# SAC/DrQ flags (keep minimal; tune as needed)
# -------------------------
_IMG_H = flags.DEFINE_integer("img_h", 64, "Input image height")
_IMG_W = flags.DEFINE_integer("img_w", 64, "Single-view width (final fused is 2W for stereo)")
_NVIEWS = flags.DEFINE_integer("nviews", 2, "Number of views fused side-by-side")
_STACK_K = flags.DEFINE_integer("stack_k", 3, "Frame stack count (channels = 3*K)")
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 2, "Action repeat")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 300, "Episode horizon/length")

_BUFFER_SIZE = flags.DEFINE_integer("buffer_size", 100_000, "Replay capacity")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "SAC batch size")
_NSTEP = flags.DEFINE_integer("nstep", 3, "N-step returns")
_GAMMA = flags.DEFINE_float("gamma", 0.99, "Discount factor")

_NUM_EXPL_STEPS = flags.DEFINE_integer("num_expl_steps", 2_000, "Uniform random exploration steps")
_STD_START = flags.DEFINE_float("std_start", 1.0, "Policy std schedule start")
_STD_END = flags.DEFINE_float("std_end", 0.1, "Policy std schedule end")
_STD_DURATION = flags.DEFINE_integer("std_duration", 100_000, "Std schedule duration")
_STDDEV_CLIP = flags.DEFINE_float("stddev_clip", 0.3, "Pre-tanh clip (Torch parity)")

# Update cadence (mirrors your old TrainConfig usage)
_UPDATE_EVERY = flags.DEFINE_integer("update_every_steps", 2, "Update every N env steps")
_UPDATE_MVMAE_EVERY = flags.DEFINE_integer("update_mvmae_every_steps", 4, "Update MV-MAE every N env steps")
_COEF_MVMAE = flags.DEFINE_float("coef_mvmae", 1.0, "MV-MAE loss coefficient")
_CRITIC_TAU = flags.DEFINE_float("critic_target_tau", 0.01, "Critic target EMA tau")

# Scan/unroll (THIS is the reference-style structural change)
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 32, "Steps per jitted scan chunk")

# Logging
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 20_480, "Print speed every N steps")
_SYNC_TIMINGS = flags.DEFINE_boolean("sync_timings", False, "If true, block for accurate timings")


# -------------------------
# Runtime env vars (mirror PPO reference)
# -------------------------
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


# -------------------------
# Helpers
# -------------------------
def _maybe_block(x):
    if hasattr(x, "block_until_ready"):
        return x.block_until_ready()
    return x

def _sync_tree(x):
    jax.tree_util.tree_map(_maybe_block, x)

def _extract_obs(state) -> jnp.ndarray:
    # Common keys used by playground wrappers: pixels/view_0 etc.
    # Your DrQV2Agent expects fused/stacked obs already from your env wrappers.
    # Here we assume your env wrapper produces a tensor obs (not dict).
    if isinstance(state.obs, dict):
        # If it is a dict, prefer 'pixels/view_0' if present, else first item.
        if "pixels/view_0" in state.obs:
            return state.obs["pixels/view_0"]
        return next(iter(state.obs.values()))
    return state.obs

def _extract_reward_done(state) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Many playground states have .reward and .done as jnp arrays.
    rew = getattr(state, "reward", None)
    done = getattr(state, "done", None)
    if rew is None:
        # fallback: might be in metrics or info
        rew = state.metrics.get("reward", jnp.asarray(0.0, jnp.float32)) if hasattr(state, "metrics") else jnp.asarray(0.0, jnp.float32)
    if done is None:
        done = state.metrics.get("done", jnp.asarray(False)) if hasattr(state, "metrics") else jnp.asarray(False)
    # Ensure scalar shapes are consistent
    rew = jnp.asarray(rew)
    done = jnp.asarray(done).astype(jnp.bool_)
    return rew, done

def make_env() -> Any:
    # Match reference: config via registry.get_default_config / registry.load
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value

    if _VISION.value:
        env_cfg.vision = True
        # IMPORTANT: single-world training here
        env_cfg.vision_config.render_batch_size = 1

    env = registry.load(_ENV_NAME.value, config=env_cfg)

    # Mirror your pipeline: action-repeat, then frame-stack
    env = ActionRepeatWrapper(env, num_repeats=_ACTION_REPEAT.value)
    env = FrameStackWrapper(env, num_frames=_STACK_K.value)

    return env, env_cfg


# -------------------------
# Main
# -------------------------
def main(argv):
    del argv

    env, env_cfg = make_env()

    # Experiment name + logdir (mirror reference)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{_ENV_NAME.value}-SAC-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"

    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {exp_name}")
    print(f"Logs are being stored in: {logdir}")

    # Save env config like reference
    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else dict(env_cfg), fp, indent=4)

    # Infer action/obs shapes
    act_dim = env.action_size
    in_channels = 3 * _STACK_K.value
    obs_shape = (_IMG_H.value, _NVIEWS.value * _IMG_W.value, in_channels)

    # Replay buffer on device
    rb = rb_init(
        capacity=_BUFFER_SIZE.value,
        obs_shape=(obs_shape[0], obs_shape[1], obs_shape[2]),
        action_dim=act_dim,
    )

    # Agent
    agent = DrQV2Agent.create(
        action_shape=(act_dim,),
        nviews=_NVIEWS.value,
        in_channels=in_channels,
        img_h_size=_IMG_H.value,
        img_w_size=_IMG_W.value,
    )
    agent_state = agent.init_state(_SEED.value)

    # Jitted policy act (Torch-parity behavior is inside agent.act)
    act_jit = jax.jit(
        lambda st, obs, step, eval_mode: DrQV2Agent.act(
            agent=agent,
            state=st,
            obs=obs,
            step=step,
            eval_mode=eval_mode,
            num_expl_steps=_NUM_EXPL_STEPS.value,
            std_start=_STD_START.value,
            std_end=_STD_END.value,
            std_duration=_STD_DURATION.value,
            clip_pre_tanh=_STDDEV_CLIP.value,
            deterministic_encoder=False,
        )
    )

    # Jitted update step (handles update_every gating inside)
    update_jit = jax.jit(
        lambda st, batch, step, stddev: DrQV2Agent.update_step(
            agent=agent,
            state=st,
            batch=batch,
            step=step,
            update_every_steps=_UPDATE_EVERY.value,
            update_mvmae_every_steps=_UPDATE_MVMAE_EVERY.value,
            coef_mvmae=_COEF_MVMAE.value,
            critic_target_tau=_CRITIC_TAU.value,
            stddev=stddev,
            stddev_clip=_STDDEV_CLIP.value,
        )
    )

    # Reset env
    rng = jax.random.PRNGKey(_SEED.value)
    rng, reset_key = jax.random.split(rng)
    env_state = env.reset(reset_key)

    # Device episode stats (avoid python sync)
    ep_return = jnp.asarray(0.0, jnp.float32)
    ep_len = jnp.asarray(0, jnp.int32)

    # ------------------------------------------------------------
    # Reference-style compiled rollout: jitted lax.scan chunk
    # ------------------------------------------------------------
    def rollout_chunk(carry, _):
        """
        One step of: act -> env.step -> rb_add -> maybe update -> maybe reset
        Runs inside lax.scan under jit, like reference rollouts.
        """
        (env_state, agent_state, rb, rng, step, ep_return, ep_len) = carry

        obs = _extract_obs(env_state)

        # Policy action
        action, agent_state2, stddev = act_jit(agent_state, obs, step, jnp.asarray(False))

        # Env step
        next_env_state = env.step(env_state, action)

        # Reward/done
        rew, done = _extract_reward_done(next_env_state)
        # discount for rb (1 - done) * gamma
        disc = (jnp.asarray(1.0, jnp.float32) - done.astype(jnp.float32)) * jnp.asarray(_GAMMA.value, jnp.float32)

        # Replay add (B=1)
        rb2 = rb_add(
            rb,
            obs_b=obs[None, ...],
            action_b=action[None, ...],
            reward_b=rew[None, ...],
            discount_b=disc[None, ...],
            done_b=done[None, ...],
        )

        # Sample + update (still inside jit; batch_size/nstep are static)
        rng2, sample_key = jax.random.split(rng)
        batch = rb_sample_jit(rb2, sample_key, batch_size=_BATCH_SIZE.value, nstep=_NSTEP.value, gamma=_GAMMA.value)

        agent_state3, metrics = update_jit(agent_state2, batch, step, stddev)

        # Episode stats update
        ep_return2 = ep_return + rew.astype(jnp.float32)
        ep_len2 = ep_len + jnp.asarray(1, jnp.int32)

        # Auto-reset on done (still inside jit)
        def _do_reset(args):
            rng_in, = args
            rng_out, k = jax.random.split(rng_in)
            s0 = env.reset(k)
            return s0, rng_out, jnp.asarray(0.0, jnp.float32), jnp.asarray(0, jnp.int32)

        def _no_reset(args):
            rng_in, = args
            return next_env_state, rng_in, ep_return2, ep_len2

        env_state3, rng3, ep_return3, ep_len3 = jax.lax.cond(
            done,
            _do_reset,
            _no_reset,
            (rng2,),
        )

        step2 = step + jnp.asarray(1, jnp.int32)

        new_carry = (env_state3, agent_state3, rb2, rng3, step2, ep_return3, ep_len3)
        # Trajectory outputs (optional)
        out = (rew, done, stddev, metrics)
        return new_carry, out

    rollout_scan_jit = jax.jit(
        lambda carry: jax.lax.scan(rollout_chunk, carry, None, length=_UNROLL_LENGTH.value),
    )

    # ------------------------------------------------------------
    # Speed logging (mirror reference)
    # ------------------------------------------------------------
    times = [time.monotonic()]
    speed_start_t = times[0]
    speed_last_t = times[0]
    speed_last_steps = 0

    total_steps = int(_NUM_TIMESTEPS.value)
    step0 = jnp.asarray(0, jnp.int32)

    carry = (env_state, agent_state, rb, rng, step0, ep_return, ep_len)

    # Drive training by chunks (Python only launches compiled scan)
    # This is analogous to reference code calling a compiled train_fn and using progress callbacks.
    while int(_maybe_block(carry[4])) < total_steps:
        t0 = time.monotonic()

        carry, traj = rollout_scan_jit(carry)

        if _SYNC_TIMINGS.value:
            _sync_tree((carry, traj))

        t1 = time.monotonic()

        # progress logging every LOG_EVERY_STEPS (like reference)
        cur_step = int(_maybe_block(carry[4]))
        if cur_step - speed_last_steps >= int(_LOG_EVERY_STEPS.value):
            now_t = time.monotonic()
            dt = now_t - speed_last_t
            dsteps = cur_step - speed_last_steps
            if dt > 0 and dsteps > 0:
                sps = dsteps / dt
                ms_per_step = (dt / dsteps) * 1e3
                avg_sps = cur_step / max(1e-9, (now_t - speed_start_t))
                print(
                    f"[speed] steps={cur_step}  "
                    f"inst={sps:,.1f} steps/s  "
                    f"inst={ms_per_step:.3f} ms/step  "
                    f"avg={avg_sps:,.1f} steps/s"
                )
            speed_last_t = now_t
            speed_last_steps = cur_step

    print("Done training.")


if __name__ == "__main__":
    app.run(main)
