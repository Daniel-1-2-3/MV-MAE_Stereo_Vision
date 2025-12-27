# train_sac_scan_with_your_env.py
from __future__ import annotations

import datetime
import json
import os
import time
import warnings
from typing import Any, Tuple

from absl import app, flags, logging
import jax
import jax.numpy as jp
from etils import epath

# === YOUR ENV ===
from Mujoco_Sim.mujoco_pick_env import StereoPickCube, default_config  # from your mujoco_pick_env.py :contentReference[oaicite:1]{index=1}

# === YOUR PIPELINE ===
from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.replay_buffer import rb_init, rb_add, rb_sample_jit
from DrQv2_Architecture.env_wrappers import FrameStackWrapper, ActionRepeatWrapper


# -------------------------
# Match the reference file’s “flags + setup” style
# -------------------------
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for experiment name")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Total env steps")

# Vision / env
_RENDER_H = flags.DEFINE_integer("render_height", 64, "Render height")
_RENDER_W = flags.DEFINE_integer("render_width", 64, "Render width")
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat wrapper repeats")
_STACK_K = flags.DEFINE_integer("stack_k", 3, "Frame stack count")

# Replay / SAC
_BUFFER_SIZE = flags.DEFINE_integer("buffer_size", 100_000, "Replay capacity")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size")
_NSTEP = flags.DEFINE_integer("nstep", 3, "N-step")
_GAMMA = flags.DEFINE_float("gamma", 0.99, "Discount")

# Exploration / updates
_NUM_EXPL_STEPS = flags.DEFINE_integer("num_expl_steps", 2_000, "Random exploration steps")
_STD_START = flags.DEFINE_float("std_start", 1.0, "Std schedule start")
_STD_END = flags.DEFINE_float("std_end", 0.1, "Std schedule end")
_STD_DURATION = flags.DEFINE_integer("std_duration", 100_000, "Std schedule duration")
_STDDEV_CLIP = flags.DEFINE_float("stddev_clip", 0.3, "Pre-tanh clip")

_UPDATE_EVERY = flags.DEFINE_integer("update_every_steps", 2, "Update every N steps")
_UPDATE_MVMAE_EVERY = flags.DEFINE_integer("update_mvmae_every_steps", 4, "MV-MAE update cadence")
_COEF_MVMAE = flags.DEFINE_float("coef_mvmae", 1.0, "MV-MAE coefficient")
_CRITIC_TAU = flags.DEFINE_float("critic_target_tau", 0.01, "Critic target tau")

# The key “reference-like” knob: scan chunk length
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 32, "Steps per jitted scan chunk")

# Logging
_LOG_EVERY_STEPS = flags.DEFINE_integer("log_every_steps", 20_480, "Print speed every N steps")
_SYNC_TIMINGS = flags.DEFINE_boolean("sync_timings", False, "Block for accurate timings")

# Reference-style env var setup
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


def _maybe_block(x):
    if hasattr(x, "block_until_ready"):
        return x.block_until_ready()
    return x


def _sync_tree(x):
    jax.tree_util.tree_map(_maybe_block, x)


def _extract_obs(state) -> jp.ndarray:
    # Your env returns obs as (1, H, 2W, 3) before stacking; wrappers may change it.
    # This helper accepts either tensor obs or dict obs.
    if isinstance(state.obs, dict):
        if "pixels/view_0" in state.obs:
            return state.obs["pixels/view_0"]
        return next(iter(state.obs.values()))
    return state.obs


def _extract_reward_done(state) -> Tuple[jp.ndarray, jp.ndarray]:
    rew = state.reward
    done = state.done
    # Make sure types are consistent
    rew = jp.asarray(rew)
    done = jp.asarray(done).astype(jp.bool_)
    return rew, done


def main(argv):
    del argv

    # -------------------------
    # Build YOUR env (StereoPickCube) like reference builds env+config
    # -------------------------
    cfg = default_config()  # from your mujoco_pick_env.py :contentReference[oaicite:2]{index=2}
    cfg.vision = True
    cfg.vision_config.render_batch_size = 1
    cfg.vision_config.render_height = _RENDER_H.value
    cfg.vision_config.render_width = _RENDER_W.value
    cfg.action_repeat = 1  # keep env’s internal action_repeat at 1; wrappers handle repeats

    env = StereoPickCube(config=cfg)

    # Wrap like your pipeline: repeat then stack
    env = ActionRepeatWrapper(env, num_repeats=_ACTION_REPEAT.value)
    env = FrameStackWrapper(env, num_frames=_STACK_K.value)

    # -------------------------
    # Experiment name + logdir (match reference)
    # -------------------------
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"StereoPickCube-SAC-{timestamp}"
    if _SUFFIX.value:
        exp_name += f"-{_SUFFIX.value}"

    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {exp_name}")
    print(f"Logs are being stored in: {logdir}")

    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(cfg.to_dict(), fp, indent=2)

    # -------------------------
    # Shapes
    # -------------------------
    act_dim = env.action_size
    # StereoPickCube returns (1, H, 2W, 3) before stacking; wrappers may change channels.
    H = int(cfg.vision_config.render_height)
    W2 = int(cfg.vision_config.render_width) * 2
    C = 3 * int(_STACK_K.value)  # frame stack turns RGB into RGB*K
    obs_shape = (H, W2, C)

    # -------------------------
    # Replay buffer
    # -------------------------
    
    rb = rb_init(
    capacity=int(_BUFFER_SIZE.value),
    obs_shape=(obs_shape[0], obs_shape[1], obs_shape[2]),
    action_shape=(int(act_dim),),
        # optional: keep defaults unless you want to override
        # reward_shape=(1,),
        # discount_shape=(1,),
        # obs_dtype=jnp.uint8,
    )


    # -------------------------
    # Agent
    # -------------------------
    agent = DrQV2Agent.create(
        action_shape=(act_dim,),
        # if your agent uses these, keep them aligned with env
        nviews=2,
        in_channels=C,
        img_h_size=H,
        img_w_size=int(cfg.vision_config.render_width),
    )
    agent_state = agent.init_state(_SEED.value)

    # Jitted act/update (same idea as your existing pipeline)
    act_jit = jax.jit(
        lambda st, obs, step, eval_mode: DrQV2Agent.act(
            agent=agent,
            state=st,
            obs=obs,
            step=step,
            eval_mode=eval_mode,
            num_expl_steps=int(_NUM_EXPL_STEPS.value),
            std_start=float(_STD_START.value),
            std_end=float(_STD_END.value),
            std_duration=int(_STD_DURATION.value),
            clip_pre_tanh=float(_STDDEV_CLIP.value),
            deterministic_encoder=False,
        )
    )

    update_jit = jax.jit(
        lambda st, batch, step, stddev: DrQV2Agent.update_step(
            agent=agent,
            state=st,
            batch=batch,
            step=step,
            update_every_steps=int(_UPDATE_EVERY.value),
            update_mvmae_every_steps=int(_UPDATE_MVMAE_EVERY.value),
            coef_mvmae=float(_COEF_MVMAE.value),
            critic_target_tau=float(_CRITIC_TAU.value),
            stddev=stddev,
            stddev_clip=float(_STDDEV_CLIP.value),
        )
    )

    # -------------------------
    # Reset
    # -------------------------
    rng = jax.random.PRNGKey(int(_SEED.value))
    rng, rk = jax.random.split(rng)
    env_state = env.reset(rk)

    step_j = jp.asarray(0, jp.int32)

    # -------------------------
    # The reference-style core: env stepping inside JIT’d scan
    # -------------------------
    def one_step(carry, _):
        env_state, agent_state, rb, rng, step = carry

        obs = _extract_obs(env_state)

        action, agent_state2, stddev = act_jit(agent_state, obs, step, jp.asarray(False))

        next_env_state = env.step(env_state, action)

        rew, done = _extract_reward_done(next_env_state)
        disc = (jp.asarray(1.0, jp.float32) - done.astype(jp.float32)) * jp.asarray(_GAMMA.value, jp.float32)

        # Add to replay (B=1)
        rb2 = rb_add(
            rb,
            obs_b=obs,
            action_b=action[None, ...],
            reward_b=rew[None, ...],
            discount_b=disc[None, ...],
            done_b=done[None, ...],
        )

        # Sample + update
        rng2, sk = jax.random.split(rng)
        batch = rb_sample_jit(
            rb2,
            sk,
            batch_size=int(_BATCH_SIZE.value),
            nstep=int(_NSTEP.value),
            gamma=float(_GAMMA.value),
        )

        agent_state3, metrics = update_jit(agent_state2, batch, step, stddev)

        step2 = step + jp.asarray(1, jp.int32)
        carry2 = (next_env_state, agent_state3, rb2, rng2, step2)
        out = (rew, done, stddev, metrics)
        return carry2, out

    rollout_scan_jit = jax.jit(lambda carry: jax.lax.scan(one_step, carry, None, length=int(_UNROLL_LENGTH.value)))

    # -------------------------
    # Speed logging (reference-like)
    # -------------------------
    t0 = time.monotonic()
    speed_start_t = t0
    speed_last_t = t0
    speed_last_steps = 0

    carry = (env_state, agent_state, rb, rng, step_j)
    total_steps = int(_NUM_TIMESTEPS.value)

    while int(_maybe_block(carry[4])) < total_steps:
        carry, traj = rollout_scan_jit(carry)

        if _SYNC_TIMINGS.value:
            _sync_tree((carry, traj))

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
