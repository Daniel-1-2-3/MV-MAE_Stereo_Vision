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

# Saving eval videos (same stack as video.py)
import numpy as np
import cv2
import imageio

# === YOUR ENV ===
from Mujoco_Sim.mujoco_pick_env import StereoPickCube, default_config

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
_UPDATE_MVMAE_EVERY = flags.DEFINE_integer("update_mvmae_every_steps", 10, "MV-MAE update cadence")
_COEF_MVMAE = flags.DEFINE_float("coef_mvmae", 1.0, "MV-MAE coefficient")
_CRITIC_TAU = flags.DEFINE_float("critic_target_tau", 0.01, "Critic target tau")

# The key “reference-like” knob: scan chunk length
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 512, "Steps per jitted scan chunk")

_DEBUG_TIMING = flags.DEFINE_boolean(
    "debug_timing",
    True,
    "Print detailed timing/compile diagnostics (may slow training).",
)
_DEBUG_COMPILE_LOWER = flags.DEFINE_boolean(
    "debug_compile_lower",
    True,
    "If debug_timing, also call .lower(...).compile() to measure compile time (no execute) for key jitted functions.",
)

# -------------------------
# Eval config (NEW)
# -------------------------
_EVAL_EVERY = 10_000     # run eval every N training env steps
_EVAL_STEPS = 1_000      # run eval for ~1000 steps
_EVAL_FPS = 20
_EVAL_RENDER_SIZE = 256

# Normalization stats (from Prepare.fuse_normalize), matching video.py
_MEAN = jp.array([0.51905, 0.47986, 0.48809], dtype=jp.float32).reshape(1, 1, 3)
_STD  = jp.array([0.17454, 0.20183, 0.19598], dtype=jp.float32).reshape(1, 1, 3)


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


def _timeit(msg: str, fn, *args, block: bool = True, **kwargs):
    """Run fn(*args, **kwargs), return (out, seconds). Optionally block_until_ready."""
    t0 = time.monotonic()
    out = fn(*args, **kwargs)
    if block:
        _sync_tree(out)
    dt = time.monotonic() - t0
    print(f"[timing] {msg}: {dt:.6f}s", flush=True)
    return out, dt


def _compile_time(jitted_fn, example_args, name: str):
    """Measure compile-only time via lower(...).compile()."""
    t0 = time.monotonic()
    _ = jitted_fn.lower(*example_args).compile()
    dt = time.monotonic() - t0
    print(f"[compile] {name} lower().compile(): {dt:.3f}s", flush=True)
    return dt


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


def _obs_to_u8_frame(obs_hw_c: jp.ndarray) -> jp.ndarray:
    """
    obs_hw_c: (H, 2W, C) float32, normalized like Prepare.fuse_normalize.
              If frame-stacked, we take the last 3 channels as the most recent RGB frame.

    returns: (H, 2W, 3) uint8 RGB in [0,255]
    """
    rgb_norm = obs_hw_c[..., -3:]  # (H, 2W, 3)
    rgb = rgb_norm * _STD + _MEAN
    rgb = jp.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(jp.uint8)


def _save_eval_gif(save_dir: epath.Path, frames_u8: np.ndarray, file_name: str, fps: int = _EVAL_FPS):
    """frames_u8: (T, H, 2W, 3) uint8."""
    resized = [cv2.resize(f, (_EVAL_RENDER_SIZE * 2, _EVAL_RENDER_SIZE)) for f in frames_u8]
    out_path = save_dir / (file_name + ".gif")
    imageio.mimsave(str(out_path), resized, fps=fps)

def main(argv):
    del argv

    # -------------------------
    # Build YOUR env (StereoPickCube) like reference builds env+config
    # -------------------------
    cfg = default_config()
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

    # Eval video dir (same name as video.py)
    eval_video_dir = logdir / "eval_video"
    eval_video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {exp_name}")
    print(f"Logs are being stored in: {logdir}")

    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
        json.dump(cfg.to_dict(), fp, indent=2)

    # -------------------------
    # Shapes
    # -------------------------
    act_dim = env.action_size
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
    )

    # -------------------------
    # Agent
    # -------------------------
    agent = DrQV2Agent.create(
        action_shape=(act_dim,),
        nviews=2,
        in_channels=C,
        img_h_size=H,
        img_w_size=int(cfg.vision_config.render_width),
    )
    agent_state = agent.init_state(_SEED.value)

    # Jitted act/update
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
    # JIT'd evaluation rollout (NEW)
    # -------------------------
    def eval_one_step(carry, _):
        eval_env_state, agent_state_e, step_e = carry

        obs = _extract_obs(eval_env_state)
        action, agent_state_e2, _stddev = act_jit(agent_state_e, obs, step_e, jp.asarray(True))

        next_eval_state = env.step(eval_env_state, action)
        rew, done = _extract_reward_done(next_eval_state)

        obs2 = _extract_obs(next_eval_state)
        frame_u8 = _obs_to_u8_frame(obs2)

        step_e2 = step_e + jp.asarray(1, jp.int32)
        return (next_eval_state, agent_state_e2, step_e2), (rew, done, frame_u8)

    eval_scan_jit = jax.jit(
        lambda eval_state, agent_state_e, step0: jax.lax.scan(
            eval_one_step,
            (eval_state, agent_state_e, step0),
            None,
            length=_EVAL_STEPS,
        )
    )

    # -------------------------
    # Compile diagnostics (compile-only)
    # -------------------------
    if _DEBUG_TIMING.value:
        print("[debug] compile diagnostics enabled", flush=True)
        obs0 = _extract_obs(env_state)

        if _DEBUG_COMPILE_LOWER.value:
            _compile_time(act_jit, (agent_state, obs0, step_j, jp.asarray(False)), "act_jit")
            carry0 = (env_state, agent_state, rb, rng, step_j)
            _compile_time(rollout_scan_jit, (carry0,), "rollout_scan_jit")
        print("[debug] compile diagnostics finished", flush=True)

    # -------------------------
    # Block timing (one log per scan block)
    # -------------------------
    carry = (env_state, agent_state, rb, rng, step_j)
    total_steps = int(_NUM_TIMESTEPS.value)
    unroll = int(_UNROLL_LENGTH.value)

    # Warm-up to keep first timing clean
    carry, _ = rollout_scan_jit(carry)
    carry[4].block_until_ready()

    while True:
        cur_step = int(carry[4])
        if cur_step >= total_steps:
            break

        t0 = time.monotonic()
        carry, traj = rollout_scan_jit(carry)
        rew, done, stddev, metrics = traj

        # block-level sync for meaningful timing / fps
        rew = rew.block_until_ready()
        carry[4].block_until_ready()
        dt = time.monotonic() - t0

        new_step = int(carry[4])
        dsteps = new_step - cur_step
        fps = (dsteps / dt) if dt > 0 else float("inf")
        ms_per_step = (dt / dsteps) * 1e3 if dsteps > 0 else float("inf")
        mean_rew = float(jp.mean(rew))

        print(
            f"[block] steps={new_step}  "
            f"block_steps={dsteps}  "
            f"dt={dt:.3f}s  "
            f"fps={fps:.1f}  "
            f"ms/step={ms_per_step:.3f}  "
            f"mean_rew={mean_rew:.4f}",
            flush=True,
        )

        # -------------------------
        # Periodic evaluation (every 10k steps) (NEW)
        # -------------------------
        if (new_step % _EVAL_EVERY) == 0:
            # Reset eval state on host (renderer.init in reset is not JIT-friendly)
            rng_host = carry[3]
            rng_host, rk_eval = jax.random.split(rng_host)
            eval_state = env.reset(rk_eval)

            # Run JIT'd eval rollout (~1000 steps)
            (eval_state_out, agent_state_out, eval_step_out), (eval_rew, eval_done, eval_frames) = eval_scan_jit(
                eval_state, carry[1], jp.asarray(0, jp.int32)
            )

            # One sync for all eval outputs
            eval_rew = eval_rew.block_until_ready()
            eval_done = eval_done.block_until_ready()
            eval_frames = eval_frames.block_until_ready()

            eval_return = float(jp.sum(eval_rew))
            eval_mean_rew = float(jp.mean(eval_rew))
            eval_done_rate = float(jp.mean(eval_done.astype(jp.float32)))

            print(
                f"[eval] at_step={new_step}  "
                f"eval_steps={_EVAL_STEPS}  "
                f"return={eval_return:.3f}  "
                f"mean_rew={eval_mean_rew:.4f}  "
                f"done_rate={eval_done_rate:.4f}",
                flush=True,
            )

            # Save GIF (format matches video.py: uint8 RGB resized to (render_size*2, render_size))
            frames_np = np.array(eval_frames)  # (T,H,2W,3) uint8
            _save_eval_gif(eval_video_dir, frames_np, file_name=f"step_{new_step:08d}", fps=_EVAL_FPS)

            # Put updated rng back into carry (so eval consumes RNG deterministically)
            carry = (carry[0], carry[1], carry[2], rng_host, carry[4])

    print("Done training.")


if __name__ == "__main__":
    app.run(main)
