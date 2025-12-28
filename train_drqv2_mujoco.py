# train_sac_scan_with_your_env.py
from __future__ import annotations

import datetime
import json
import os
import time
import warnings
from typing import Tuple

from absl import app, flags, logging
import jax
import jax.numpy as jp
from etils import epath

# host-side video saving
import numpy as np
import cv2
import imageio

# === YOUR ENV ===
from Mujoco_Sim.mujoco_pick_env import StereoPickCube, default_config

# === YOUR PIPELINE ===
from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.replay_buffer import rb_init, rb_add, rb_sample_jit
from DrQv2_Architecture.env_wrappers import FrameStackWrapper


# -------------------------
# Flags (trimmed to only what this file uses)
# -------------------------
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for experiment name")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Total env steps")

_RENDER_H = flags.DEFINE_integer("render_height", 64, "Render height")
_RENDER_W = flags.DEFINE_integer("render_width", 64, "Render width")
_STACK_K = flags.DEFINE_integer("stack_k", 3, "Frame stack count")

_BUFFER_SIZE = flags.DEFINE_integer("buffer_size", 100_000, "Replay capacity")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size")
_NSTEP = flags.DEFINE_integer("nstep", 3, "N-step")
_GAMMA = flags.DEFINE_float("gamma", 0.99, "Discount")

_NUM_EXPL_STEPS = flags.DEFINE_integer("num_expl_steps", 2_000, "Random exploration steps")
_STD_START = flags.DEFINE_float("std_start", 1.0, "Std schedule start")
_STD_END = flags.DEFINE_float("std_end", 0.1, "Std schedule end")
_STD_DURATION = flags.DEFINE_integer("std_duration", 100_000, "Std schedule duration")
_STDDEV_CLIP = flags.DEFINE_float("stddev_clip", 0.3, "Pre-tanh clip")

_UPDATE_EVERY = flags.DEFINE_integer("update_every_steps", 2, "Update every N steps")
_UPDATE_MVMAE_EVERY = flags.DEFINE_integer("update_mvmae_every_steps", 10, "MV-MAE update cadence")
_COEF_MVMAE = flags.DEFINE_float("coef_mvmae", 1.0, "MV-MAE coefficient")
_CRITIC_TAU = flags.DEFINE_float("critic_target_tau", 0.01, "Critic target tau")

_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 512, "Steps per jitted scan chunk")

# eval runs AFTER EVERY training chunk
_EVAL_STEPS = flags.DEFINE_integer("eval_steps", 1000, "Eval rollout length")
_EVAL_FPS = flags.DEFINE_integer("eval_fps", 20, "GIF fps")
_EVAL_RENDER_SIZE = flags.DEFINE_integer("eval_render_size", 256, "GIF height (width is 2x)")

# if True: reset-on-done happens *inside* the jitted scan (requires env.reset to be JIT-safe)
# if False: we do reset-on-done on host between chunks (always safe, but not “reset inside scan”)
_RESET_IN_JIT = flags.DEFINE_boolean("reset_in_jit", True, "Reset env inside JIT scan when done")

# env var setup
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# From your Prepare.fuse_normalize pipeline (video.py)
_MEAN = jp.array([0.51905, 0.47986, 0.48809], dtype=jp.float32).reshape(1, 1, 3)
_STD  = jp.array([0.17454, 0.20183, 0.19598], dtype=jp.float32).reshape(1, 1, 3)


def _extract_obs(state) -> jp.ndarray:
    if isinstance(state.obs, dict):
        if "pixels/view_0" in state.obs:
            return state.obs["pixels/view_0"]
        return next(iter(state.obs.values()))
    return state.obs


def _extract_reward_done(state) -> Tuple[jp.ndarray, jp.ndarray]:
    rew = jp.asarray(state.reward)
    done = jp.asarray(state.done).astype(jp.bool_)
    return rew, done


def _obs_to_u8_frame(obs: jp.ndarray) -> jp.ndarray:
    """obs: (1,H,W2,C) or (H,W2,C) normalized -> (H,W2,3) uint8."""
    obs = jp.asarray(obs)
    if obs.ndim == 4:
        obs = obs[0]
    rgb_norm = obs[..., -3:]  # newest frame if stacked
    rgb = jp.clip(rgb_norm * _STD + _MEAN, 0.0, 1.0)
    return (rgb * 255.0).astype(jp.uint8)


def _save_eval_gif(save_dir: epath.Path, frames_u8: np.ndarray, file_name: str, fps: int, render_size: int):
    """frames_u8: (T,H,W2,3) uint8."""
    resized = [cv2.resize(f, (render_size * 2, render_size)) for f in frames_u8]
    out_path = save_dir / (file_name + ".gif")
    imageio.mimsave(str(out_path), resized, fps=fps)


def main(argv):
    del argv

    # ---- env ----
    cfg = default_config()
    cfg.vision = True
    cfg.vision_config.render_batch_size = 1
    cfg.vision_config.render_height = _RENDER_H.value
    cfg.vision_config.render_width = _RENDER_W.value
    cfg.action_repeat = 1

    env = StereoPickCube(config=cfg)
    env = FrameStackWrapper(env, num_frames=_STACK_K.value)

    # ---- logdir ----
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"StereoPickCube-SAC-{timestamp}"
    if _SUFFIX.value:
        exp_name += f"-{_SUFFIX.value}"

    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    eval_video_dir = logdir / "eval_video"
    eval_video_dir.mkdir(parents=True, exist_ok=True)

    with open(logdir / "checkpoints" / "config.json", "w", encoding="utf-8") as fp:
        json.dump(cfg.to_dict(), fp, indent=2)

    print(f"Experiment name: {exp_name}")
    print(f"Logs are being stored in: {logdir}")

    # ---- shapes ----
    act_dim = env.action_size
    H = int(cfg.vision_config.render_height)
    W2 = int(cfg.vision_config.render_width) * 2
    C = 3 * int(_STACK_K.value)
    obs_shape = (H, W2, C)

    # ---- replay ----
    rb = rb_init(
        capacity=int(_BUFFER_SIZE.value),
        obs_shape=obs_shape,
        action_shape=(int(act_dim),),
    )

    # ---- agent ----
    agent = DrQV2Agent.create(
        action_shape=(act_dim,),
        nviews=2,
        in_channels=C,
        img_h_size=H,
        img_w_size=int(cfg.vision_config.render_width),
    )
    agent_state = agent.init_state(_SEED.value)

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
    
    # ---- reset (prime render token cache BEFORE any JIT traces reset) ----
    rng = jax.random.PRNGKey(int(_SEED.value))
    rng, rk = jax.random.split(rng)

    # First reset on host: this will run renderer.init(...) exactly once and cache the token in env.
    env_state = env.reset(rk)
    # Now reset is traceable (it will take the cached-token path), so we can JIT it.
    reset_jit = jax.jit(env.reset)
    step_j = jp.asarray(0, jp.int32)
    
    # ---- training scan ----
    def one_step(carry, _):
        env_state, agent_state, rb, rng, step = carry

        obs = _extract_obs(env_state)
        action, agent_state2, stddev = act_jit(agent_state, obs, step, jp.asarray(False))

        st1 = env.step(env_state, action)
        rew, done = _extract_reward_done(st1)

        disc = (jp.asarray(1.0, jp.float32) - done.astype(jp.float32)) * jp.asarray(_GAMMA.value, jp.float32)

        rb2 = rb_add(
            rb,
            obs_b=obs,
            action_b=action[None, ...],
            reward_b=rew[None, ...],
            discount_b=disc[None, ...],
            done_b=done[None, ...],
        )

        rng2, sk = jax.random.split(rng)
        batch = rb_sample_jit(
            rb2,
            sk,
            batch_size=int(_BATCH_SIZE.value),
            nstep=int(_NSTEP.value),
            gamma=float(_GAMMA.value),
        )

        agent_state3, metrics = update_jit(agent_state2, batch, step, stddev)

        # Proper reset-on-done/truncate:
        # - If reset_in_jit=True: do it inside scan (requires env.reset to be JIT-safe).
        # - If reset_in_jit=False: leave st1 as-is; we’ll do host reset between chunks.
        def _do_reset(rng_reset):
            return env.reset(rng_reset)

        # Proper reset-on-done inside JIT (requires reset_jit to be traceable)
        if _RESET_IN_JIT.value:
            rng3, rk_reset = jax.random.split(rng2)

            # Make sure pred is a scalar bool
            pred = jp.asarray(done).reshape(())

            st2 = jax.lax.cond(
                pred,
                lambda _unused: reset_jit(rk_reset),
                lambda _unused: st1,
                operand=None,
            )
        else:
            rng3 = rng2
            st2 = st1

        step2 = step + jp.asarray(1, jp.int32)
        return (st2, agent_state3, rb2, rng3, step2), (rew, done, stddev, metrics)

    rollout_scan_jit = jax.jit(
        lambda carry: jax.lax.scan(one_step, carry, None, length=int(_UNROLL_LENGTH.value))
    )

    # ---- eval scan (jitted), records frames ----
    def eval_one_step(carry, _):
        st, agent_state_e, step_e = carry
        obs = _extract_obs(st)
        action, agent_state_e2, _ = act_jit(agent_state_e, obs, step_e, jp.asarray(True))
        st1 = env.step(st, action)
        rew, done = _extract_reward_done(st1)
        frame_u8 = _obs_to_u8_frame(_extract_obs(st1))
        return (st1, agent_state_e2, step_e + 1), (rew, done, frame_u8)

    eval_scan_jit = jax.jit(
        lambda st0, agent_state_e: jax.lax.scan(
            eval_one_step,
            (st0, agent_state_e, jp.asarray(0, jp.int32)),
            None,
            length=int(_EVAL_STEPS.value),
        )
    )

    # ---- loop ----
    carry = (env_state, agent_state, rb, rng, step_j)
    total_steps = int(_NUM_TIMESTEPS.value)

    # warm-up compile
    carry, _ = rollout_scan_jit(carry)
    carry[4].block_until_ready()

    while True:
        cur_step = int(carry[4])
        if cur_step >= total_steps:
            break

        t0 = time.monotonic()
        carry, traj = rollout_scan_jit(carry)
        # single sync per chunk
        carry[4].block_until_ready()
        dt = time.monotonic() - t0

        rew, done, stddev, metrics = traj
        new_step = int(carry[4])
        dsteps = new_step - cur_step
        fps = dsteps / dt if dt > 0 else float("inf")
        ms_per_step = (dt / dsteps) * 1e3 if dsteps > 0 else float("inf")
        mean_rew = float(jp.mean(rew))

        print(
            f"[block] steps={new_step}  block_steps={dsteps}  dt={dt:.3f}s  "
            f"fps={fps:.1f}  ms/step={ms_per_step:.3f}  mean_rew={mean_rew:.4f}",
            flush=True,
        )

        # ---- host reset on done (ONLY if reset_in_jit=False) ----
        # This ensures you truly reset physics even if env.reset isn't JIT-safe.
        if not _RESET_IN_JIT.value:
            # done from traj is shape (unroll,) bool; if ANY done happened, reset once.
            any_done = bool(jp.any(done).block_until_ready())
            if any_done:
                rng_host = carry[3]
                rng_host, rk_reset = jax.random.split(rng_host)
                st_reset = env.reset(rk_reset)
                carry = (st_reset, carry[1], carry[2], rng_host, carry[4])

        # ---- eval after every chunk ----
        rng_host = carry[3]
        rng_host, rk_eval = jax.random.split(rng_host)
        eval_state0 = env.reset(rk_eval)

        (eval_state_out, agent_state_out, _), (eval_rew, eval_done, eval_frames) = eval_scan_jit(
            eval_state0, carry[1]
        )

        # one sync for eval outputs
        eval_rew = eval_rew.block_until_ready()
        eval_done = eval_done.block_until_ready()
        eval_frames = eval_frames.block_until_ready()

        eval_return = float(jp.sum(eval_rew))
        eval_mean = float(jp.mean(eval_rew))
        eval_done_rate = float(jp.mean(eval_done.astype(jp.float32)))

        print(
            f"[eval] at_step={new_step}  eval_steps={int(_EVAL_STEPS.value)}  "
            f"return={eval_return:.3f}  mean_rew={eval_mean:.4f}  done_rate={eval_done_rate:.4f}",
            flush=True,
        )

        frames_np = np.array(eval_frames)  # (T,H,W2,3) uint8
        _save_eval_gif(
            eval_video_dir,
            frames_np,
            file_name=f"step_{new_step:08d}",
            fps=int(_EVAL_FPS.value),
            render_size=int(_EVAL_RENDER_SIZE.value),
        )

        # put updated rng back
        carry = (carry[0], carry[1], carry[2], rng_host, carry[4])

    print("Done training.")


if __name__ == "__main__":
    app.run(main)
