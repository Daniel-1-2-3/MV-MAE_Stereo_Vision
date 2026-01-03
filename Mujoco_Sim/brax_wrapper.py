# =========================
# Mujoco_Sim/brax_wrapper.py
# =========================
import os
import re
import sys
import time
import traceback
from typing import Optional, Tuple

import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack


# -----------------------------
# Wrapper-side probes
# -----------------------------

_WRP_CUDART = None


def _wrap_probe_enabled() -> bool:
    return os.environ.get("WRAPPER_PROBE", "0") == "1"


def _wrap_probe_dump_maps() -> bool:
    return os.environ.get("PROBE_DUMP_MAPS", "0") == "1"


def _wrap_probe_cuda_clear_mode() -> str:
    m = os.environ.get("PROBE_CUDA_CLEAR_MODE", "peek").strip().lower()
    return "get" if m == "get" else "peek"


def _wrap_probe_max_steps() -> int:
    try:
        return int(os.environ.get("WRAPPER_PROBE_MAX_STEPS", "1"))
    except Exception:
        return 1


def _wrap_probe_exit_after_reset() -> bool:
    return os.environ.get("WRAPPER_PROBE_EXIT_AFTER_RESET", "0") == "1"


def _wp(msg: str) -> None:
    print(msg, flush=True)


def _wp_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _wrap_dump_cuda_maps(tag: str) -> None:
    if not (_wrap_probe_enabled() and _wrap_probe_dump_maps()):
        return
    pats = [
        r".*libcuda\.so(\.[0-9]+)*$",
        r".*libcudart\.so(\.[0-9]+)*$",
        r".*libnvrtc\.so(\.[0-9]+)*$",
        r".*libnvJitLink\.so(\.[0-9]+)*$",
        r".*libnvidia-ptxjitcompiler\.so(\.[0-9]+)*$",
        r".*libcublas\.so(\.[0-9]+)*$",
        r".*libcudnn\.so(\.[0-9]+)*$",
    ]
    seen = set()
    uniq = []
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                path = parts[-1]
                if any(re.match(p, path) for p in pats):
                    if path not in seen:
                        seen.add(path)
                        uniq.append(path)
    except Exception as e:
        _wp(f"[WRAPPER_PROBE] [{tag}] maps read failed: {e!r}")
        return

    _wp(f"[WRAPPER_PROBE] [{tag}] CUDA libs in /proc/self/maps ({len(uniq)}):")
    for h in uniq:
        _wp(f"  {h}")


def _wrap_load_cudart():
    global _WRP_CUDART
    if _WRP_CUDART is not None:
        return _WRP_CUDART
    try:
        import ctypes

        for name in ("libcudart.so.12", "libcudart.so"):
            try:
                _WRP_CUDART = ctypes.CDLL(name)
                _WRP_CUDART.cudaPeekAtLastError.restype = ctypes.c_int
                _WRP_CUDART.cudaGetLastError.restype = ctypes.c_int
                _WRP_CUDART.cudaGetErrorString.restype = ctypes.c_char_p
                return _WRP_CUDART
            except Exception:
                pass
    except Exception:
        pass
    _WRP_CUDART = None
    return None


def _wrap_cuda_err(tag: str) -> tuple[int, str]:
    lib = _wrap_load_cudart()
    mode = _wrap_probe_cuda_clear_mode()
    if lib is None:
        if _wrap_probe_enabled():
            _wp(f"[WRAPPER_PROBE] [{tag}] libcudart not loadable (ctypes).")
        return (-999, "no libcudart")
    try:
        if mode == "get":
            err = int(lib.cudaGetLastError())
            mode_s = "Get"
        else:
            err = int(lib.cudaPeekAtLastError())
            mode_s = "Peek"
        msg = lib.cudaGetErrorString(err)
        msg_s = msg.decode("utf-8", "ignore") if msg else ""
        if _wrap_probe_enabled():
            _wp(f"[WRAPPER_PROBE] [{tag}] cuda{mode_s}AtLastError -> ({err}, {msg_s!r})")
        return (err, msg_s)
    except Exception as e:
        if _wrap_probe_enabled():
            _wp(f"[WRAPPER_PROBE] [{tag}] cuda last-error query failed: {e!r}")
        return (-998, "query failed")


def _wrap_jax_small_op(tag: str) -> None:
    if not _wrap_probe_enabled():
        return
    try:
        y = jax.jit(lambda: (jp.arange(4096, dtype=jp.float32) * 3).sum())()
        jax.block_until_ready(y)
        _wp(f"[WRAPPER_PROBE] [{tag}] PASS: small JAX op OK")
    except Exception as e:
        _wp(f"[WRAPPER_PROBE] [{tag}] FAIL: small JAX op -> {e!r}")
        traceback.print_exc()
        raise


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    # GPU->GPU via the modern DLPack protocol (stream-aware).
    # Let JAX request the view on its own stream to avoid races.
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    # Force capsule path (avoids torch calling JAX's __dlpack__ directly)
    return tpack.from_dlpack(jax.dlpack.to_dlpack(x))


def _rebuffer_xla(rgb: jax.Array) -> jax.Array:
    # Forces a new XLA-owned buffer on GPU.
    if _wrap_probe_enabled():
        _wp(f"[WRAPPER_PROBE] _rebuffer_xla: entering; rgb shape={getattr(rgb,'shape',None)} dtype={getattr(rgb,'dtype',None)}")
        _wrap_cuda_err("_rebuffer_xla(pre)")
    z = jp.array(0, dtype=rgb.dtype)
    rgb2 = rgb + z  # <-- your failure site in stack traces
    if _wrap_probe_enabled():
        _wp("[WRAPPER_PROBE] _rebuffer_xla: created rgb2 (dispatch may happen now)")
        _wrap_cuda_err("_rebuffer_xla(post-create)")
    return rgb2


class RSLRLBraxWrapper:
    """Batched MJX env wrapper for a Torch RL loop (DrQv2)."""

    def __init__(
        self,
        env,
        batch_size: int,
        seed: int,
        episode_length: int,
        action_repeat: int = 1,
        render_callback=None,
        randomization_fn=None,
        device_rank: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self._raw_env = env
        self.batch_size = int(batch_size)
        self.num_envs = int(batch_size)
        self.episode_length = int(episode_length)
        self.action_repeat = int(action_repeat)

        self._probe_step_idx = 0

        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # Safety: Madrona renderer must be created with num_worlds == batch_size
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"StereoPickCube(render_batch_size=...) must equal num_envs. Got render_batch_size={rbs}, num_envs={self.batch_size}."
                )

        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        self._reset_physics_batched = jax.vmap(self._raw_env.reset_physics)
        self._step_physics_batched = jax.vmap(self._raw_env.step_physics, in_axes=(0, 0), out_axes=0)

        self.reset_fn = jax.jit(self._reset_physics_batched)

        if self.action_repeat == 1:
            self.step_fn = jax.jit(self._step_physics_batched)
        else:

            def _step_repeat(state, action):
                def body(i, st):
                    return self._step_physics_batched(st, action)

                return jax.lax.fori_loop(0, self.action_repeat, body, state)

            self.step_fn = jax.jit(_step_repeat)

        self.env_state = None  # set in reset()

        if _wrap_probe_enabled():
            _wp(f"[WRAPPER_PROBE] init time={_wp_ts()} pid={os.getpid()}")
            _wp(f"[WRAPPER_PROBE] torch={torch.__version__} torch.cuda={torch.version.cuda} device={self.device}")
            _wp(f"[WRAPPER_PROBE] jax={jax.__version__} devices={jax.devices()}")
            _wp(f"[WRAPPER_PROBE] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')}")
            _wp(f"[WRAPPER_PROBE] XLA_FLAGS={os.environ.get('XLA_FLAGS','')}")
            _wrap_dump_cuda_maps("wrapper.__init__")
            _wrap_cuda_err("wrapper.__init__")
            _wrap_jax_small_op("wrapper.__init__ small JAX op")

    def _probe_gate(self) -> bool:
        if not _wrap_probe_enabled():
            return False
        return self._probe_step_idx < _wrap_probe_max_steps()

    def reset(self) -> TensorDict:
        self._probe_step_idx = 0

        if self._probe_gate():
            _wp(f"[WRAPPER_PROBE] ===== reset() begin time={_wp_ts()} =====")
            _wrap_cuda_err("reset(begin)")
            _wrap_dump_cuda_maps("reset(begin)")
            _wrap_jax_small_op("reset(begin) small JAX op")

        # Jitted, batched physics reset
        try:
            self.env_state = self.reset_fn(self.key_reset)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: reset_fn (jit reset) -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err("reset(reset_fn EXC)")
            _wrap_dump_cuda_maps("reset(reset_fn EXC)")
            raise

        if self._probe_gate():
            _wp("[WRAPPER_PROBE] reset_fn OK")
            _wrap_cuda_err("reset(after reset_fn)")
            _wrap_jax_small_op("reset(after reset_fn) small JAX op")

        # Non-jit renderer init + render
        try:
            debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
            self._raw_env._ensure_render_token(self.env_state.data, debug)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: _ensure_render_token -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err("reset(_ensure_render_token EXC)")
            _wrap_dump_cuda_maps("reset(_ensure_render_token EXC)")
            raise

        if self._probe_gate():
            _wp("[WRAPPER_PROBE] _ensure_render_token OK")
            _wrap_cuda_err("reset(after _ensure_render_token)")

        # Render raw RGBA (uint8)
        try:
            rgb = self._raw_env.render_rgba(self.env_state.data)
            jax.block_until_ready(rgb)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: render_rgba / block_until_ready(rgb) -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err("reset(render_rgba EXC)")
            _wrap_dump_cuda_maps("reset(render_rgba EXC)")
            raise

        if self._probe_gate():
            _wp(f"[WRAPPER_PROBE] render_rgba OK rgb shape={getattr(rgb,'shape',None)} dtype={getattr(rgb,'dtype',None)}")
            _wrap_cuda_err("reset(after render_rgba)")
            _wrap_jax_small_op("reset(after render_rgba) small JAX op")

        # Rebuffer: THIS is the exact stage you crash on in training
        try:
            rgb2 = _rebuffer_xla(rgb)
            jax.block_until_ready(rgb2)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: rebuffer_xla(rgb) / block_until_ready(rgb2) -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err("reset(rebuffer EXC)")
            _wrap_dump_cuda_maps("reset(rebuffer EXC)")
            raise

        if self._probe_gate():
            _wp("[WRAPPER_PROBE] rebuffer_xla(rgb) OK")
            _wrap_cuda_err("reset(after rebuffer)")
            _wrap_jax_small_op("reset(after rebuffer) small JAX op")

        # Export via DLPack
        try:
            rgb_t = _jax_to_torch(rgb2)
            if rgb_t.device != self.device:
                rgb_t = rgb_t.to(self.device, non_blocking=True)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: jax->torch dlpack -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err("reset(dlpack EXC)")
            _wrap_dump_cuda_maps("reset(dlpack EXC)")
            raise

        if self._probe_gate():
            _wp(f"[WRAPPER_PROBE] dlpack OK rgb_t device={rgb_t.device} dtype={rgb_t.dtype} shape={tuple(rgb_t.shape)}")
            # synchronize Torch to surface stream issues
            torch.cuda.synchronize() if rgb_t.is_cuda else None
            _wrap_cuda_err("reset(after dlpack + torch sync)")

        # Split stereo views
        left = rgb_t[:, 0, :, :, :3]
        right = rgb_t[:, 1, :, :, :3]
        B, H, W, C = left.shape
        obs_t = torch.empty((B, H, 2 * W, C), device=self.device, dtype=left.dtype)
        obs_t[:, :, :W, :].copy_(left)
        obs_t[:, :, W:, :].copy_(right)

        if self._probe_gate():
            _wp("[WRAPPER_PROBE] obs packing OK")
            torch.cuda.synchronize() if obs_t.is_cuda else None
            _wrap_cuda_err("reset(after obs pack + torch sync)")
            _wp(f"[WRAPPER_PROBE] ===== reset() end time={_wp_ts()} =====")

        if _wrap_probe_exit_after_reset():
            _wp("[WRAPPER_PROBE] WRAPPER_PROBE_EXIT_AFTER_RESET=1 -> exiting after reset() for debugging.")
            sys.exit(0)

        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."
        self._probe_step_idx += 1

        if self._probe_gate():
            _wp(f"[WRAPPER_PROBE] ----- step({self._probe_step_idx}) begin time={_wp_ts()} -----")
            _wrap_cuda_err(f"step{self._probe_step_idx}(begin)")

        action = torch.clamp(action, -1.0, 1.0)
        try:
            action_jax = _torch_to_jax(action)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: torch->jax dlpack action -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err(f"step{self._probe_step_idx}(action dlpack EXC)")
            raise

        try:
            self.env_state = self.step_fn(self.env_state, action_jax)
        except Exception as e:
            _wp(f"[WRAPPER_PROBE] FAIL: step_fn (jit step) -> {e!r}")
            traceback.print_exc()
            _wrap_cuda_err(f"step{self._probe_step_idx}(step_fn EXC)")
            raise

        if self._probe_gate():
            _wp(f"[WRAPPER_PROBE] step_fn OK")
            _wrap_cuda_err(f"step{self._probe_step_idx}(after step_fn)")
            _wrap_jax_small_op(f"step{self._probe_step_idx}(after step_fn) small JAX op")

        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
        self._raw_env._ensure_render_token(self.env_state.data, debug)

        rgb = self._raw_env.render_rgba(self.env_state.data)
        jax.block_until_ready(rgb)

        if self._probe_gate():
            _wp(f"[WRAPPER_PROBE] render_rgba OK rgb shape={getattr(rgb,'shape',None)} dtype={getattr(rgb,'dtype',None)}")
            _wrap_cuda_err(f"step{self._probe_step_idx}(after render_rgba)")

        rgb2 = _rebuffer_xla(rgb)
        jax.block_until_ready(rgb2)

        if self._probe_gate():
            _wp("[WRAPPER_PROBE] rebuffer_xla(rgb) OK")
            _wrap_cuda_err(f"step{self._probe_step_idx}(after rebuffer)")

        rgb_t = _jax_to_torch(rgb2)
        if rgb_t.device != self.device:
            rgb_t = rgb_t.to(self.device, non_blocking=True)

        left = rgb_t[:, 0, :, :, :3]
        right = rgb_t[:, 1, :, :, :3]
        B, H, W, C = left.shape
        obs_t = torch.empty((B, H, 2 * W, C), device=self.device, dtype=left.dtype)
        obs_t[:, :, :W, :].copy_(left)
        obs_t[:, :, W:, :].copy_(right)
        reward_t = _jax_to_torch(self.env_state.reward).to(self.device)
        done_t = _jax_to_torch(self.env_state.done).to(self.device)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self._probe_gate():
            torch.cuda.synchronize() if obs_t.is_cuda else None
            _wrap_cuda_err(f"step{self._probe_step_idx}(end + torch sync)")
            _wp(f"[WRAPPER_PROBE] ----- step({self._probe_step_idx}) end time={_wp_ts()} -----")

        return obs_td, reward_t, done_t, info
