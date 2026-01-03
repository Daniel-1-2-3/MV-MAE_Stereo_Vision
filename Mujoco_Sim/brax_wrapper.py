import os
from typing import Optional, Tuple

import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack

# -----------------------------------------------------------------------------
# CUDA context restore workaround
# -----------------------------------------------------------------------------
# Goal:
#   Madrona render custom call succeeds, but the next unrelated JAX kernel fails
#   (cuModuleGetFunction / cudaErrorInvalidValue). This strongly suggests the render
#   path is switching the current CUDA context on the calling thread and not restoring it.
#
# What we do:
#   1) Capture a baseline CUDA context ONCE (after forcing primary context current).
#   2) After every render, restore that baseline context on the Python thread.
#   3) Immediately run a tiny “canary” kernel (rgb + 0) to verify kernels still launch.
#
# Note:
#   If the poisoning occurs only on XLA internal threads, Python-level restore won’t fix it;
#   the real fix would be in the Madrona custom call (save/restore CUDA context in C++).
# -----------------------------------------------------------------------------
import ctypes
import ctypes.util


def _load_cdll(name: str, fallback_names: Tuple[str, ...]) -> ctypes.CDLL:
    path = ctypes.util.find_library(name)
    if path:
        return ctypes.CDLL(path)
    last_err = None
    for fb in fallback_names:
        try:
            return ctypes.CDLL(fb)
        except OSError as e:
            last_err = e
    raise OSError(f"Could not load {name}. Tried: {fallback_names}. Last error: {last_err}")


# CUDA Driver API
_libcuda = _load_cdll("cuda", ("libcuda.so.1", "libcuda.so"))
# CUDA Runtime API (for cudaFree(0) to force primary context current on this thread)
_libcudart = _load_cdll("cudart", ("libcudart.so.12", "libcudart.so"))

CUcontext = ctypes.c_void_p

_libcuda.cuInit.argtypes = [ctypes.c_uint]
_libcuda.cuInit.restype = ctypes.c_int

_libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(CUcontext)]
_libcuda.cuCtxGetCurrent.restype = ctypes.c_int

_libcuda.cuCtxSetCurrent.argtypes = [CUcontext]
_libcuda.cuCtxSetCurrent.restype = ctypes.c_int

_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
_libcudart.cudaFree.restype = ctypes.c_int


def _cu_check(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} failed (CUDA error code {rc})")


_cu_check(_libcuda.cuInit(0), "cuInit")

_BASE_CTX: Optional[int] = None


def _cuda_ctx_get_current() -> int:
    ctx = CUcontext()
    _cu_check(_libcuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    return int(ctx.value or 0)


def _cuda_ctx_set_current(ctx_value: int) -> None:
    _cu_check(_libcuda.cuCtxSetCurrent(CUcontext(ctx_value)), "cuCtxSetCurrent")


def _capture_base_ctx_once() -> int:
    """Capture a baseline context pointer once for this process.

    We call cudaFree(0) first to ensure the device primary context becomes current
    on *this* thread, then read cuCtxGetCurrent().
    """
    global _BASE_CTX
    if _BASE_CTX is not None:
        return _BASE_CTX

    # Force primary context current on this host thread.
    _cu_check(_libcudart.cudaFree(None), "cudaFree(0)")

    ctx = _cuda_ctx_get_current()
    if ctx == 0:
        raise RuntimeError(
            "Baseline CUDA context is 0 even after cudaFree(0). "
            "This usually means no CUDA device is available to this process."
        )
    _BASE_CTX = ctx

    if os.environ.get("CUDA_CTX_PROBE", "0") == "1":
        print(f"[ctx] captured base ctx = 0x{_BASE_CTX:x}")
    return _BASE_CTX


def _restore_base_ctx() -> None:
    base = _capture_base_ctx_once()
    _cuda_ctx_set_current(base)


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    # GPU->GPU via DLPack (stream-aware). Let JAX pull the view.
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    # Force capsule path (avoids torch calling JAX's __dlpack__ directly)
    return tpack.from_dlpack(jax.dlpack.to_dlpack(x))


class RSLRLBraxWrapper:
    """Batched MJX env wrapper for a Torch RL loop (DrQv2).

    Key rules for stability + speed:
    - Physics is jitted + vmapped (parallel worlds).
    - Madrona renderer init/render is **never** called under vmap/jit.
      Rendering happens from Python on batched `mjx.Data`.
    - All obs/action transfer uses DLPack to avoid CPU copies.
    - Workaround: restore baseline CUDA context after render.
    """

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

        # Torch device for the RL learner tensors
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
                    f"StereoPickCube(render_batch_size=...) must equal num_envs. "
                    f"Got render_batch_size={rbs}, num_envs={self.batch_size}."
                )

        # RNG keys per env
        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        # Batched physics: reset_physics / step_physics
        self._reset_physics_batched = jax.vmap(self._raw_env.reset_physics)
        self._step_physics_batched = jax.vmap(self._raw_env.step_physics, in_axes=(0, 0), out_axes=0)

        self.reset_fn = jax.jit(self._reset_physics_batched)

        if self.action_repeat == 1:
            self.step_fn = jax.jit(self._step_physics_batched)
        else:

            def _step_repeat(state, action):
                def body(_, st):
                    return self._step_physics_batched(st, action)

                return jax.lax.fori_loop(0, self.action_repeat, body, state)

            self.step_fn = jax.jit(_step_repeat)

        self.env_state = None  # set in reset()

    def _render_and_guard(self) -> jax.Array:
        """Render RGBA and guard against CUDA context poisoning.

        Returns:
          rgb2: jax.Array uint8 with a post-render canary op applied.
        """
        # IMPORTANT: do NOT allow any debug path that does an internal smoke render
        # inside _ensure_render_token, because that can poison CUDA before we restore.
        self._raw_env._ensure_render_token(self.env_state.data, debug=False)

        rgb = self._raw_env.render_rgba(self.env_state.data)

        # Force the custom call to actually execute now.
        jax.block_until_ready(rgb)

        if os.environ.get("CUDA_CTX_PROBE", "0") == "1":
            after = _cuda_ctx_get_current()
            print(f"[ctx] after render, current ctx = 0x{after:x}")

        # Restore baseline context for this host thread.
        _restore_base_ctx()

        # Canary: first post-render JAX kernel (this is where you were failing).
        rgb2 = rgb + jp.array(0, dtype=rgb.dtype)
        jax.block_until_ready(rgb2)

        return rgb2

    def reset(self) -> TensorDict:
        # Ensure baseline context is captured after JAX is alive.
        jax.block_until_ready(jp.array(0, dtype=jp.int32))
        _capture_base_ctx_once()

        # Jitted, batched physics reset
        self.env_state = self.reset_fn(self.key_reset)

        # Render + guard
        rgb2 = self._render_and_guard()

        # Export via capsule DLPack
        rgb_t = _jax_to_torch(rgb2)
        if rgb_t.device != self.device:
            rgb_t = rgb_t.to(self.device, non_blocking=True)

        # Split stereo views, concatenate in Torch
        left = rgb_t[:, 0, :, :, :3]   # [B,H,W,3]
        right = rgb_t[:, 1, :, :, :3]  # [B,H,W,3]
        B, H, W, C = left.shape
        obs_t = torch.empty((B, H, 2 * W, C), device=self.device, dtype=left.dtype)
        obs_t[:, :, :W, :].copy_(left)
        obs_t[:, :, W:, :].copy_(right)

        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        # Torch -> JAX (GPU->GPU), physics step (jitted)
        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        # Render + guard
        rgb2 = self._render_and_guard()

        # Export via capsule DLPack
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

        # Keep info minimal to avoid host sync every step
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}
        return obs_td, reward_t, done_t, info
