# Mujoco_Sim/brax_wrapper.py
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # pragma: no cover
    from tensordict import TensorDict  # pragma: no cover


# ---------------------------
# Torch/JAX helpers
# ---------------------------

def _torch_to_jax(x) -> jax.Array:
    # Local import so the renderer worker process never imports torch.
    from torch.utils import dlpack as tpack
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch_hostcopy(x: jax.Array, device):
    """Break any GPU pointer sharing: JAX -> host numpy -> Torch tensor -> (optional) GPU."""
    import torch
    arr = np.asarray(jax.device_get(x))
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        t = t.to(device, non_blocking=True)
    return t


# ---------------------------
# Worker construction spec
# ---------------------------

@dataclass(frozen=True)
class _EnvCtorSpec:
    module: str
    cls_name: str
    config: Any
    config_overrides: Optional[Dict[str, Any]]
    render_batch_size: int
    render_width: int
    render_height: int


@dataclass(frozen=True)
class _ShmSpec:
    # Render-only inputs (already-forwarded kinematics from main process)
    geom_xpos_name: str
    geom_xmat_name: str
    cam_xpos_name: str
    cam_xmat_name: str
    light_xpos_name: str
    light_xdir_name: str

    # Raw RGBA output from Madrona (uint8)
    out_rgba_name: str

    # Shapes
    B: int
    geom_xpos_shape: Tuple[int, ...]
    geom_xmat_shape: Tuple[int, ...]
    cam_xpos_shape: Tuple[int, ...]
    cam_xmat_shape: Tuple[int, ...]
    light_xpos_shape: Tuple[int, ...]
    light_xdir_shape: Tuple[int, ...]
    H: int
    W: int


def _import_env_from_spec(spec: _EnvCtorSpec):
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.cls_name)

    # Must match your env ctor signature (StereoPickCube does).
    env = cls(
        config=spec.config,
        config_overrides=spec.config_overrides,
        render_batch_size=int(spec.render_batch_size),
        render_width=int(spec.render_width),
        render_height=int(spec.render_height),
    )
    return env


def _broadcast_tree_to_batch(tree, B: int):
    """Broadcast a pytree of arrays/scalars to a leading batch dim B."""
    def _bcast(x):
        if not hasattr(x, "ndim"):
            return x
        if x.ndim == 0:
            return jp.broadcast_to(x, (B,))
        return jp.broadcast_to(x, (B,) + x.shape)
    return jax.tree_util.tree_map(_bcast, tree)


def _strip_leading_batch_dim(shape: Tuple[int, ...], B: int, *, unbatched_rank: int) -> Tuple[int, ...]:
    """Strip a leading batch dim B only when the rank matches the batched case.

    This avoids the common footgun where an unbatched tensor happens to have
    its first dimension equal to B (e.g. ngeom == num_worlds).
    """
    shape = tuple(shape)
    if len(shape) == unbatched_rank + 1 and shape[0] == B:
        return tuple(shape[1:])
    return shape


def _stitch_stereo_uint8_rgba_to_obs_uint8(rgb_rgba_u8: np.ndarray) -> np.ndarray:
    """
    Input:  [B, 2, H, W, 4] uint8
    Output: [B, H, 2W, 3] uint8
    """
    rgb = rgb_rgba_u8[..., :3]   # [B,2,H,W,3]
    left = rgb[:, 0]             # [B,H,W,3]
    right = rgb[:, 1]            # [B,H,W,3]
    obs = np.concatenate([left, right], axis=2)  # [B,H,2W,3]
    return np.ascontiguousarray(obs)


def _renderer_worker_main(
    env_spec: _EnvCtorSpec,
    shm: _ShmSpec,
    req_ev,
    done_ev,
    stop_ev,
    pipe_conn,
):
    """
    Process A: owns Madrona renderer + render_token.
    Does **only** renderer.init / renderer.render + device_get.
    No MJX forward, no vmap/jit, no Torch.

    The main process must send already-forwarded pose buffers:
      - geom_xpos / geom_xmat
      - cam_xpos / cam_xmat
      - (optional) light_xpos / light_xdir
    because Madrona-MJX supports rendering position/rotation updates for
    rigid bodies, cameras, and lights.
    """
    try:
        env = _import_env_from_spec(env_spec)
        B = shm.B

        # --- Grab MJX model and renderer from env (matches your env snippet) ---
        mjx_model = env.mjx_model  # property -> self._mjx_model
        renderer = env.renderer    # BatchRenderer created in env __init__
        render_token = None

        # Attach shared memory (inputs: xforms, output: raw RGBA)
        geom_xpos_shm = SharedMemory(name=shm.geom_xpos_name)
        geom_xmat_shm = SharedMemory(name=shm.geom_xmat_name)
        cam_xpos_shm = SharedMemory(name=shm.cam_xpos_name)
        cam_xmat_shm = SharedMemory(name=shm.cam_xmat_name)

        light_xpos_shm = SharedMemory(name=shm.light_xpos_name) if shm.light_xpos_name else None
        light_xdir_shm = SharedMemory(name=shm.light_xdir_name) if shm.light_xdir_name else None

        out_rgba_shm = SharedMemory(name=shm.out_rgba_name)

        geom_xpos_np = np.ndarray((B,) + tuple(shm.geom_xpos_shape), dtype=np.float32, buffer=geom_xpos_shm.buf)
        geom_xmat_np = np.ndarray((B,) + tuple(shm.geom_xmat_shape), dtype=np.float32, buffer=geom_xmat_shm.buf)
        cam_xpos_np = np.ndarray((B,) + tuple(shm.cam_xpos_shape), dtype=np.float32, buffer=cam_xpos_shm.buf)
        cam_xmat_np = np.ndarray((B,) + tuple(shm.cam_xmat_shape), dtype=np.float32, buffer=cam_xmat_shm.buf)

        light_xpos_np = None
        light_xdir_np = None
        if light_xpos_shm is not None and light_xdir_shm is not None:
            light_xpos_np = np.ndarray((B,) + tuple(shm.light_xpos_shape), dtype=np.float32, buffer=light_xpos_shm.buf)
            light_xdir_np = np.ndarray((B,) + tuple(shm.light_xdir_shape), dtype=np.float32, buffer=light_xdir_shm.buf)

        out_rgba_np = np.ndarray((B, 2, shm.H, shm.W, 4), dtype=np.uint8, buffer=out_rgba_shm.buf)

        # Build a template mjx.Data once (NO mjx.forward here).
        # If the model is already batched (recommended for Madrona), mjx.make_data
        # returns a batched data structure with leading dim B. Otherwise, we
        # broadcast a single-world template to B.
        data0 = mjx.make_data(mjx_model)
        qpos0 = getattr(data0, "qpos", None)
        already_batched = (
            hasattr(qpos0, "ndim")
            and qpos0.ndim >= 2
            and int(qpos0.shape[0]) == B
        )
        data_b = data0 if already_batched else _broadcast_tree_to_batch(data0, B)

        # Init token ONCE on batched data (renderer compiles its custom calls here)
        render_token, _init_rgb, _init_depth = renderer.init(data_b, mjx_model)

        print(f"[renderer-worker] ready (token init ok, compiled ok) B={B} H={shm.H} W={shm.W}", flush=True)
        pipe_conn.send(("ready", {"B": B, "H": shm.H, "W": shm.W}))

        first = True
        while not stop_ev.is_set():
            if not req_ev.wait(timeout=0.1):
                continue
            req_ev.clear()
            if stop_ev.is_set():
                break

            if first:
                print("[renderer-worker] first request", flush=True)
                first = False

            # Read poses -> device (host->device copies; no extra compute)
            geom_xpos = jp.asarray(geom_xpos_np)
            geom_xmat = jp.asarray(geom_xmat_np)
            cam_xpos = jp.asarray(cam_xpos_np)
            cam_xmat = jp.asarray(cam_xmat_np)

            repl = dict(
                geom_xpos=geom_xpos,
                geom_xmat=geom_xmat,
                cam_xpos=cam_xpos,
                cam_xmat=cam_xmat,
            )
            if light_xpos_np is not None and light_xdir_np is not None:
                repl["light_xpos"] = jp.asarray(light_xpos_np)
                repl["light_xdir"] = jp.asarray(light_xdir_np)

            data_b = data_b.replace(**repl)

            # Render (compiled): rgb is still unsafe as JAX value
            render_token, rgb, _depth = renderer.render(render_token, data_b, mjx_model)
            jax.block_until_ready(rgb)

            # Hard copy to CPU uint8 (this is the boundary between worker/main)
            rgb_h = np.asarray(jax.device_get(rgb))  # [B,2,H,W,4] uint8
            np.copyto(out_rgba_np, rgb_h, casting="no")

            done_ev.set()

        # cleanup
        geom_xpos_shm.close(); geom_xmat_shm.close(); cam_xpos_shm.close(); cam_xmat_shm.close()
        if light_xpos_shm is not None:
            light_xpos_shm.close()
        if light_xdir_shm is not None:
            light_xdir_shm.close()
        out_rgba_shm.close()
        pipe_conn.close()

    except Exception as e:
        tb = traceback.format_exc()
        try:
            pipe_conn.send(("error", {"exc": repr(e), "traceback": tb}))
        except Exception:
            pass
        print("[renderer-worker] FATAL:\n" + tb, flush=True)
        raise


class _RemoteRendererClient:
    """
    Main-process handle to renderer worker.
    Inputs/outputs via shared memory; request/response via Events.
    """

    def __init__(self, env_spec: _EnvCtorSpec, mjx_model: mjx.Model):

        B = int(env_spec.render_batch_size)
        H = int(env_spec.render_height)
        W = int(env_spec.render_width)

        # Discover the exact per-world pose buffer shapes from a template mjx.Data.
        # (No mjx.forward; this is just shape inspection.)
        d0 = mjx.make_data(mjx_model)

        geom_xpos_shape = _strip_leading_batch_dim(tuple(getattr(d0, "geom_xpos").shape), B, unbatched_rank=2)
        geom_xmat_shape = _strip_leading_batch_dim(tuple(getattr(d0, "geom_xmat").shape), B, unbatched_rank=3)
        cam_xpos_shape = _strip_leading_batch_dim(tuple(getattr(d0, "cam_xpos").shape), B, unbatched_rank=2)
        cam_xmat_shape = _strip_leading_batch_dim(tuple(getattr(d0, "cam_xmat").shape), B, unbatched_rank=3)

        # Lights are optional depending on the MJCF.
        has_light = hasattr(d0, "light_xpos") and hasattr(d0, "light_xdir")
        light_xpos_shape: Tuple[int, ...] = (
            _strip_leading_batch_dim(tuple(getattr(d0, "light_xpos").shape), B, unbatched_rank=2) if has_light else tuple()
        )
        light_xdir_shape: Tuple[int, ...] = (
            _strip_leading_batch_dim(tuple(getattr(d0, "light_xdir").shape), B, unbatched_rank=2) if has_light else tuple()
        )

        def _nbytes_f32(shape: Tuple[int, ...]) -> int:
            n = int(np.prod(shape)) if len(shape) else 0
            return int(B * n * 4)

        # shm allocations (render-only)
        self._geom_xpos = SharedMemory(create=True, size=_nbytes_f32(geom_xpos_shape))
        self._geom_xmat = SharedMemory(create=True, size=_nbytes_f32(geom_xmat_shape))
        self._cam_xpos = SharedMemory(create=True, size=_nbytes_f32(cam_xpos_shape))
        self._cam_xmat = SharedMemory(create=True, size=_nbytes_f32(cam_xmat_shape))

        self._light_xpos = SharedMemory(create=True, size=_nbytes_f32(light_xpos_shape)) if has_light else None
        self._light_xdir = SharedMemory(create=True, size=_nbytes_f32(light_xdir_shape)) if has_light else None

        # Raw RGBA out: [B,2,H,W,4] uint8
        self._out_rgba = SharedMemory(create=True, size=int(B * 2 * H * W * 4))

        self.shm_spec = _ShmSpec(
            geom_xpos_name=self._geom_xpos.name,
            geom_xmat_name=self._geom_xmat.name,
            cam_xpos_name=self._cam_xpos.name,
            cam_xmat_name=self._cam_xmat.name,
            light_xpos_name=self._light_xpos.name if self._light_xpos is not None else "",
            light_xdir_name=self._light_xdir.name if self._light_xdir is not None else "",
            out_rgba_name=self._out_rgba.name,
            B=B,
            geom_xpos_shape=geom_xpos_shape,
            geom_xmat_shape=geom_xmat_shape,
            cam_xpos_shape=cam_xpos_shape,
            cam_xmat_shape=cam_xmat_shape,
            light_xpos_shape=light_xpos_shape,
            light_xdir_shape=light_xdir_shape,
            H=H,
            W=W,
        )

        # typed views
        self.geom_xpos_np = np.ndarray((B,) + geom_xpos_shape, dtype=np.float32, buffer=self._geom_xpos.buf)
        self.geom_xmat_np = np.ndarray((B,) + geom_xmat_shape, dtype=np.float32, buffer=self._geom_xmat.buf)
        self.cam_xpos_np = np.ndarray((B,) + cam_xpos_shape, dtype=np.float32, buffer=self._cam_xpos.buf)
        self.cam_xmat_np = np.ndarray((B,) + cam_xmat_shape, dtype=np.float32, buffer=self._cam_xmat.buf)

        self.light_xpos_np = (
            np.ndarray((B,) + light_xpos_shape, dtype=np.float32, buffer=self._light_xpos.buf)
            if self._light_xpos is not None else None
        )
        self.light_xdir_np = (
            np.ndarray((B,) + light_xdir_shape, dtype=np.float32, buffer=self._light_xdir.buf)
            if self._light_xdir is not None else None
        )

        self.out_rgba_np = np.ndarray((B, 2, H, W, 4), dtype=np.uint8, buffer=self._out_rgba.buf)

        ctx = mp.get_context("spawn")
        self.req_ev = ctx.Event()
        self.done_ev = ctx.Event()
        self.stop_ev = ctx.Event()
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        self._conn = parent_conn

        print(f"[renderer-main] spawn worker (B={B}, H={H}, W={W})", flush=True)
        self.proc = ctx.Process(
            target=_renderer_worker_main,
            args=(env_spec, self.shm_spec, self.req_ev, self.done_ev, self.stop_ev, child_conn),
            daemon=True,
        )
        self.proc.start()

        # wait for ready/error
        t0 = time.time()
        while True:
            if self._conn.poll(0.1):
                msg, payload = self._conn.recv()
                if msg == "ready":
                    print(f"[renderer-main] worker ready: {payload}", flush=True)
                    break
                if msg == "error":
                    raise RuntimeError(f"Renderer worker failed: {payload.get('exc')}\n{payload.get('traceback')}")
            if not self.proc.is_alive():
                raise RuntimeError("Renderer worker died during startup.")
            if time.time() - t0 > 600:
                raise RuntimeError("Renderer worker init timed out (>600s).")

    def close(self):
        try:
            self.stop_ev.set()
            self.req_ev.set()
        except Exception:
            pass

        try:
            if self.proc.is_alive():
                self.proc.join(timeout=2.0)
        except Exception:
            pass

        try:
            self._conn.close()
        except Exception:
            pass

        shms = [self._geom_xpos, self._geom_xmat, self._cam_xpos, self._cam_xmat, self._out_rgba]
        if self._light_xpos is not None:
            shms.append(self._light_xpos)
        if self._light_xdir is not None:
            shms.append(self._light_xdir)

        for shm in shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def render_rgba_from_cpu_xforms(
        self,
        geom_xpos_cpu: np.ndarray,
        geom_xmat_cpu: np.ndarray,
        cam_xpos_cpu: np.ndarray,
        cam_xmat_cpu: np.ndarray,
        light_xpos_cpu: Optional[np.ndarray] = None,
        light_xdir_cpu: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.proc.is_alive():
            if self._conn.poll(0.0):
                msg, payload = self._conn.recv()
                if msg == "error":
                    raise RuntimeError(f"Renderer worker crashed: {payload.get('exc')}\n{payload.get('traceback')}")
            raise RuntimeError("Renderer worker not alive.")

        np.copyto(self.geom_xpos_np, geom_xpos_cpu, casting="unsafe")
        np.copyto(self.geom_xmat_np, geom_xmat_cpu, casting="unsafe")
        np.copyto(self.cam_xpos_np, cam_xpos_cpu, casting="unsafe")
        np.copyto(self.cam_xmat_np, cam_xmat_cpu, casting="unsafe")

        if self.light_xpos_np is not None and light_xpos_cpu is not None:
            np.copyto(self.light_xpos_np, light_xpos_cpu, casting="unsafe")
        if self.light_xdir_np is not None and light_xdir_cpu is not None:
            np.copyto(self.light_xdir_np, light_xdir_cpu, casting="unsafe")

        self.done_ev.clear()
        self.req_ev.set()
        self.done_ev.wait()

        return np.ascontiguousarray(self.out_rgba_np)  # [B,2,H,W,4] uint8


# ---------------------------
# Public wrapper: physics in main, render in worker
# ---------------------------

class RSLRLBraxWrapper:
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
        import torch
        from tensordict import TensorDict  # noqa: F401 (runtime import for main only)

        self._raw_env = env
        self.batch_size = int(batch_size)
        self.num_envs = int(batch_size)
        self.episode_length = int(episode_length)
        self.action_repeat = int(action_repeat)
        self.render_callback = render_callback

        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # sanity: renderer worlds must match envs
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"env.render_batch_size must equal num_envs. Got {rbs} vs {self.batch_size}"
                )

        # RNG keys per env (physics)
        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        # Batched physics
        self._reset_batched = jax.vmap(self._raw_env.reset_physics)   # single-world -> batched
        self._step_batched = jax.vmap(self._raw_env.step_physics, in_axes=(0, 0), out_axes=0)

        self.reset_fn = jax.jit(self._reset_batched)

        if self.action_repeat == 1:
            self.step_fn = jax.jit(self._step_batched)
        else:
            def _step_repeat(state, action):
                def body(_, st):
                    return self._step_batched(st, action)
                return jax.lax.fori_loop(0, self.action_repeat, body, state)
            self.step_fn = jax.jit(_step_repeat)

        self.env_state = None
        self._remote: Optional[_RemoteRendererClient] = None

        # Spec to reconstruct env inside worker (do not pickle mujoco handles)
        env_mod = self._raw_env.__class__.__module__
        env_cls = self._raw_env.__class__.__name__
        cfg = getattr(self._raw_env, "_config", None)
        cfg_over = getattr(self._raw_env, "_config_overrides", None)

        self._env_spec = _EnvCtorSpec(
            module=env_mod,
            cls_name=env_cls,
            config=cfg,
            config_overrides=cfg_over,
            render_batch_size=self.batch_size,
            render_width=int(getattr(self._raw_env, "render_width", 64)),
            render_height=int(getattr(self._raw_env, "render_height", 64)),
        )

    def close(self):
        if self._remote is not None:
            self._remote.close()
            self._remote = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _ensure_remote(self):
        if self._remote is None:
            self._remote = _RemoteRendererClient(self._env_spec, self._raw_env.mjx_model)

    def _render_obs_from_state(self) -> torch.Tensor:
        import torch
        assert self.env_state is not None
        self._ensure_remote()

        data = self.env_state.data

        # Minimal CPU handoff: already-forwarded kinematics only
        geom_xpos_cpu = np.asarray(jax.device_get(data.geom_xpos)).astype(np.float32, copy=False)
        geom_xmat_cpu = np.asarray(jax.device_get(data.geom_xmat)).astype(np.float32, copy=False)
        cam_xpos_cpu = np.asarray(jax.device_get(data.cam_xpos)).astype(np.float32, copy=False)
        cam_xmat_cpu = np.asarray(jax.device_get(data.cam_xmat)).astype(np.float32, copy=False)

        light_xpos_cpu = None
        light_xdir_cpu = None
        if hasattr(data, "light_xpos") and hasattr(data, "light_xdir"):
            light_xpos_cpu = np.asarray(jax.device_get(data.light_xpos)).astype(np.float32, copy=False)
            light_xdir_cpu = np.asarray(jax.device_get(data.light_xdir)).astype(np.float32, copy=False)

        rgba = self._remote.render_rgba_from_cpu_xforms(
            geom_xpos_cpu,
            geom_xmat_cpu,
            cam_xpos_cpu,
            cam_xmat_cpu,
            light_xpos_cpu=light_xpos_cpu,
            light_xdir_cpu=light_xdir_cpu,
        )  # [B,2,H,W,4] uint8

        obs_h = _stitch_stereo_uint8_rgba_to_obs_uint8(rgba)  # [B,H,2W,3] uint8
        obs_t = torch.from_numpy(obs_h)
        if self.device.type == "cuda":
            obs_t = obs_t.to(self.device, non_blocking=True)
        return obs_t

    def reset(self) -> TensorDict:
        from tensordict import TensorDict
        self.env_state = self.reset_fn(self.key_reset)

        obs_t = self._render_obs_from_state()
        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        import torch
        from tensordict import TensorDict
        assert self.env_state is not None, "Call reset() before step()."

        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        obs_t = self._render_obs_from_state()

        reward_t = _jax_to_torch_hostcopy(self.env_state.reward, self.device).reshape(-1).to(torch.float32)
        done_t = _jax_to_torch_hostcopy(self.env_state.done, self.device).reshape(-1).to(torch.bool)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self.render_callback is not None:
            self.render_callback(None, self.env_state)

        return obs_td, reward_t, done_t, info
