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
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack

from madrona_mjx.wrapper import _identity_randomization_fn


# ---------------------------
# Torch/JAX helpers
# ---------------------------

def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    # actions: zero-copy is fine (not a renderer output)
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


def _jax_to_torch_hostcopy(x: jax.Array, device: torch.device) -> torch.Tensor:
    """Break any GPU pointer sharing: JAX -> host numpy -> Torch tensor -> (optional) GPU."""
    arr = np.asarray(jax.device_get(x))
    t = torch.from_numpy(arr)
    if device.type == "cuda":
        t = t.to(device, non_blocking=True)
    return t


# ---------------------------
# Renderer worker spec
# ---------------------------

@dataclass(frozen=True)
class _EnvCtorSpec:
    module: str
    cls_name: str
    # config objects are typically pickleable for ml_collections ConfigDict
    config: Any
    config_overrides: Optional[Dict[str, Any]]
    render_batch_size: int
    render_width: int
    render_height: int


@dataclass(frozen=True)
class _ShmSpec:
    # shm names
    qpos_name: str
    qvel_name: str
    ctrl_name: str
    mocap_pos_name: str
    mocap_quat_name: str
    out_obs_name: str
    # shapes
    B: int
    nq: int
    nv: int
    nu: int
    nmocap: int
    H: int
    W: int


def _import_env_from_spec(spec: _EnvCtorSpec):
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.cls_name)
    # NOTE: StereoPickCube signature matches these args; if you use a different env class,
    # adjust this constructor accordingly.
    env = cls(
        config=spec.config,
        config_overrides=spec.config_overrides,
        render_batch_size=int(spec.render_batch_size),
        render_width=int(spec.render_width),
        render_height=int(spec.render_height),
    )
    return env


def _renderer_worker_main(
    env_spec: _EnvCtorSpec,
    shm: _ShmSpec,
    req_ev,
    done_ev,
    stop_ev,
    pipe_conn,
):
    """
    Runs in a separate OS process.
    Owns Madrona + its JAX CUDA context.
    Communicates via shared memory + events.
    """
    try:
        # Be explicit: avoid accidental CPU backend
        # (inherits JAX_PLATFORMS from env; fine)
        first_req = True

        env = _import_env_from_spec(env_spec)
        B = shm.B

        mjx_model = env.mjx_model
        renderer = env.make_renderer_debug(B)

        # identity randomization / vmap axes exactly like debug.py patterns
        v_mjx_model, v_in_axes = _identity_randomization_fn(mjx_model, B)

        # --- init_one: make_data + forward + renderer.init ---
        def init_one(rng, model):
            data = mjx.make_data(model)
            data = mjx.forward(model, data)
            render_token, rgb, depth = renderer.init(data, model)
            return data, render_token, rgb, depth

        init_one_jit = jax.jit(init_one)

        rng = jax.random.PRNGKey(2)
        rng, *keys = jax.random.split(rng, B + 1)

        v_mjx_data, render_token, _rgb0, _depth0 = jax.vmap(
            init_one_jit, in_axes=[0, v_in_axes]
        )(jp.asarray(keys), v_mjx_model)

        # render compiled step (debug-like)
        def step(data):
            def step_(d):
                _, rgb, depth = renderer.render(render_token, d)
                return d, rgb, depth
            return jax.vmap(step_)(data)

        render_step = jax.jit(step).lower(v_mjx_data).compile()

        # Attach shared memory
        qpos_shm = SharedMemory(name=shm.qpos_name)
        qvel_shm = SharedMemory(name=shm.qvel_name)
        ctrl_shm = SharedMemory(name=shm.ctrl_name)
        mocap_pos_shm = SharedMemory(name=shm.mocap_pos_name)
        mocap_quat_shm = SharedMemory(name=shm.mocap_quat_name)
        out_obs_shm = SharedMemory(name=shm.out_obs_name)

        qpos_np = np.ndarray((B, shm.nq), dtype=np.float32, buffer=qpos_shm.buf)
        qvel_np = np.ndarray((B, shm.nv), dtype=np.float32, buffer=qvel_shm.buf)
        ctrl_np = np.ndarray((B, shm.nu), dtype=np.float32, buffer=ctrl_shm.buf)
        mocap_pos_np = np.ndarray((B, shm.nmocap, 3), dtype=np.float32, buffer=mocap_pos_shm.buf)
        mocap_quat_np = np.ndarray((B, shm.nmocap, 4), dtype=np.float32, buffer=mocap_quat_shm.buf)
        out_obs_np = np.ndarray((B, shm.H, 2 * shm.W, 3), dtype=np.uint8, buffer=out_obs_shm.buf)

        pipe_conn.send(("ready", {"B": B, "H": shm.H, "W": shm.W}))

        while not stop_ev.is_set():
            if not req_ev.wait(timeout=0.1):
                continue
            req_ev.clear()

            if stop_ev.is_set():
                break

            if first_req:
                print(f"[renderer-worker] first render request received (B={B}, H={shm.H}, W={shm.W})", flush=True)
                first_req = False

            # Read CPU state (already in shared memory), upload to device as float32
            qpos = jp.asarray(qpos_np)          # host->device copy
            qvel = jp.asarray(qvel_np)
            ctrl = jp.asarray(ctrl_np)
            mocap_pos = jp.asarray(mocap_pos_np)
            mocap_quat = jp.asarray(mocap_quat_np)

            # Update data and forward before render (so derived geom/cam transforms are correct)
            # v_mjx_data is a batched mjx.Data pytree from init.
            data = v_mjx_data.replace(
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                mocap_pos=mocap_pos,
                mocap_quat=mocap_quat,
            )

            # forward per-world (vmap over model axes and data axis)
            data = jax.vmap(lambda m, d: mjx.forward(m, d), in_axes=(v_in_axes, 0))(v_mjx_model, data)

            # Render (compiled)
            _d_out, rgba, _depth = render_step(data)  # rgba: uint8 [B,2,H,W,4] per your renderer

            # Hard break from JAX/CUDA state: copy to host as uint8
            rgba_h = np.asarray(jax.device_get(rgba))  # [B,2,H,W,4] uint8 on CPU

            # Stitch stereo in NumPy: [B,H,2W,3]
            rgb = rgba_h[..., :3]
            left = rgb[:, 0]
            right = rgb[:, 1]
            obs_h = np.concatenate([left, right], axis=2)
            obs_h = np.ascontiguousarray(obs_h)

            # Write to shared out buffer
            np.copyto(out_obs_np, obs_h, casting="no")

            done_ev.set()

        # cleanup
        qpos_shm.close()
        qvel_shm.close()
        ctrl_shm.close()
        mocap_pos_shm.close()
        mocap_quat_shm.close()
        out_obs_shm.close()

        pipe_conn.close()

    except Exception as e:
        tb = traceback.format_exc()
        try:
            pipe_conn.send(("error", {"exc": repr(e), "traceback": tb}))
        except Exception:
            pass
        # Print once for cluster logs
        print("[renderer-worker] FATAL:\n" + tb, flush=True)
        raise


class _RemoteRendererClient:
    """
    Main-process handle to the renderer subprocess.
    Shared memory for inputs/outputs; Events for request/response.
    """

    def __init__(self, env_spec: _EnvCtorSpec, mjx_model: mjx.Model, device: torch.device):
        self.device = device

        B = int(env_spec.render_batch_size)
        H = int(env_spec.render_width)   # debug uses width twice
        W = int(env_spec.render_width)

        nq = int(mjx_model.nq)
        nv = int(mjx_model.nv)
        nu = int(getattr(mjx_model, "nu", 0))  # mjx.Model has nu (actuators)
        if nu <= 0:
            # Fallback: ctrl length often equals nv or number of actuators; but for Panda it should be >0.
            # If this triggers, we’ll still allocate a minimal ctrl to avoid crashes.
            nu = 1

        # mocap sizes
        # mjx_model.nmocap exists in MuJoCo/MJX
        nmocap = int(getattr(mjx_model, "nmocap", 1))
        if nmocap <= 0:
            nmocap = 1

        self.shm_spec = _ShmSpec(
            qpos_name="",
            qvel_name="",
            ctrl_name="",
            mocap_pos_name="",
            mocap_quat_name="",
            out_obs_name="",
            B=B, nq=nq, nv=nv, nu=nu, nmocap=nmocap, H=H, W=W
        )

        # Allocate shared memory blocks
        self._qpos = SharedMemory(create=True, size=B * nq * 4)
        self._qvel = SharedMemory(create=True, size=B * nv * 4)
        self._ctrl = SharedMemory(create=True, size=B * nu * 4)
        self._mocap_pos = SharedMemory(create=True, size=B * nmocap * 3 * 4)
        self._mocap_quat = SharedMemory(create=True, size=B * nmocap * 4 * 4)
        self._out_obs = SharedMemory(create=True, size=B * H * (2 * W) * 3)

        self.shm_spec = _ShmSpec(
            qpos_name=self._qpos.name,
            qvel_name=self._qvel.name,
            ctrl_name=self._ctrl.name,
            mocap_pos_name=self._mocap_pos.name,
            mocap_quat_name=self._mocap_quat.name,
            out_obs_name=self._out_obs.name,
            B=B, nq=nq, nv=nv, nu=nu, nmocap=nmocap, H=H, W=W
        )

        # Views for writing/reading
        self.qpos_np = np.ndarray((B, nq), dtype=np.float32, buffer=self._qpos.buf)
        self.qvel_np = np.ndarray((B, nv), dtype=np.float32, buffer=self._qvel.buf)
        self.ctrl_np = np.ndarray((B, nu), dtype=np.float32, buffer=self._ctrl.buf)
        self.mocap_pos_np = np.ndarray((B, nmocap, 3), dtype=np.float32, buffer=self._mocap_pos.buf)
        self.mocap_quat_np = np.ndarray((B, nmocap, 4), dtype=np.float32, buffer=self._mocap_quat.buf)
        self.out_obs_np = np.ndarray((B, H, 2 * W, 3), dtype=np.uint8, buffer=self._out_obs.buf)

        # Sync primitives
        ctx = mp.get_context("spawn")
        self.req_ev = ctx.Event()
        self.done_ev = ctx.Event()
        self.stop_ev = ctx.Event()
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        self._conn = parent_conn

        print(f"[renderer-main] starting renderer worker (spawn) B={B} H={H} W={W}", flush=True)

        self.proc = ctx.Process(
            target=_renderer_worker_main,
            args=(env_spec, self.shm_spec, self.req_ev, self.done_ev, self.stop_ev, child_conn),
            daemon=True,
        )
        self.proc.start()

        # Wait for ready/error (single print, no periodic spam)
        t0 = time.time()
        while True:
            if self._conn.poll(0.1):
                msg, payload = self._conn.recv()
                if msg == "ready":
                    print(f"[renderer-main] worker ready: {payload}", flush=True)
                    break
                if msg == "error":
                    raise RuntimeError(f"Renderer worker failed during init: {payload.get('exc')}\n{payload.get('traceback')}")
            if not self.proc.is_alive():
                raise RuntimeError("Renderer worker died during startup (no error payload).")
            if time.time() - t0 > 600:
                raise RuntimeError("Renderer worker init timed out (>600s).")

    def close(self):
        try:
            if hasattr(self, "stop_ev"):
                self.stop_ev.set()
            if hasattr(self, "req_ev"):
                self.req_ev.set()
        except Exception:
            pass

        try:
            if hasattr(self, "proc") and self.proc.is_alive():
                self.proc.join(timeout=2.0)
        except Exception:
            pass

        try:
            if hasattr(self, "_conn"):
                self._conn.close()
        except Exception:
            pass

        # Unlink shared memory
        for shm in [getattr(self, "_qpos", None), getattr(self, "_qvel", None), getattr(self, "_ctrl", None),
                    getattr(self, "_mocap_pos", None), getattr(self, "_mocap_quat", None), getattr(self, "_out_obs", None)]:
            if shm is None:
                continue
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

    def render_obs(
        self,
        qpos_cpu: np.ndarray,
        qvel_cpu: np.ndarray,
        ctrl_cpu: np.ndarray,
        mocap_pos_cpu: np.ndarray,
        mocap_quat_cpu: np.ndarray,
    ) -> torch.Tensor:
        if not self.proc.is_alive():
            # try to pull error if available
            if self._conn.poll(0.0):
                msg, payload = self._conn.recv()
                if msg == "error":
                    raise RuntimeError(f"Renderer worker crashed: {payload.get('exc')}\n{payload.get('traceback')}")
            raise RuntimeError("Renderer worker is not alive.")

        # Write inputs
        np.copyto(self.qpos_np, qpos_cpu, casting="unsafe")
        np.copyto(self.qvel_np, qvel_cpu, casting="unsafe")

        # ctrl/mocap might have different shapes; copy defensively
        np.copyto(self.ctrl_np, ctrl_cpu, casting="unsafe")
        np.copyto(self.mocap_pos_np, mocap_pos_cpu, casting="unsafe")
        np.copyto(self.mocap_quat_np, mocap_quat_cpu, casting="unsafe")

        # Request
        self.done_ev.clear()
        self.req_ev.set()

        # Wait
        self.done_ev.wait()

        # Read output and move to torch device
        obs_h = np.ascontiguousarray(self.out_obs_np)  # [B,H,2W,3] uint8
        obs_t = torch.from_numpy(obs_h)
        if self.device.type == "cuda":
            obs_t = obs_t.to(self.device, non_blocking=True)
        return obs_t


# ---------------------------
# Public wrapper
# ---------------------------

class RSLRLBraxWrapper:
    """
    Torch-facing vector-env wrapper.
    Physics in main process (JAX/MJX).
    Rendering in a spawned subprocess (separate JAX CUDA context), returning CPU uint8 pixels via shm.
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
        self.render_callback = render_callback

        # Torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{int(device_rank) if device_rank is not None else 0}")
            else:
                self.device = torch.device("cpu")

        # Enforce renderer batch size matches envs
        if hasattr(self._raw_env, "render_batch_size"):
            rbs = int(getattr(self._raw_env, "render_batch_size"))
            if rbs != self.batch_size:
                raise ValueError(
                    f"StereoPickCube(render_batch_size) must equal num_envs. "
                    f"Got render_batch_size={rbs}, num_envs={self.batch_size}."
                )

        # RNG keys per env (physics)
        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        # Batched physics
        self._reset_batched = jax.vmap(self._raw_env.reset)
        self._step_batched = jax.vmap(self._raw_env.step, in_axes=(0, 0), out_axes=0)

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

        # Renderer subprocess client (lazy init)
        self._remote: Optional[_RemoteRendererClient] = None

        # Build env ctor spec for worker (reconstruct env inside worker; do not pickle mujoco objects)
        env_mod = self._raw_env.__class__.__module__
        env_cls = self._raw_env.__class__.__name__

        # These fields exist on StereoPickCube; if you change env, adjust here.
        cfg = getattr(self._raw_env, "_config", None)
        cfg_over = getattr(self._raw_env, "_config_overrides", None)
        rw = int(getattr(self._raw_env, "render_width", 64))
        rh = int(getattr(self._raw_env, "render_height", 64))

        self._env_spec = _EnvCtorSpec(
            module=env_mod,
            cls_name=env_cls,
            config=cfg,
            config_overrides=cfg_over,
            render_batch_size=self.batch_size,
            render_width=rw,
            render_height=rh,
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
            # requires mjx_model for shapes
            mjx_model = self._raw_env.mjx_model
            self._remote = _RemoteRendererClient(self._env_spec, mjx_model, self.device)

    def _render_obs_from_state(self) -> torch.Tensor:
        assert self.env_state is not None

        self._ensure_remote()

        data = self.env_state.data

        # Copy minimal state to CPU for cross-process handoff
        # These device_get calls synchronize physics stream, but they’re small.
        qpos_cpu = np.asarray(jax.device_get(data.qpos)).astype(np.float32, copy=False)
        qvel_cpu = np.asarray(jax.device_get(data.qvel)).astype(np.float32, copy=False)

        # ctrl might not exist or might be scalar; keep it robust
        ctrl = getattr(data, "ctrl", None)
        if ctrl is None:
            ctrl_cpu = np.zeros((self.batch_size, self._remote.shm_spec.nu), dtype=np.float32)
        else:
            ctrl_cpu = np.asarray(jax.device_get(ctrl)).astype(np.float32, copy=False)
            if ctrl_cpu.ndim == 1:
                ctrl_cpu = np.broadcast_to(ctrl_cpu[None, :], (self.batch_size, ctrl_cpu.shape[0])).copy()

        mocap_pos = getattr(data, "mocap_pos", None)
        mocap_quat = getattr(data, "mocap_quat", None)

        if mocap_pos is None:
            mocap_pos_cpu = np.zeros((self.batch_size, self._remote.shm_spec.nmocap, 3), dtype=np.float32)
        else:
            mocap_pos_cpu = np.asarray(jax.device_get(mocap_pos)).astype(np.float32, copy=False)

        if mocap_quat is None:
            mocap_quat_cpu = np.zeros((self.batch_size, self._remote.shm_spec.nmocap, 4), dtype=np.float32)
        else:
            mocap_quat_cpu = np.asarray(jax.device_get(mocap_quat)).astype(np.float32, copy=False)

        return self._remote.render_obs(qpos_cpu, qvel_cpu, ctrl_cpu, mocap_pos_cpu, mocap_quat_cpu)

    def reset(self) -> TensorDict:
        self.env_state = self.reset_fn(self.key_reset)

        obs_t = self._render_obs_from_state()
        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        obs_t = self._render_obs_from_state()

        # Safe (no-sharing) reward/done transfers too
        reward_t = _jax_to_torch_hostcopy(self.env_state.reward, self.device).reshape(-1).to(torch.float32)
        done_t = _jax_to_torch_hostcopy(self.env_state.done, self.device).reshape(-1).to(torch.bool)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self.render_callback is not None:
            self.render_callback(None, self.env_state)

        return obs_td, reward_t, done_t, info
