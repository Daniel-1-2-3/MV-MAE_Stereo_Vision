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


# ---------------------------
# Torch/JAX helpers
# ---------------------------

def _torch_to_jax(x: torch.Tensor) -> jax.Array:
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
    qpos_name: str
    qvel_name: str
    ctrl_name: str
    mocap_pos_name: str
    mocap_quat_name: str
    out_obs_name: str
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
    Produces CPU uint8 pixels in shared memory.
    """
    try:
        env = _import_env_from_spec(env_spec)
        B = shm.B

        # --- Grab MJX model and renderer from env (matches your env snippet) ---
        mjx_model = env.mjx_model  # property -> self._mjx_model
        renderer = env.renderer    # BatchRenderer created in env __init__
        render_token = None

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

        # Build a template batched mjx.Data once.
        # We mirror your env logic: a proper mjx.Data that has all fields, then batched.
        data0 = mjx.make_data(mjx_model)
        data0 = mjx.forward(mjx_model, data0)
        data_b = _broadcast_tree_to_batch(data0, B)

        # IMPORTANT: do NOT vmap mjx.forward over a broadcasted mjx.Data.
        # This is the path that crashes inside MJX scan/take with cudaErrorInvalidValue.
        # Renderer init works fine on the batched data as-is.
        # Init token ONCE on batched data (your exact contract)
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

            # Read CPU state -> device
            qpos = jp.asarray(qpos_np)
            qvel = jp.asarray(qvel_np)
            ctrl = jp.asarray(ctrl_np)
            mocap_pos = jp.asarray(mocap_pos_np)
            mocap_quat = jp.asarray(mocap_quat_np)

            # Create once, near where you build data_b / init the renderer token:
            fwd_batched = jax.jit(jax.vmap(lambda d: mjx.forward(mjx_model, d)))

            # Then per request:
            data_b = data_b.replace(
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                mocap_pos=mocap_pos,
                mocap_quat=mocap_quat,
            )
            data_b = fwd_batched(data_b)

            # Render (compiled): rgb is still unsafe as JAX value
            render_token, rgb, _depth = renderer.render(render_token, data_b, mjx_model)
            jax.block_until_ready(rgb)

            # Break context poisoning: hard copy to CPU uint8
            rgb_h = np.asarray(jax.device_get(rgb))  # expected [B,2,H,W,4] uint8

            # Stitch stereo and write to shm
            obs_h = _stitch_stereo_uint8_rgba_to_obs_uint8(rgb_h)
            np.copyto(out_obs_np, obs_h, casting="no")

            done_ev.set()

        # cleanup
        qpos_shm.close(); qvel_shm.close(); ctrl_shm.close()
        mocap_pos_shm.close(); mocap_quat_shm.close(); out_obs_shm.close()
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

    def __init__(self, env_spec: _EnvCtorSpec, mjx_model: mjx.Model, device: torch.device):
        self.device = device

        B = int(env_spec.render_batch_size)
        H = int(env_spec.render_height)
        W = int(env_spec.render_width)

        nq = int(mjx_model.nq)
        nv = int(mjx_model.nv)
        nu = int(getattr(mjx_model, "nu", 0)) or 1
        nmocap = int(getattr(mjx_model, "nmocap", 0)) or 1

        # shm allocations
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

        # typed views
        self.qpos_np = np.ndarray((B, nq), dtype=np.float32, buffer=self._qpos.buf)
        self.qvel_np = np.ndarray((B, nv), dtype=np.float32, buffer=self._qvel.buf)
        self.ctrl_np = np.ndarray((B, nu), dtype=np.float32, buffer=self._ctrl.buf)
        self.mocap_pos_np = np.ndarray((B, nmocap, 3), dtype=np.float32, buffer=self._mocap_pos.buf)
        self.mocap_quat_np = np.ndarray((B, nmocap, 4), dtype=np.float32, buffer=self._mocap_quat.buf)
        self.out_obs_np = np.ndarray((B, H, 2 * W, 3), dtype=np.uint8, buffer=self._out_obs.buf)

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

        for shm in [self._qpos, self._qvel, self._ctrl, self._mocap_pos, self._mocap_quat, self._out_obs]:
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

    def render_obs_from_cpu_state(
        self,
        qpos_cpu: np.ndarray,
        qvel_cpu: np.ndarray,
        ctrl_cpu: np.ndarray,
        mocap_pos_cpu: np.ndarray,
        mocap_quat_cpu: np.ndarray,
    ) -> torch.Tensor:
        if not self.proc.is_alive():
            if self._conn.poll(0.0):
                msg, payload = self._conn.recv()
                if msg == "error":
                    raise RuntimeError(f"Renderer worker crashed: {payload.get('exc')}\n{payload.get('traceback')}")
            raise RuntimeError("Renderer worker not alive.")

        np.copyto(self.qpos_np, qpos_cpu, casting="unsafe")
        np.copyto(self.qvel_np, qvel_cpu, casting="unsafe")
        np.copyto(self.ctrl_np, ctrl_cpu, casting="unsafe")
        np.copyto(self.mocap_pos_np, mocap_pos_cpu, casting="unsafe")
        np.copyto(self.mocap_quat_np, mocap_quat_cpu, casting="unsafe")

        self.done_ev.clear()
        self.req_ev.set()
        self.done_ev.wait()

        obs_h = np.ascontiguousarray(self.out_obs_np)  # [B,H,2W,3] uint8
        obs_t = torch.from_numpy(obs_h)
        if self.device.type == "cuda":
            obs_t = obs_t.to(self.device, non_blocking=True)
        return obs_t


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
            self._remote = _RemoteRendererClient(self._env_spec, self._raw_env.mjx_model, self.device)

    def _render_obs_from_state(self) -> torch.Tensor:
        assert self.env_state is not None
        self._ensure_remote()

        data = self.env_state.data

        # Minimal CPU handoff
        qpos_cpu = np.asarray(jax.device_get(data.qpos)).astype(np.float32, copy=False)
        qvel_cpu = np.asarray(jax.device_get(data.qvel)).astype(np.float32, copy=False)

        ctrl = getattr(data, "ctrl", None)
        if ctrl is None:
            ctrl_cpu = np.zeros_like(self._remote.ctrl_np)
        else:
            ctrl_cpu = np.asarray(jax.device_get(ctrl)).astype(np.float32, copy=False)

        mocap_pos = getattr(data, "mocap_pos", None)
        mocap_quat = getattr(data, "mocap_quat", None)
        if mocap_pos is None:
            mocap_pos_cpu = np.zeros_like(self._remote.mocap_pos_np)
        else:
            mocap_pos_cpu = np.asarray(jax.device_get(mocap_pos)).astype(np.float32, copy=False)
        if mocap_quat is None:
            mocap_quat_cpu = np.zeros_like(self._remote.mocap_quat_np)
        else:
            mocap_quat_cpu = np.asarray(jax.device_get(mocap_quat)).astype(np.float32, copy=False)

        return self._remote.render_obs_from_cpu_state(qpos_cpu, qvel_cpu, ctrl_cpu, mocap_pos_cpu, mocap_quat_cpu)

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

        reward_t = _jax_to_torch_hostcopy(self.env_state.reward, self.device).reshape(-1).to(torch.float32)
        done_t = _jax_to_torch_hostcopy(self.env_state.done, self.device).reshape(-1).to(torch.bool)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self.render_callback is not None:
            self.render_callback(None, self.env_state)

        return obs_td, reward_t, done_t, info
