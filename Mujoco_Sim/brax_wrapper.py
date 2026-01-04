# Mujoco_Sim/brax_wrapper.py
import os
import importlib
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils import dlpack as tpack

from madrona_mjx.wrapper import _identity_randomization_fn


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:
        return jax.dlpack.from_dlpack(tpack.to_dlpack(x))


# ---------------------------
# Shared-memory renderer worker
# ---------------------------

@dataclass
class _ShmSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str  # numpy dtype str


def _np_dtype(dtype_str: str):
    return np.dtype(dtype_str)


def _create_shm_array(spec: _ShmSpec, create: bool):
    dtype = _np_dtype(spec.dtype)
    nbytes = int(np.prod(spec.shape)) * dtype.itemsize
    shm = SharedMemory(name=spec.name, create=create, size=nbytes if create else 0)
    arr = np.ndarray(spec.shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


def _renderer_worker_main(
    env_module: str,
    env_class: str,
    env_config,
    render_width: int,
    render_height: int,
    chunk: int,
    gpu_id: int,
    enabled_geom_groups: np.ndarray,
    add_cam_debug_geo: bool,
    use_rasterizer: bool,
    shm_qpos: _ShmSpec,
    shm_qvel: _ShmSpec,
    shm_obs: _ShmSpec,
    req_evt: mp.Event,
    resp_evt: mp.Event,
    stop_evt: mp.Event,
):
    """
    Process A: owns Madrona + JAX renderer. Receives qpos/qvel on CPU, renders, writes obs on CPU.
    """
    # Import inside child
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco import mjx
    from madrona_mjx.wrapper import _identity_randomization_fn

    # Build env inside worker (so it can build mjx_model the same way)
    mod = importlib.import_module(env_module)
    EnvCls = getattr(mod, env_class)
    env = EnvCls(config=env_config, render_batch_size=chunk,
                 render_width=render_width, render_height=render_height)

    mjx_model = env.mjx_model

    # Build renderer (debug semantics)
    renderer = env.make_renderer_debug(chunk)

    # Init token using debug-style init (small chunk)
    v_mjx_model, v_in_axes = _identity_randomization_fn(mjx_model, chunk)

    def init_one(rng, model):
        data = mjx.make_data(model)
        data = mjx.forward(model, data)
        render_token, _rgb, _depth = renderer.init(data, model)
        return render_token

    init_one_jit = jax.jit(init_one)
    rng = jax.random.PRNGKey(2)
    rng, *keys = jax.random.split(rng, chunk + 1)

    render_token = jax.vmap(init_one_jit, in_axes=[0, v_in_axes])(jp.asarray(keys), v_mjx_model)

    # Compile render step exactly like debug.py (token captured; vmap over data)
    def step(data_batch):
        def step_(d):
            _, rgb, depth = renderer.render(render_token, d)
            return d, rgb, depth
        return jax.vmap(step_)(data_batch)

    # Template data for building per-world data quickly
    template = mjx.make_data(mjx_model)

    # NOTE: we compile step on a "dummy" batch of data
    def _bcast_data(d0):
        return jax.tree_map(lambda x: jp.broadcast_to(x, (chunk,) + x.shape), d0)

    dummy = _bcast_data(template)
    step_compiled = jax.jit(step).lower(dummy).compile()

    # Attach shared memory
    shm_qpos_h, qpos_h = _create_shm_array(shm_qpos, create=False)
    shm_qvel_h, qvel_h = _create_shm_array(shm_qvel, create=False)
    shm_obs_h, obs_h = _create_shm_array(shm_obs, create=False)

    try:
        while not stop_evt.is_set():
            # Wait for request
            if not req_evt.wait(timeout=0.1):
                continue
            req_evt.clear()

            # Render full batch in chunks
            B = qpos_h.shape[0]
            H = render_width
            W = render_width  # debug.py uses width twice

            for i in range(0, B, chunk):
                qp = qpos_h[i:i+chunk]
                qv = qvel_h[i:i+chunk]

                # Build per-world data: template -> replace qpos/qvel -> forward
                # Create a fresh per-world data via vmap to stay close to debug semantics.
                def make_forward(qp1, qv1):
                    d = template.replace(qpos=jp.asarray(qp1), qvel=jp.asarray(qv1))
                    d = mjx.forward(mjx_model, d)
                    return d

                data_batch = jax.vmap(make_forward)(qp, qv)
                _dout, rgba, _depth = step_compiled(data_batch)

                # Immediately break from JAX: copy to CPU uint8
                rgba_cpu = np.asarray(jax.device_get(rgba))  # [C,2,H,W,4] uint8
                rgb = rgba_cpu[..., :3]
                left = rgb[:, 0]
                right = rgb[:, 1]
                stitched = np.concatenate([left, right], axis=2)  # [C,H,2W,3] uint8

                obs_h[i:i+chunk] = stitched

            resp_evt.set()

    finally:
        shm_qpos_h.close()
        shm_qvel_h.close()
        shm_obs_h.close()


class _RemoteRendererClient:
    """
    Process B side: writes qpos/qvel to shm, waits, reads obs from shm (CPU uint8).
    """
    def __init__(self, env, batch_size: int, chunk: int):
        # Ensure spawn (CUDA-safe)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        self.B = int(batch_size)
        self.chunk = int(chunk)

        # Pull render config from env
        vc = env._config.vision_config
        self.render_w = int(getattr(vc, "render_width", getattr(env, "render_width", 64)))
        self.render_h = int(getattr(vc, "render_height", getattr(env, "render_height", 64)))
        self.gpu_id = int(getattr(vc, "gpu_id", 0))
        self.enabled_geom_groups = np.array(getattr(vc, "enabled_geom_groups", [0, 1, 2]))
        self.add_cam_debug_geo = bool(getattr(vc, "add_cam_debug_geo", False))
        self.use_rasterizer = bool(getattr(vc, "use_rasterizer", False))

        # Shapes: we need nq/nv (available once env has mjx_model)
        nq = int(env.mjx_model.nq)
        nv = int(env.mjx_model.nv)

        # Shared memory specs
        self._shm_qpos = _ShmSpec(name=f"shm_qpos_{os.getpid()}", shape=(self.B, nq), dtype="float32")
        self._shm_qvel = _ShmSpec(name=f"shm_qvel_{os.getpid()}", shape=(self.B, nv), dtype="float32")
        self._shm_obs  = _ShmSpec(name=f"shm_obs_{os.getpid()}",  shape=(self.B, self.render_w, 2*self.render_w, 3), dtype="uint8")

        # Create SHM
        self._qpos_shm, self.qpos = _create_shm_array(self._shm_qpos, create=True)
        self._qvel_shm, self.qvel = _create_shm_array(self._shm_qvel, create=True)
        self._obs_shm,  self.obs  = _create_shm_array(self._shm_obs,  create=True)

        # Sync primitives
        self.req_evt = mp.Event()
        self.resp_evt = mp.Event()
        self.stop_evt = mp.Event()

        # Worker args: env reconstructed inside worker using module/class/config
        env_module = env.__class__.__module__
        env_class = env.__class__.__name__
        env_config = env._config  # ConfigDict should be picklable

        self.proc = mp.Process(
            target=_renderer_worker_main,
            args=(
                env_module, env_class, env_config,
                self.render_w, self.render_h, self.chunk,
                self.gpu_id, self.enabled_geom_groups,
                self.add_cam_debug_geo, self.use_rasterizer,
                self._shm_qpos, self._shm_qvel, self._shm_obs,
                self.req_evt, self.resp_evt, self.stop_evt
            ),
            daemon=True,
        )
        self.proc.start()

    def render(self, qpos_cpu: np.ndarray, qvel_cpu: np.ndarray) -> np.ndarray:
        # Write request buffers
        self.qpos[:] = qpos_cpu.astype(np.float32, copy=False)
        self.qvel[:] = qvel_cpu.astype(np.float32, copy=False)

        # Signal + wait
        self.resp_evt.clear()
        self.req_evt.set()
        self.resp_evt.wait()

        # Return a view (do NOT mutate it without copying)
        return self.obs

    def close(self):
        try:
            self.stop_evt.set()
            self.req_evt.set()
            if self.proc.is_alive():
                self.proc.join(timeout=2)
        finally:
            for shm in (self._qpos_shm, self._qvel_shm, self._obs_shm):
                try:
                    shm.close()
                except Exception:
                    pass
            # unlink only from creator process
            for shm in (self._qpos_shm, self._qvel_shm, self._obs_shm):
                try:
                    shm.unlink()
                except Exception:
                    pass


# ---------------------------
# Your wrapper (Process B)
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
        remote_render: bool = True,
        render_chunk: int = 16,   # start conservative; 16 matches debug default scale
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

        # Physics RNG per env
        key = jax.random.PRNGKey(int(seed))
        self.key_reset = jax.random.split(key, self.batch_size)

        # Batched physics (still in Process B)
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

        # Remote renderer (Process A)
        self.remote_render = bool(remote_render)
        self._remote = None
        self._render_chunk = int(render_chunk)

    def _ensure_remote(self):
        if self._remote is None:
            self._remote = _RemoteRendererClient(self._raw_env, self.batch_size, chunk=self._render_chunk)

    def reset(self) -> TensorDict:
        self.env_state = self.reset_fn(self.key_reset)

        if self.remote_render:
            self._ensure_remote()
            # Send qpos/qvel to renderer process
            qpos_cpu = np.asarray(jax.device_get(self.env_state.data.qpos))
            qvel_cpu = np.asarray(jax.device_get(self.env_state.data.qvel))
            obs_cpu = self._remote.render(qpos_cpu, qvel_cpu)  # CPU uint8 [B,H,2W,3]
            obs_t = torch.from_numpy(np.ascontiguousarray(obs_cpu))
            if self.device.type == "cuda":
                obs_t = obs_t.to(self.device, non_blocking=True)
        else:
            raise RuntimeError("remote_render=False no longer supported in this variant.")

        return TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)

    def step(self, action: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert self.env_state is not None, "Call reset() before step()."

        action = torch.clamp(action, -1.0, 1.0)
        action_jax = _torch_to_jax(action)

        self.env_state = self.step_fn(self.env_state, action_jax)

        self._ensure_remote()
        qpos_cpu = np.asarray(jax.device_get(self.env_state.data.qpos))
        qvel_cpu = np.asarray(jax.device_get(self.env_state.data.qvel))
        obs_cpu = self._remote.render(qpos_cpu, qvel_cpu)
        obs_t = torch.from_numpy(np.ascontiguousarray(obs_cpu))
        if self.device.type == "cuda":
            obs_t = obs_t.to(self.device, non_blocking=True)

        reward_t = torch.from_numpy(np.asarray(jax.device_get(self.env_state.reward))).reshape(-1).to(torch.float32)
        done_t = torch.from_numpy(np.asarray(jax.device_get(self.env_state.done))).reshape(-1).to(torch.bool)

        if self.device.type == "cuda":
            reward_t = reward_t.to(self.device, non_blocking=True)
            done_t = done_t.to(self.device, non_blocking=True)

        obs_td = TensorDict({"state": obs_t}, batch_size=[self.batch_size], device=self.device)
        info = {"truncation": obs_t.new_zeros((self.batch_size,), dtype=torch.bool)}

        if self.render_callback is not None:
            self.render_callback(None, self.env_state)

        return obs_td, reward_t, done_t, info

    def close(self):
        if self._remote is not None:
            self._remote.close()
            self._remote = None
