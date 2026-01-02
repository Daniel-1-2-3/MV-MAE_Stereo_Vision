# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrappers for MuJoCo Playground environments that interop with torch.

This version mirrors the *working* approach you showed:

- The environment's physics methods are **single-world**.
- The wrapper does the batching with `jax.vmap`.
- Only the **batched physics step** is JIT'd.
- Rendering happens from Python via `env.render_obs(...)` and is never traced into XLA.
"""

from collections import deque
import functools
import os
from typing import Any, Optional, Dict, Callable

import jax
import numpy as np

try:
    from rsl_rl.env import VecEnv  # pytype: disable=import-error
except ImportError:
    VecEnv = object

try:
    import torch  # pytype: disable=import-error
except ImportError:
    torch = None

try:
    from tensordict import TensorDict  # pytype: disable=import-error
except ImportError:
    TensorDict = None


def _jax_to_torch(tensor):
    """Zero-copy JAX->Torch via DLPack (device stays on GPU)."""
    import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel
    tensor = jax.block_until_ready(tensor)
    return tpack.from_dlpack(tensor)


def _torch_to_jax(tensor):
    """Zero-copy Torch->JAX via DLPack (device stays on GPU)."""
    from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel
    return from_dlpack(tensor)


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception as exc:
        raise ValueError("No runs in this directory: " + root) from exc

    if load_run == -1 or load_run == "-1":
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: m.zfill(15))
        model = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    load_path = os.path.join(load_run, model)
    return load_path


class RSLRLBraxWrapper(VecEnv):
    """Wrapper for MJX envs (JAX) that interop with torch/RSL-RL VecEnv.

    Expected env interface (single-world physics):
      - env.reset_physics(key[2]) -> State (single-world, small placeholder obs)
      - env.step_physics(State, action[act_dim]) -> State (single-world, no rendering)
      - env.render_obs(data_batched, info_batched) -> pixels [B,H,2W,3]
      - env._ensure_render_token(data_batched, debug) -> initializes renderer token (non-jit)

    This wrapper:
      - vmaps reset_physics / step_physics to create a batch of size B
      - jits only the vmapped step_physics (hot path)
      - renders outside jit
    """

    def __init__(
        self,
        env,
        num_actors,
        seed,
        episode_length,
        action_repeat,
        randomization_fn=None,
        render_callback=None,
        device_rank=None,
    ):
        if torch is None:
            raise ImportError("torch is required for RSLRLBraxWrapper")

        self.env = env
        self._raw_env = env
        self.render_callback = render_callback

        cfg = getattr(env, "_config", None) or getattr(env, "cfg", None)
        self.cfg = cfg if cfg is not None else {}

        self.seed = int(seed)
        self.batch_size = int(num_actors)
        self.num_envs = int(num_actors)

        # The Madrona renderer is constructed with a fixed num_worlds.
        # To mirror the working setup, ensure wrapper batch size matches env.render_batch_size.
        if hasattr(env, "render_batch_size") and int(getattr(env, "render_batch_size")) != self.batch_size:
            raise ValueError(
                f"num_actors (batch_size)={self.batch_size} must equal env.render_batch_size={int(getattr(env, 'render_batch_size'))} for Madrona."
            )

        self.key = jax.random.PRNGKey(self.seed)

        if device_rank is not None:
            gpu_devices = jax.devices("gpu")
            if not gpu_devices:
                raise RuntimeError("device_rank was set but no GPU devices are visible to JAX.")
            if device_rank < 0 or device_rank >= len(gpu_devices):
                raise ValueError(f"device_rank={device_rank} out of range for {len(gpu_devices)} GPUs.")
            self.key = jax.device_put(self.key, gpu_devices[device_rank])
            self.device = f"cuda:{device_rank}"
            print(f"Device -- {gpu_devices[device_rank]}")
            print(f"Key device -- {self.key.devices()}")
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        key_reset, key_randomization = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        if randomization_fn is not None:
            randomization_rng = jax.random.split(key_randomization, self.batch_size)
            self.v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
        else:
            self.v_randomization_fn = None

        # ---- Observation metadata ----
        H = int(getattr(env, "render_height", 0) or 0)
        W = int(getattr(env, "render_width", 0) or 0)
        C = 3
        if H > 0 and W > 0:
            self.num_obs = H * (2 * W) * C
            print(f"obs expected shape: [B, {H}, {2*W}, {C}]")
        else:
            self.num_obs = 0
            print("obs shape metadata unknown (env has no render_height/render_width).")

        self.num_actions = int(getattr(env, "action_size"))
        self.max_episode_length = int(episode_length)

        self.success_queue = deque(maxlen=100)

        # ---- Split physics and rendering ----
        if not hasattr(env, "step_physics") or not hasattr(env, "reset_physics") or not hasattr(env, "render_obs"):
            raise RuntimeError(
                "Env must provide reset_physics / step_physics / render_obs for split execution."
            )

        # Batched reset: vmap single-world reset over keys[B,2]
        def _reset_batched(keys):
            return jax.vmap(self.env.reset_physics)(keys)

        # Batched step: vmap single-world step over state/action
        def _step_batched(state, action):
            return jax.vmap(self.env.step_physics, in_axes=(0, 0))(state, action)

        self.reset_physics_fn = _reset_batched  # keep unjitted (called infrequently)
        self.step_physics_fn = jax.jit(_step_batched)  # hot path: JIT vmapped physics
        self.render_obs_fn = env.render_obs  # ALWAYS non-jit

        print("JITing batched physics step (render stays non-jit)")
        _ = self.step_physics_fn  # force attribute creation
        print("Done JITing batched physics step")

        # Store physics-only state (small obs).
        self.env_state = None

    def _ensure_renderer(self):
        # Initialize renderer token once, using current batched state.
        if hasattr(self.env, "_ensure_render_token") and self.env_state is not None:
            debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"
            self.env._ensure_render_token(self.env_state.data, debug)  # pylint: disable=protected-access

    def _render_pixels(self):
        # Non-jit render. Pixels are computed lazily; sync happens in _jax_to_torch.
        self._ensure_renderer()
        return self.render_obs_fn(self.env_state.data, self.env_state.info)

    def step(self, action):
        if torch is None:
            raise ImportError("torch is required")

        action = torch.clip(action, -1.0, 1.0)
        action = _torch_to_jax(action)

        # JIT'd batched physics
        self.env_state = self.step_physics_fn(self.env_state, action)

        # Non-jit render (smoke-test style)
        pixels = self._render_pixels()

        obs_t = _jax_to_torch(pixels)
        obs = {"state": obs_t}

        reward = _jax_to_torch(self.env_state.reward)
        done = _jax_to_torch(self.env_state.done)

        info = self.env_state.info
        trunc = info.get("truncation", None)
        if trunc is None:
            trunc = jax.numpy.zeros_like(self.env_state.done)
        truncation = _jax_to_torch(trunc)

        info_ret = {
            "time_outs": truncation,
            "observations": {"critic": None},
            "log": {},
        }

        last_episode_success_count = []
        if "last_episode_success_count" in info:
            lec = _jax_to_torch(info["last_episode_success_count"])
            try:
                lec_done = lec[done > 0].float().tolist()
            except Exception:
                lec_done = []
            last_episode_success_count = lec_done

        if len(last_episode_success_count) > 0:
            self.success_queue.extend(last_episode_success_count)

        info_ret["log"]["last_episode_success_count"] = (
            float(np.mean(self.success_queue)) if len(self.success_queue) > 0 else 0.0
        )

        for k, v in self.env_state.metrics.items():
            if k not in info_ret["log"]:
                info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

        if TensorDict is None:
            return obs, reward, done, info_ret

        obs_td = TensorDict(obs, batch_size=[self.num_envs])
        return obs_td, reward, done, info_ret

    def reset(self):
        self.key, key_reset = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        self.env_state = self.reset_physics_fn(self.key_reset)

        pixels = self._render_pixels()
        obs_t = _jax_to_torch(pixels)
        obs = {"state": obs_t}

        if TensorDict is None:
            return obs

        return TensorDict(obs, batch_size=[self.num_envs])

    def get_observations(self):
        return self.reset()