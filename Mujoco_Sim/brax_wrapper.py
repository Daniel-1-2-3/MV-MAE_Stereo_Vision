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
"""Wrappers for MuJoCo Playground environments that interop with torch."""

from collections import deque
import functools
import os
from typing import Any, Optional, Dict

import jax
import jax.numpy as jnp
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
    import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel
    return tpack.from_dlpack(tensor)


def _torch_to_jax(tensor):
    from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel
    return from_dlpack(tensor)


class RSLRLBraxWrapper(VecEnv):
    """MJX env wrapper that keeps Madrona rendering OUTSIDE jit."""

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

        self.seed = int(seed)
        self.batch_size = int(num_actors)
        self.num_envs = int(num_actors)

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

        # require the split APIs
        if not hasattr(env, "reset_physics") or not hasattr(env, "step_physics") or not hasattr(env, "compute_obs"):
            raise AttributeError(
                "Env must implement reset_physics(), step_physics(), and compute_obs() "
                "so rendering can stay outside jax.jit."
            )

        # Reset stays non-jit, step_physics is jitted
        self.reset_fn = env.reset_physics
        self.step_fn = jax.jit(env.step_physics)

        # split key into per-env keys
        key_reset, _ = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        H = int(getattr(env, "render_height", 0) or 0)
        W = int(getattr(env, "render_width", 0) or 0)
        if H > 0 and W > 0:
            print(f"obs (env_state.obs) expected shape: [B, {H}, {2*W}, 3]")

        self.num_actions = int(getattr(env, "action_size"))
        self.max_episode_length = int(episode_length)
        self.success_queue = deque(maxlen=100)

        self.env_state = None

    def _render_outside_jit(self):
        obs = self.env.compute_obs(self.env_state.data, self.env_state.info)
        self.env_state = self.env_state.replace(obs=obs)

    def step(self, action):
        action = torch.clip(action, -1.0, 1.0).to(dtype=torch.float32)
        action = _torch_to_jax(action)

        # physics (jitted)
        self.env_state = self.step_fn(self.env_state, action)

        # render (NOT jitted) -> removes backend_config warnings
        self._render_outside_jit()

        obs_t = _jax_to_torch(self.env_state.obs)
        obs = {"state": obs_t}

        reward = _jax_to_torch(self.env_state.reward)
        done = _jax_to_torch(self.env_state.done)

        info = self.env_state.info
        trunc = info.get("truncation", None)
        if trunc is None:
            trunc = jnp.zeros_like(self.env_state.done)
        truncation = _jax_to_torch(trunc)

        info_ret = {
            "time_outs": truncation,
            "observations": {"critic": None},
            "log": {},
        }

        for k, v in self.env_state.metrics.items():
            if k not in info_ret["log"]:
                info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

        obs_td = TensorDict(obs, batch_size=[self.num_envs])
        return obs_td, reward, done, info_ret

    def reset(self):
        self.key, key_reset = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        self.env_state = self.reset_fn(self.key_reset)
        self._render_outside_jit()

        obs_t = _jax_to_torch(self.env_state.obs)
        obs = {"state": obs_t}
        return TensorDict(obs, batch_size=[self.num_envs])

    def get_observations(self):
        return self.reset()
