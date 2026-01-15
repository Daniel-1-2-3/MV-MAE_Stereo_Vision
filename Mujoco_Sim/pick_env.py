from collections import deque
import functools
import os
from typing import Any

import jax
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

try:
    from rsl_rl.env import VecEnv
except ImportError:
    VecEnv = object


def _jax_to_torch(tensor):
    import torch.utils.dlpack as tpack
    return tpack.from_dlpack(tensor)


def _torch_to_jax(tensor):
    from jax.dlpack import from_dlpack
    return from_dlpack(tensor)


class RSLRLBraxWrapper(VecEnv):
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
            raise ImportError("torch is required")

        self.env = env
        self._raw_env = env
        self.render_callback = render_callback

        self.seed = int(seed)
        self.batch_size = int(num_actors)
        self.num_envs = int(num_actors)

        self.key = jax.random.PRNGKey(self.seed)

        if device_rank is not None:
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("No GPU devices visible to JAX.")
            self.key = jax.device_put(self.key, gpus[device_rank])
            self.device = f"cuda:{device_rank}"
            print(f"Device -- {gpus[device_rank]}")
            print(f"Key device -- {self.key.devices()}")
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # REQUIRE the split API so render stays out of jit
        if not (hasattr(env, "reset_physics") and hasattr(env, "step_physics") and hasattr(env, "compute_obs")):
            raise AttributeError("Env must implement reset_physics/step_physics/compute_obs.")

        self.reset_fn = env.reset_physics
        self.step_fn = jax.jit(env.step_physics)

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

        self.env_state = self.step_fn(self.env_state, action)
        self._render_outside_jit()

        obs_t = _jax_to_torch(self.env_state.obs)
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

        for k, v in self.env_state.metrics.items():
            if k not in info_ret["log"]:
                info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

        obs_td = TensorDict(obs, batch_size=[self.num_envs])
        return obs_td, reward, done, info_ret

    def reset(self):
        self.key, key_reset = jax.random.split(self.key)
        keys = jax.random.split(key_reset, self.batch_size)

        self.env_state = self.reset_fn(keys)
        self._render_outside_jit()

        obs_t = _jax_to_torch(self.env_state.obs)
        obs = {"state": obs_t}
        return TensorDict(obs, batch_size=[self.num_envs])

    def get_observations(self):
        return self.reset()
