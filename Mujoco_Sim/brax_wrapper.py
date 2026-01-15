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
    """Zero-copy JAX->Torch via DLPack (device stays on GPU)."""
    import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel
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

    Key rule (to match micro-repro "no warning" pattern):
      - JIT ONLY the physics step.
      - DO NOT call Madrona renderer.render() inside a jitted function.
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

        # split key into two for reset and randomization
        key_reset, key_randomization = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        if randomization_fn is not None:
            randomization_rng = jax.random.split(key_randomization, self.batch_size)
            self.v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
        else:
            self.v_randomization_fn = None

        H = int(getattr(env, "render_height", 0) or 0)
        W = int(getattr(env, "render_width", 0) or 0)
        C = 3
        if H > 0 and W > 0:
            self.num_obs = H * (2 * W) * C  # metadata only
            print(f"obs (env_state.obs) expected shape: [B, {H}, {2*W}, {C}]")
        else:
            self.num_obs = 0
            print("obs shape metadata unknown (env has no render_height/render_width).")

        self.num_actions = int(getattr(env, "action_size"))
        self.max_episode_length = int(episode_length)
        self.success_queue = deque(maxlen=100)

        # IMPORTANT: use physics-only API
        if not hasattr(env, "reset_physics") or not hasattr(env, "step_physics") or not hasattr(env, "compute_obs"):
            raise AttributeError(
                "Env must provide reset_physics(), step_physics(), and compute_obs() "
                "so rendering can remain outside jit."
            )

        print("JITing step_physics (render stays outside jit)")
        self.reset_fn = env.reset_physics
        self.step_fn = jax.jit(env.step_physics)
        print("Done JITing step_physics")

        self.env_state = None

    def _fill_obs_outside_jit(self):
        """Compute pixels via Madrona outside jit (matches micro-repro behavior)."""
        obs = self.env.compute_obs(self.env_state.data, self.env_state.info)
        self.env_state = self.env_state.replace(obs=obs)

    def step(self, action):
        if torch is None:
            raise ImportError("torch is required")

        # action: torch tensor [B, num_actions]
        action = torch.clip(action, -1.0, 1.0).to(dtype=torch.float32)
        action = _torch_to_jax(action)

        # Physics step (jitted)
        self.env_state = self.step_fn(self.env_state, action)

        # Render outside jit (no XLA backend_config warnings)
        self._fill_obs_outside_jit()

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

        obs_td = TensorDict(obs, batch_size=[self.num_envs])
        return obs_td, reward, done, info_ret

    def reset(self):
        # per-env keys each reset
        self.key, key_reset = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        # Physics reset (not jitted)
        self.env_state = self.reset_fn(self.key_reset)

        # Render outside jit
        self._fill_obs_outside_jit()

        obs_t = _jax_to_torch(self.env_state.obs)
        obs = {"state": obs_t}

        if TensorDict is None:
            return obs
        return TensorDict(obs, batch_size=[self.num_envs])

    def get_observations(self):
        return self.reset()

    def render(self, mode="human"):  # pylint: disable=unused-argument
        if self.render_callback is not None:
            self.render_callback(self.env.env.env, self.env_state)
        else:
            raise ValueError("No render callback specified")

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info["action_space"] = self.action_space  # pytype: disable=attribute-error
        info["observation_space"] = self.observation_space  # pytype: disable=attribute-error
        return info
