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

Key behavior (by design):
- MJX physics + reward are jitted (fast).
- Madrona raytracing is **NOT** jitted (avoids XLA custom-call backend-config issues).
  Rendering runs immediately after each jitted step/reset, and pixels are written into
  env_state.obs.
"""

from __future__ import annotations

from collections import deque
import functools
import os
from typing import Any, Optional

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

from Custom_Mujoco_Playground._src import wrapper
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

  return os.path.join(load_run, model)


class RSLRLBraxWrapper(VecEnv):
  """Wrapper for Brax/MJX environments that interop with torch."""

  def __init__(
      self,
      env,
      num_actors: int,
      seed: int,
      episode_length: int,
      action_repeat: int,
      randomization_fn=None,
      render_callback=None,
      device_rank: Optional[int] = None,
  ):
    import torch  # pytype: disable=import-error # pylint: disable=redefined-outer-name,unused-import,import-outside-toplevel

    cfg = getattr(env, "_config", None) or getattr(env, "cfg", None) or getattr(self.env, "cfg", None)
    self.cfg = cfg if cfg is not None else {}

    if not hasattr(self, "device"):
      self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    self.seed = int(seed)
    self.batch_size = int(num_actors)
    self.num_envs = int(num_actors)

    # Create key BEFORE any device_put
    self.key = jax.random.PRNGKey(self.seed)

    if device_rank is not None:
      gpu_devices = jax.devices("gpu")
      if device_rank < 0 or device_rank >= len(gpu_devices):
        raise ValueError(f"device_rank={device_rank} but only {len(gpu_devices)} GPU devices visible to JAX")
      self.key = jax.device_put(self.key, gpu_devices[device_rank])
      self.device = f"cuda:{device_rank}"
      print(f"Device -- {gpu_devices[device_rank]}")
      print(f"Key device -- {self.key.devices()}")

    # split key into two for reset and randomization
    key_reset, key_randomization = jax.random.split(self.key)
    self.key_reset = jax.random.split(key_reset, self.batch_size)

    if randomization_fn is not None:
      randomization_rng = jax.random.split(key_randomization, self.batch_size)
      v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
    else:
      v_randomization_fn = None

    # Wrap env for brax training (autoreset, episode length, etc.)
    self.env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    # Raw env access (matches your previous assumption)
    raw = self.env.env.env.unwrapped
    self._raw_env = raw

    # IMPORTANT: renderer calls must NOT be jitted.
    # The raw env must expose init_render(data) and render_pixels(token, data).
    if not hasattr(raw, "init_render") or not hasattr(raw, "render_pixels"):
      raise AttributeError(
          "Raw env must implement init_render(data_batched) and render_pixels(render_token, data_batched). "
          "See pick_env_rewritten.py."
      )
    self._render_init_fn = raw.init_render          # NOT jitted
    self._render_pixels_fn = raw.render_pixels      # NOT jitted
    self._render_token = None

    self.render_callback = render_callback

    self.asymmetric_obs = False
    obs_shape = self.env.env.unwrapped.observation_size
    print(f"obs_shape: {obs_shape}")

    if isinstance(obs_shape, dict):
      print("Asymmetric observation space")
      self.asymmetric_obs = True
      self.num_obs = obs_shape["state"]
      self.num_privileged_obs = obs_shape["privileged_state"]
    else:
      self.num_obs = obs_shape
      self.num_privileged_obs = None

    self.num_actions = self.env.env.unwrapped.action_size
    self.max_episode_length = int(episode_length)

    self.success_queue = deque(maxlen=100)

    # JIT reset/step: safe because env.reset/env.step no longer call renderer.
    print("JITing reset and step (physics only)")
    self.reset_fn = jax.jit(self.env.reset)
    self.step_fn = jax.jit(self.env.step)
    print("Done JITing reset and step")

    self.env_state = None

  def _ensure_render_token(self):
    """Initialize render token if not set."""
    if self._render_token is None:
      self._render_token = self._render_init_fn(self.env_state.data)

  def step(self, action):
    import torch  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

    action = torch.clip(action, -1.0, 1.0)  # pytype: disable=attribute-error
    action = _torch_to_jax(action)

    # Fast jitted physics step
    self.env_state = self.step_fn(self.env_state, action)

    # Raytracing OUTSIDE jit
    self._ensure_render_token()
    pixels = self._render_pixels_fn(self._render_token, self.env_state.data)  # uint8 [B,H,2W,3]
    self.env_state = self.env_state.replace(obs=pixels)

    critic_obs = None
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}

    reward = _jax_to_torch(self.env_state.reward)
    done = _jax_to_torch(self.env_state.done)
    info = self.env_state.info
    truncation = _jax_to_torch(info["truncation"])

    info_ret = {
        "time_outs": truncation,
        "observations": {"critic": critic_obs},
        "log": {},
    }

    if "last_episode_success_count" in info:
      last_episode_success_count = (
          _jax_to_torch(info["last_episode_success_count"])[done > 0]
          .float()
          .tolist()
      )
      if len(last_episode_success_count) > 0:
        self.success_queue.extend(last_episode_success_count)
      info_ret["log"]["last_episode_success_count"] = np.mean(self.success_queue)

    for k, v in self.env_state.metrics.items():
      if k not in info_ret["log"]:
        info_ret["log"][k] = _jax_to_torch(v).float().mean().item()

    obs = TensorDict(obs, batch_size=[self.num_envs])
    return obs, reward, done, info_ret

  def reset(self):
    # Jitted reset (physics only)
    self.env_state = self.reset_fn(self.key_reset)

    # New episode => re-init token OUTSIDE jit
    self._render_token = self._render_init_fn(self.env_state.data)

    pixels = self._render_pixels_fn(self._render_token, self.env_state.data)
    self.env_state = self.env_state.replace(obs=pixels)

    if self.asymmetric_obs:
      obs = _jax_to_torch(self.env_state.obs["state"])
      critic_obs = _jax_to_torch(self.env_state.obs["privileged_state"])
      obs = {"state": obs, "privileged_state": critic_obs}
    else:
      obs = _jax_to_torch(self.env_state.obs)
      obs = {"state": obs}

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
