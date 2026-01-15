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

Important:
- step_physics is jitted (fast)
- rendering is NEVER jitted
- we "warm up" JAX kernels BEFORE initializing Madrona renderer to avoid
  post-Madrona first-time compilation (your current failure mode).
"""

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
    import torch.utils.dlpack as tpack  # pylint: disable=import-outside-toplevel
    return tpack.from_dlpack(tensor)


def _torch_to_jax(tensor):
    from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel
    return from_dlpack(tensor)


def _arr_device(x):
    try:
        return x.device()
    except TypeError:
        return x.device
    except Exception:
        try:
            return next(iter(x.devices()))
        except Exception:
            return None


class RSLRLBraxWrapper(VecEnv):
    """MJX env (JAX) interop with torch/RSL-RL VecEnv."""

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
            print(f"Key device -- {self.key.devices() if hasattr(self.key, 'devices') else self.key.device}")
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # per-env keys
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
            self.num_obs = H * (2 * W) * C
            print(f"obs (pixels) expected shape: [B, {H}, {2*W}, {C}]")
        else:
            self.num_obs = 0
            print("obs shape metadata unknown (env has no render_height/render_width).")

        self.num_actions = int(getattr(env, "action_size"))
        self.max_episode_length = int(episode_length)
        self.success_queue = deque(maxlen=100)

        # Functions:
        # - reset_physics is NOT jitted
        # - step_physics IS jitted
        self.reset_physics_fn = getattr(env, "reset_physics", None)
        self.step_physics_fn = getattr(env, "step_physics", None)
        if self.reset_physics_fn is None or self.step_physics_fn is None:
            raise AttributeError("Env must implement reset_physics and step_physics for this wrapper.")

        print("JITing step_physics")
        self.step_physics_fn = jax.jit(self.step_physics_fn, donate_argnums=(0,))
        print("Done JITing step_physics")

        self.env_state = None

        # Warmup JAX kernels BEFORE touching Madrona renderer
        self._warmup_jax_then_init_renderer()

    def _warmup_jax_then_init_renderer(self):
        debug = os.environ.get("PICK_ENV_DEBUG", "0") == "1"

        # 1) Warmup reset_physics (compiles common ops used in reset path)
        st = self.reset_physics_fn(self.key_reset)
        # Force sync on something simple from the state
        jax.block_until_ready(st.data.qpos)

        # 2) Warmup step_physics (compiles the big step kernel)
        zero_act = jnp.zeros((self.batch_size, self.num_actions), dtype=jnp.float32)
        st2 = self.step_physics_fn(st, zero_act)
        jax.block_until_ready(st2.data.qpos)

        # 3) Now init renderer exactly once (eager), and store token into env + state
        tok = self.env.init_renderer_once(self.key_reset, debug=debug)
        jax.block_until_ready(tok)

        # Update the state we’ll keep, so first reset() doesn’t need to redo anything
        info = dict(st2.info)
        info["render_token"] = tok
        self.env_state = st2.replace(info=info)

        if debug:
            # Smoke render once here (still eager)
            pix = self.env.render_pixels(tok, self.env_state.data)
            pix = self.env.postprocess_pixels(pix, self.env_state.info)
            jax.block_until_ready(pix)
            print("[diag] wrapper warmup render OK:", tuple(pix.shape), pix.dtype)

    def step(self, action):
        if torch is None:
            raise ImportError("torch is required")

        # action: torch [B, A]
        action = torch.clip(action, -1.0, 1.0)
        action_j = _torch_to_jax(action).astype(jnp.float32)

        # JIT fast path: physics only
        self.env_state = self.step_physics_fn(self.env_state, action_j)

        # EAGER: render pixels and overwrite obs
        tok = self.env_state.info.get("render_token", None)
        if tok is None:
            tok = self.env.init_renderer_once(self.key_reset, debug=False)
            info = dict(self.env_state.info)
            info["render_token"] = tok
            self.env_state = self.env_state.replace(info=info)

        pixels = self.env.render_pixels(tok, self.env_state.data)
        pixels = self.env.postprocess_pixels(pixels, self.env_state.info)
        obs_t = _jax_to_torch(pixels)
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

        if TensorDict is None:
            return obs, reward, done, info_ret

        obs_td = TensorDict(obs, batch_size=[self.num_envs])
        return obs_td, reward, done, info_ret

    def reset(self):
        # new keys
        self.key, key_reset = jax.random.split(self.key)
        self.key_reset = jax.random.split(key_reset, self.batch_size)

        # reset physics (no render)
        self.env_state = self.reset_physics_fn(self.key_reset)

        # ensure renderer token is initialized (should already be from warmup)
        tok = self.env_state.info.get("render_token", None)
        if tok is None:
            tok = self.env.init_renderer_once(self.key_reset, debug=False)
            info = dict(self.env_state.info)
            info["render_token"] = tok
            self.env_state = self.env_state.replace(info=info)

        # render eagerly
        pixels = self.env.render_pixels(tok, self.env_state.data)
        pixels = self.env.postprocess_pixels(pixels, self.env_state.info)
        obs_t = _jax_to_torch(pixels)
        obs = {"state": obs_t}

        if TensorDict is None:
            return obs
        return TensorDict(obs, batch_size=[self.num_envs])

    def get_observations(self):
        return self.reset()

    def render(self, mode="human"):
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
