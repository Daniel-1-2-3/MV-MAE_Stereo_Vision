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
"""Train a PPO agent using JAX on the specified environment."""

import datetime
import functools
import json
import os
import time
from typing import Any, NamedTuple
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import Custom_Mujoco_Playground
from Custom_Mujoco_Playground import registry
from Custom_Mujoco_Playground import wrapper
from Custom_Mujoco_Playground.config import dm_control_suite_params
from Custom_Mujoco_Playground.config import locomotion_params
from Custom_Mujoco_Playground.config import manipulation_params
import tensorboardX
import wandb
import argparse
import torch
import DrQv2_Architecture.utils as utils
from DrQv2_Architecture.logger import Logger
from DrQv2_Architecture.replay_buffer import ReplayBufferStorage, make_replay_loader
from DrQv2_Architecture.video import VideoRecorder
from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.env_wrappers import ExtendedTimeStepWrapper, ActionRepeatWrapper, FrameStackWrapper
from Mujoco_Sim.mujoco_pick_env import StereoPickCube
from gymnasium.spaces import Box

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax") # Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax") # Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=UserWarning, module="absl") # Suppress UserWarnings from absl (used by JAX and TensorFlow)

import argparse
from argparse import BooleanOptionalAction

"""
The logic for making the loop JIT-able is to split the entire pipeline into train 
and eval chunks. We run n timesteps each training chunk, run an eval chunk, then
run another n timesteps for the next training chunk. total_timesteps = num_chunks x steps per chunk

For a training loop to be JIT able, there cannot be any python elements, like for loops, ifs, numpy, etc.
Everything must b written in jax code. 
"""

def make_agent(self):
        # DrQv2 agent takes action_shape as (A, ) tuple
        return DrQV2Agent(
            action_shape = (self.action_space.shape[0], ),
            device = self.device,
            lr = self.lr,

            nviews = self.nviews,
            mvmae_patch_size = self.mvmae_patch_size,
            mvmae_encoder_embed_dim = self.mvmae_encoder_embed_dim,
            mvmae_decoder_embed_dim = self.mvmae_decoder_embed_dim,
            mvmae_encoder_heads = self.mvmae_encoder_heads,
            mvmae_decoder_heads = self.mvmae_decoder_heads,
            in_channels = self.in_channels,
            img_h_size = self.img_h_size,
            img_w_size = self.img_w_size,
            masking_ratio = self.masking_ratio,
            coef_mvmae = self.coef_mvmae,
            
            feature_dim = self.feature_dim,
            hidden_dim = self.hidden_dim,

            critic_target_tau = self.critic_target_tau,
            num_expl_steps = self.num_expl_steps,
            update_every_steps = self.update_every_steps,
            update_mvmae_every_steps = self.update_mvmae_every_steps,
            stddev_schedule = self.stddev_schedule,
            stddev_clip = self.stddev_clip,
            use_tb = self.use_tb
        )
class TrainCarry(NamedTuple): # Keeping track of variables between each training step
    key: jax.Array
    step: jp.Array             # global step counter
    env_state: Any              # JAX pytree
    agent_state: Any            # JAX pytree (params/opt states/etc)
    buffer_state: Any           # JAX pytree
    mvmae_counter: jp.Array   # optional separate counter (can just use step)

agent = 

def train(carry: TrainCarry, *, env, steps_per_chunk: int, learning_starts: int, update_mvmae_every: int) -> TrainCarry:
    def single_step(carry: TrainCarry, _):
        key, step, env_state, agent_state, buffer_state, mvmae_counter = carry
        key, episode_key = jax.random.split(key)
        with torch.no_grad():
            action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)
        
        
        
    
       