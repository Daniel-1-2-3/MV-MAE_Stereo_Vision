from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda.pick import PandaPickCube
from Custom_Mujoco_Playground._src.manipulation.franka_emika_panda.pick import default_config
from ml_collections import config_dict
from typing import Any, Dict, Optional, Union
from pathlib import Path
from mujoco_playground._src.manipulation.franka_emika_panda import panda
import os
import time
import torch
import mujoco
from MAE_Model.prepare_input import Prepare
from gymnasium.spaces import Box
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math
from Custom_Mujoco_Playground._src import mjx_env

from madrona_mjx.renderer import BatchRenderer # type: ignore
_GLOBAL_MADRONA_RENDERER = None
_GLOBAL_MJX_MODEL = None

def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=1024,
      render_width=64,
      render_height=64,
      use_rasterizer=False,
      enabled_geom_groups=[0, 1, 2],
  )

def default_config():
  config = config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.005,
      episode_length=200,
      action_repeat=1,
      discount=0.99,
      img_h_size=64,
      img_w_wise=64,
      # Size of cartesian increment.
      action_scale=0.005,
      reward_config=config_dict.create(
          reward_scales=config_dict.create(
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Do not collide cube with gripper
              no_box_collision=0.05,
              # Destabilizes training in cartesian action space.
              robot_target_qpos=0.0,
          ),
          action_rate=-0.0005,
          no_soln_reward=-0.01,
          lifted_reward=0.5,
          success_reward=2.0,
      ),
      vision=True,
      vision_config=default_vision_config(),
      obs_noise=config_dict.create(brightness=[1.0, 1.0]),
      box_init_range=0.05,
      success_threshold=0.05,
      action_history_length=1,
      impl='jax',
      nconmax=12 * 1024,
      njmax=128,
  )
  return config

class StereoPickCube(PandaPickCube):
    def __init__(self, 
        config=default_config(), 
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        mjx_env.MjxEnv.__init__(config, config_overrides)
        self._vision = config.vision
        
        xml_path = (
            Path(os.getcwd())
            / "Mujoco_Sim"
            / "franka_emika_panda"
            / "mjx_single_cube_stereo.xml"
        )

        self._xml_path = xml_path.as_posix()
        self._model_assets = panda.get_assets()
        
        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(
                xml_path.read_text(), assets=self._model_assets
            )
        )
        mj_model.opt.timestep = config.sim_dt
        
        
            
        