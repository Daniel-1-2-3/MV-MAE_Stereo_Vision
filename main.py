import jax
if not hasattr(jax.sharding, "AxisType"):
    class AxisType(str):
        pass
    jax.sharding.AxisType = AxisType

from pathlib import Path
from absl import app
from train_drqv2_mujoco import Workshop, get_args, save_agent
from Custom_Mujoco_Playground.learning.train_jax_ppo import main

def main():
    args = get_args()
    app.run(main)