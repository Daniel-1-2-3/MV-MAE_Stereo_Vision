from pathlib import Path
from absl import app
from train_drqv2_mujoco import Workshop, get_args, save_agent
from Custom_Mujoco_Playground.learning.train_jax_ppo import main

def main():
    app.run(main)