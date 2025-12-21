from pathlib import Path
from train_drqv2_mujoco import Workshop, get_args, save_agent

def main():
    print("Root path:", Path.cwd())
    args = get_args()
    workspace = Workshop(**vars(args))
    workspace.train()
    save_agent(workspace.agent)