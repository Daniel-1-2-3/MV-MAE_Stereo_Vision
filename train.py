"""Train MV-MAE + DrQ-v2 on needle lifting from soft tissue, from stereo images only.

    python train.py                                    # defaults from configs/needle_tissue.yaml
    python train.py train.demo_path=demos/needle_demos.pt
    python train.py env.num_envs=16 train.replay_capacity=60000 log.run_name=small
    python train.py log.wandb_mode=offline             # no internet: sync later with `wandb sync`

Any config value can be overridden as section.key=value (see config.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/needle_tissue.yaml")
simulation_app, args, overrides = launch(parser)

import torch  # noqa: E402

from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from trainer import Trainer  # noqa: E402


def main() -> None:
    cfg = load_config(args.config, overrides)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Load the demos before building the simulator, so a wrong path fails in seconds.
    demo = torch.load(cfg.train.demo_path, map_location="cpu") if cfg.train.demo_path else None
    env = make_env(cfg, device=args.device)
    if demo is not None:
        print(f"loaded {demo['meta']['num_episodes']} demo episodes from {cfg.train.demo_path}")
        for section in ("env", "camera"):
            if demo["meta"].get(section) != cfg.to_dict()[section]:
                print(f"WARNING: demos were recorded with a different '{section}' config than this run")
    trainer = Trainer(cfg, env, device=args.device, demo_data=demo)
    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Print the error and exit at once: Isaac Sim's shutdown can hang after an error,
        # which would leave the process running with the error never printed.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
