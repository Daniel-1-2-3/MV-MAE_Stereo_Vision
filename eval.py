"""Evaluate a trained checkpoint and save stereo videos.

    python eval.py --checkpoint runs/<run>/ckpt_final.pt --rounds 2

Each round runs one deterministic episode in every parallel environment. The
config stored in the checkpoint is used (overrides still apply, e.g. env.num_envs=8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--rounds", type=int, default=1)
parser.add_argument("--out", default="outputs/eval")
simulation_app, args, overrides = launch(parser)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vis  # noqa: E402
from agent.drqv2 import MVMAEDrQV2Agent  # noqa: E402
from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from trainer import evaluate_policy  # noqa: E402


def main() -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(overrides=overrides, base=ckpt["config"])
    env = make_env(cfg, device=args.device)
    h, w = env.obs_shape[-2:]
    agent = MVMAEDrQV2Agent(cfg.agent, cfg.mvmae, (h, w), env.obs_shape[0], env.action_dim, args.device)
    agent.load_state_dict(ckpt["agent"], load_optimizers=False)
    all_metrics = []
    for r in range(args.rounds):
        metrics, frames, fail_frames = evaluate_policy(env, agent, ckpt["env_steps"], record_video=True,
                                                       video_scale=cfg.log.video_scale)
        vis.save_mp4(frames, out / f"episode_{r}.mp4", fps=1.0 / env.step_dt)
        vis.save_mp4(fail_frames, out / f"failure_{r}.mp4", fps=1.0 / env.step_dt)
        all_metrics.append(metrics)
        print(f"round {r}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()), flush=True)
    summary = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("mean over rounds: " + ", ".join(f"{k}={v:.3f}" for k, v in summary.items()))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
