"""Run the robustness suite on a saved checkpoint, with more episodes than during training.

    python robustness_eval.py --checkpoint runs/<run>/ckpt_best.pt --episodes 256
    python robustness_eval.py --checkpoint runs/<run>/ckpt_best.pt --conditions gaussian_noise blur --episodes 64
    python robustness_eval.py --checkpoint runs/<mvmae run>/ckpt_best.pt --recon   # + MV-MAE "reconstruct, then act"

Same conditions, seeds and numbers as the suite that runs during training (robust.* settings, see robustness.py),
with --episodes episodes per condition (rounds of one deterministic episode in every parallel env; every condition
sees the same needle placements). The config stored in the checkpoint is used; overrides still apply (e.g.
env.num_envs=32). Writes to --out (default outputs/robustness/<run>_<checkpoint>/): robust.json with every number
(per-episode results included), robust_curves.png/.pdf (success vs severity per family, nominal at 0),
robust_bars.png/.pdf (every condition) and videos of the --video_conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=256, help="episodes per condition")
parser.add_argument("--conditions", nargs="*", default=None, help="condition or family names (default: robust.conditions)")
parser.add_argument("--video_conditions", nargs="*", default=None, help="default: robust.video_conditions")
parser.add_argument("--recon", action="store_true",
                    help="also run the MV-MAE 'reconstruct, then act' conditions (robustness.RECON_FAMILIES; skipped "
                         "for agents without MV-MAE, left out of mean_success)")
parser.add_argument("--label", default=None, help="name in the figures (default: agent.encoder)")
parser.add_argument("--out", default=None)
simulation_app, args, overrides = launch(parser)

import torch  # noqa: E402

import robustness  # noqa: E402
from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from trainer import build_agent  # noqa: E402


def main() -> None:
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = load_config(overrides=overrides, base=ckpt["config"])
    out = Path(args.out) if args.out else Path("outputs/robustness") / f"{ckpt_path.parent.name}_{ckpt_path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    names = list(args.conditions if args.conditions is not None else cfg.robust.conditions)
    if args.recon:
        names += list(robustness.RECON_FAMILIES)
    conditions = robustness.resolve_conditions(names)
    videos = [c.name for c in robustness.resolve_conditions(
        args.video_conditions if args.video_conditions is not None else cfg.robust.video_conditions)]
    env = make_env(cfg, device=args.device)
    agent = build_agent(cfg, env, args.device)
    agent.load_state_dict(ckpt["agent"], load_optimizers=False)
    print(f"checkpoint {ckpt_path} ({ckpt['env_steps']} env steps, agent.encoder={cfg.agent.encoder}): "
          f"{len(conditions)} conditions x {args.episodes} episodes on {env.num_envs} envs", flush=True)
    res = robustness.run_suite(env, agent, ckpt["env_steps"], cfg.robust, conditions, videos, args.episodes,
                               video_dir=out / "videos", video_scale=3)
    res["checkpoint"] = str(ckpt_path)
    res["encoder"] = cfg.agent.encoder
    print("\n".join(robustness.summary_lines(res)))
    print(f"saved {robustness.save_json(res, out / 'robust.json')}")
    for f in robustness.save_figures({args.label or cfg.agent.encoder: res}, out, "robust"):
        print(f"saved {f}")
    env.close()


if __name__ == "__main__":
    import os
    import sys
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
