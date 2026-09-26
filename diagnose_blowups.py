"""Find out what the physics glitches ("blowups") are.

Runs the environment with a policy and, every time an episode is cut by one of
the blowup_* checks, records the state just before the reset: which check fired,
how fast the needle / joints were moving, where the tool tip and needle were
relative to the tissue, whether the gripper was closed, how deep the tissue was
dented. Prints a summary, writes every record to blowups.json and saves short
stereo clips (the frames leading up to a glitch).

    # the trained policy, with its training-time exploration noise (most like training):
    python diagnose_blowups.py --checkpoint runs/<run>/ckpt_latest.pt env.num_envs=8
    # the scripted demo controller plus action noise (no checkpoint needed):
    python diagnose_blowups.py --noise 0.3 env.num_envs=8
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/needle_tissue.yaml", help="used when no --checkpoint is given")
parser.add_argument("--checkpoint", default="", help="policy to run; empty = scripted controller")
parser.add_argument("--noise", type=float, default=0.3, help="Gaussian action noise for the scripted controller")
parser.add_argument("--episodes", type=int, default=20, help="episodes per env")
parser.add_argument("--clips", type=int, default=6, help="number of glitch clips to save")
parser.add_argument("--out", default="outputs/blowups")
simulation_app, args, overrides = launch(parser)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vis  # noqa: E402
from agent.drqv2 import MVMAEDrQV2Agent  # noqa: E402
from agent.replay import FrameStacker  # noqa: E402
from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from surgical_env.scripted import ScriptedNeedleLifter  # noqa: E402

CLIP_FRAMES = 20


def summarize(records: list[dict], episodes: int) -> str:
    lines = [f"episodes run: {episodes} | glitches: {len(records)} ({100.0 * len(records) / max(episodes, 1):.1f}% of episodes)"]
    if not records:
        return "\n".join(lines)
    lines.append("by cause: " + ", ".join(f"{k} {v}" for k, v in Counter(r["cause"] for r in records).most_common()))
    lines.append("fastest joint: " + ", ".join(f"{k} {v}" for k, v in Counter(r["fastest_joint"] for r in records).most_common()))
    keys = [k for k, v in records[0].items() if isinstance(v, float)]
    lines.append(f"{'':22s} {'p10':>9s} {'median':>9s} {'p90':>9s}")
    for k in keys:
        v = np.array([r[k] for r in records])
        lines.append(f"{k:22s} {np.percentile(v, 10):9.3f} {np.median(v):9.3f} {np.percentile(v, 90):9.3f}")

    def frac(pred) -> str:
        return f"{100.0 * np.mean([pred(r) for r in records]):5.1f}%"

    lines.append("share of glitches where:")
    lines.append(f"  tool tip within 5 mm of the needle      {frac(lambda r: r['tip_to_needle_mm'] < 5.0)}")
    lines.append(f"  gripper commanded closed                {frac(lambda r: r['gripper_command'] < 0.0)}")
    lines.append(f"  tool tip below the tissue surface       {frac(lambda r: r['tip_height_mm'] < 0.0)}")
    lines.append(f"  tissue dented more than 3 mm            {frac(lambda r: r['tissue_dent_mm'] > 3.0)}")
    lines.append(f"  needle below the tissue surface         {frac(lambda r: r['needle_height_mm'] < 0.0)}")
    lines.append(f"  in the first 5 steps of the episode     {frac(lambda r: r['episode_step'] <= 5)}")
    return "\n".join(lines)


def main() -> None:
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = load_config(overrides=overrides, base=ckpt["config"])
    else:
        ckpt, cfg = None, load_config(args.config, overrides)
    env = make_env(cfg, device=args.device)
    env.env.record_blowups = True
    n, dev = env.num_envs, env.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if ckpt is not None:
        h, w = env.obs_shape[-2:]
        agent = MVMAEDrQV2Agent(cfg.agent, cfg.mvmae, (h, w), env.obs_shape[0], env.action_dim, args.device)
        agent.load_state_dict(ckpt["agent"], load_optimizers=False)
        agent.train(False)
        stacker = FrameStacker(n, agent.frame_stack, env.obs_shape, dev)
        print(f"policy: {args.checkpoint} at {ckpt['env_steps']} env steps (training exploration noise)")
    else:
        controller = ScriptedNeedleLifter(n, env.step_dt, dev, cfg.env.ik_pos_scale, cfg.env.ik_rot_scale)
        print(f"policy: scripted controller + N(0, {args.noise}) action noise")

    obs = env.reset()
    stack = stacker.reset(obs) if ckpt is not None else None
    if ckpt is None:
        controller.reset_idx()
    history = [deque(maxlen=CLIP_FRAMES) for _ in range(n)]
    episodes, clips, seen = 0, 0, 0
    with torch.no_grad():
        while episodes < args.episodes * n:
            for e in range(n):
                history[e].append(obs[e].cpu())
            if ckpt is not None:
                action = agent.act(stack, ckpt["env_steps"], eval_mode=False)
            else:
                action = controller.compute(*env.poses_in_base())
                action = (action + args.noise * torch.randn_like(action)).clamp(-1.0, 1.0)
            obs, reward, term, trunc, info = env.step(action)
            done = term | trunc
            records = env.env.blowup_records
            for rec in records[seen:]:
                print(f"glitch #{len(records[:seen]) + 1}: " + ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
                if clips < args.clips:
                    frames = [vis.stereo_frame(f, scale=3) for f in history[rec["env_id"]]]
                    vis.save_mp4(frames, out / f"glitch_{clips}_{rec['cause']}.mp4", fps=5.0)  # slow motion
                    clips += 1
                seen += 1
            if bool(done.any()):
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                episodes += ids.numel()
                for e in ids.tolist():
                    history[e].clear()
                if ckpt is None:
                    controller.reset_idx(ids)
            if ckpt is not None:
                stack = stacker.step(obs, done)

    records = env.env.blowup_records
    text = summarize(records, episodes)
    print("\n" + "=" * 72 + "\n" + text + "\n" + "=" * 72)
    (out / "blowups.json").write_text(json.dumps({"episodes": episodes, "records": records}, indent=1))
    (out / "summary.txt").write_text(text + "\n")
    print(f"records, summary and {clips} clips (slow motion, last {CLIP_FRAMES} frames before each glitch) in {out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
