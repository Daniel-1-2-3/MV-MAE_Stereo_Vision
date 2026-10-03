"""Evaluate a checkpoint and sort every episode by how it ended, with videos of the failures.

    python eval_failures.py --checkpoint runs/<run>/ckpt_best.pt --rounds 4
    python eval_failures.py --checkpoint runs/<run>/ckpt_latest.pt --rounds 4 --videos_per_bucket 8

Each round runs one deterministic episode (the policy's mean action, like eval/ in
training) in every parallel env. Every episode goes into one bucket:

    success        needle lifted and within the success distance of the goal at the end
    never_lifted   the needle never rose above the lift height (grasp failed / never tried)
    lifted_dropped the needle was lifted at some point but was not lifted at the end
    missed_goal    the needle was still lifted at the end but not close enough to the goal
    glitch_cut     the episode was cut early by a blowup_* time-out

For each bucket the summary gives the needle's starting position on the pad and how it
landed (yaw / tilt after the drop), so position- or pose-dependent failures stand out.
The needle's spawn orientation is not randomised (only x/y within +-env.needle_xy_range),
so yaw/tilt only vary by how the needle settled.

Outputs (default outputs/eval_failures/):
  summary.txt     bucket counts and per-bucket start-pose statistics
  episodes.csv    one row per episode
  <bucket>_<k>.mp4  stereo videos (left | right) of up to --videos_per_bucket episodes per bucket
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--rounds", type=int, default=4, help="episodes per env")
parser.add_argument("--videos_per_bucket", type=int, default=5)
parser.add_argument("--out", default="outputs/eval_failures")
simulation_app, args, overrides = launch(parser)

import csv  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vis  # noqa: E402
from agent.replay import FrameStacker  # noqa: E402
from config import load_config  # noqa: E402
from surgical_env import mdp  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from trainer import build_agent, robot_state  # noqa: E402

BUCKETS = ("success", "never_lifted", "lifted_dropped", "missed_goal", "glitch_cut")


def yaw_tilt_deg(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(w, x, y, z) quaternions -> yaw about world z and tilt of the body z axis from world z, degrees."""
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    body_z_up = 1 - 2 * (x * x + y * y)  # z component of the rotated body z axis
    tilt = torch.acos(body_z_up.clamp(-1.0, 1.0))
    return torch.rad2deg(yaw), torch.rad2deg(tilt)


@torch.no_grad()
def run_round(env, agent, env_step: int, settle_steps: int) -> tuple[list[dict], list[torch.Tensor], torch.Tensor]:
    n, dev = env.num_envs, env.device
    scene = env.env.scene
    needle = scene["object"]
    top = env.env.cfg.tissue_top_w
    max_steps = env.max_episode_steps

    obs = env.reset()
    stacker = FrameStacker(n, agent.frame_stack, env.obs_shape, dev)
    stack = stacker.reset(obs)
    history = [obs.clone()]
    valid = torch.ones(n, dtype=torch.long, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    lifted_steps = torch.zeros(n, device=dev)
    lifted_prev = torch.zeros(n, dtype=torch.bool, device=dev)
    success_any = torch.zeros(n, dtype=torch.bool, device=dev)
    min_goal = torch.full((n,), float("inf"), device=dev)
    max_height = torch.zeros(n, device=dev)
    rec = {k: torch.zeros(n, device=dev) for k in ("x_mm", "y_mm", "yaw_deg", "tilt_deg", "settle_height_mm")}
    rec_bool = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ("final_success", "terminated", "cut_early")}
    final_goal = torch.zeros(n, device=dev)
    final_lifted = torch.zeros(n, dtype=torch.bool, device=dev)  # lifted on the last step inside the episode

    for t in range(max_steps + 1):
        if t == settle_steps:  # the needle has landed on the pad: record how it lies
            local = needle.data.root_pos_w - scene.env_origins
            yaw, tilt = yaw_tilt_deg(needle.data.root_quat_w)
            rec["x_mm"], rec["y_mm"] = local[:, 0] * 1000, local[:, 1] * 1000
            rec["yaw_deg"], rec["tilt_deg"] = yaw, tilt
            rec["settle_height_mm"] = (needle.data.root_pos_w[:, 2] - top) * 1000
        action = agent.act(stack, env_step, eval_mode=True, proprio=robot_state(env, agent))
        obs, reward, term, trunc, info = env.step(action)
        done = term | trunc
        live = alive & ~done  # still inside the same episode after this step
        lifted = info["needle_lifted"]
        ever_lifted |= lifted & live
        lifted_steps += (lifted & live).float()
        success_any |= info["success_now"] & live
        goal_distance = mdp.needle_goal_distance(env.env)
        min_goal = torch.where(live, torch.minimum(min_goal, goal_distance), min_goal)
        height = (needle.data.root_pos_w[:, 2] - top) * 1000
        max_height = torch.where(live, torch.maximum(max_height, height), max_height)
        length += alive.long()
        newly = done & alive
        rec_bool["final_success"] |= newly & info["final_success"]
        rec_bool["terminated"] |= newly & term
        rec_bool["cut_early"] |= newly & trunc & (length < max_steps)
        final_goal = torch.where(newly, info["final_goal_distance"], final_goal)
        # the frame returned with `done` already shows the reset scene, so use the previous step
        final_lifted |= newly & lifted_prev
        lifted_prev = torch.where(live, lifted, lifted_prev)
        success_any |= newly & info["final_success"]
        history.append(obs.clone())
        valid += live.long()
        alive &= ~done
        stack = stacker.step(obs, done)
        if not bool(alive.any()):
            break

    episodes = []
    for e in range(n):
        if bool(rec_bool["final_success"][e]):
            bucket = "success"
        elif bool(rec_bool["cut_early"][e]):
            bucket = "glitch_cut"
        elif not bool(ever_lifted[e]):
            bucket = "never_lifted"
        elif bool(final_lifted[e]):
            bucket = "missed_goal"
        else:
            bucket = "lifted_dropped"
        episodes.append({
            "env": e,
            "bucket": bucket,
            "length": int(length[e]),
            "success_any": bool(success_any[e]),
            "lifted_steps": int(lifted_steps[e]),
            "max_needle_height_mm": round(float(max_height[e]), 2),
            "min_goal_distance_mm": round(float(min_goal[e]) * 1000, 2),
            "final_goal_distance_mm": round(float(final_goal[e]) * 1000, 2),
            "needle_dropped_off": bool(rec_bool["terminated"][e]),
            **{k: round(float(v[e]), 2) for k, v in rec.items()},
        })
    return episodes, history, valid


def summarize(episodes: list[dict]) -> str:
    total = len(episodes)
    lines = [f"episodes: {total}"]
    for b in BUCKETS:
        k = sum(ep["bucket"] == b for ep in episodes)
        lines.append(f"  {b:15s} {k:4d}  ({100.0 * k / max(total, 1):5.1f}%)")
    lines.append("")
    lines.append("start pose per bucket (mean +- std): needle x, y on the pad (mm from centre), distance from centre,")
    lines.append("yaw and tilt after landing (deg), and how far it got")
    header = f"{'bucket':15s} {'x_mm':>13s} {'y_mm':>13s} {'r_mm':>13s} {'yaw_deg':>13s} {'tilt_deg':>13s} {'lifted_steps':>13s} {'min_goal_mm':>13s}"
    lines.append(header)
    for b in BUCKETS:
        eps = [ep for ep in episodes if ep["bucket"] == b]
        if not eps:
            continue

        def ms(vals):
            v = np.array(vals, dtype=float)
            return f"{v.mean():6.1f}+-{v.std():5.1f}"

        r = [math.hypot(ep["x_mm"], ep["y_mm"]) for ep in eps]
        lines.append(f"{b:15s} {ms([ep['x_mm'] for ep in eps]):>13s} {ms([ep['y_mm'] for ep in eps]):>13s} {ms(r):>13s} "
                     f"{ms([ep['yaw_deg'] for ep in eps]):>13s} {ms([ep['tilt_deg'] for ep in eps]):>13s} "
                     f"{ms([ep['lifted_steps'] for ep in eps]):>13s} {ms([ep['min_goal_distance_mm'] for ep in eps]):>13s}")
    # success rate by distance of the needle from the pad centre (quartiles)
    r_all = np.array([math.hypot(ep["x_mm"], ep["y_mm"]) for ep in episodes])
    ok = np.array([ep["bucket"] == "success" for ep in episodes])
    if total >= 8:
        lines.append("")
        lines.append("success rate by needle distance from the pad centre (quartiles):")
        edges = np.quantile(r_all, [0, 0.25, 0.5, 0.75, 1.0])
        for i in range(4):
            m = (r_all >= edges[i]) & (r_all <= edges[i + 1])
            lines.append(f"  {edges[i]:5.1f}-{edges[i + 1]:5.1f} mm: {100.0 * ok[m].mean():5.1f}% of {int(m.sum())}")
    return "\n".join(lines)


def main() -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(overrides=overrides, base=ckpt["config"])
    env = make_env(cfg, device=args.device)
    agent = build_agent(cfg, env, args.device)
    agent.load_state_dict(ckpt["agent"], load_optimizers=False)
    agent.train(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / env.step_dt
    settle_steps = 5  # 0.2 s: the needle, dropped from 1.5 cm, has landed
    print(f"checkpoint {args.checkpoint} ({ckpt['env_steps']} env steps), {args.rounds} rounds x {env.num_envs} envs")

    episodes, saved = [], {b: 0 for b in BUCKETS}
    for r in range(args.rounds):
        eps, history, valid = run_round(env, agent, ckpt["env_steps"], settle_steps)
        for ep in eps:
            ep["round"] = r
            b, e = ep["bucket"], ep["env"]
            if saved[b] < args.videos_per_bucket:
                frames = [vis.stereo_frame(history[i][e], scale=3) for i in range(int(valid[e]))]
                vis.save_mp4(frames, out / f"{b}_{saved[b]}.mp4", fps=fps)
                ep["video"] = f"{b}_{saved[b]}.mp4"
                saved[b] += 1
        episodes.extend(eps)
        counts = {b: sum(ep["bucket"] == b for ep in eps) for b in BUCKETS}
        print(f"round {r}: " + ", ".join(f"{b} {k}" for b, k in counts.items()), flush=True)

    text = summarize(episodes)
    print("\n" + "=" * 100 + "\n" + text + "\n" + "=" * 100)
    (out / "summary.txt").write_text(text + "\n")
    keys = ["round", "env", "bucket", "video"] + [k for k in episodes[0] if k not in ("round", "env", "bucket", "video")]
    with open(out / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for ep in episodes:
            writer.writerow({k: ep.get(k, "") for k in keys})
    print(f"summary, per-episode csv and failure videos in {out}")
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
