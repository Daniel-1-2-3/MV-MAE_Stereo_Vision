"""Measure what env_0's paper camera costs: environment steps per second with and without it.

    python tools/bench_hires_camera.py                       # defaults from configs/needle_tissue.yaml (32 envs)
    python tools/bench_hires_camera.py env.num_envs=16 --steps 500 --repeats 3

One simulator, random actions, timing env.step only (synchronised):
  A  the env exactly as training builds it with hires.enabled=false (no paper camera)
  B  with the paper camera prim in env_0 but no render product: what training runs with (hires.enabled=true)
  C  with its render products alive and every frame read and encoded: what one evaluation episode costs
A and B alternate --repeats times (the prim is added and removed) so their difference can be told from noise.
B must stay within ~5 % of A. Writes outputs/bench_hires/bench.json (and a sample video from C).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_app import launch  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/needle_tissue.yaml")
parser.add_argument("--steps", type=int, default=300, help="timed env steps per measurement")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--out", default="outputs/bench_hires")
simulation_app, args, overrides = launch(parser)

import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402


def steps_per_s(env, steps: int, warmup: int, capture=None) -> float:
    env.reset()
    act = lambda: torch.rand(env.num_envs, env.action_dim, device=env.device) * 2 - 1  # noqa: E731
    for _ in range(warmup):
        env.step(act())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act())
        if capture is not None:
            capture()
    torch.cuda.synchronize()
    return steps * env.num_envs / (time.perf_counter() - t0)


def main() -> None:
    import isaaclab.sim as sim_utils

    cfg = load_config(args.config, overrides)
    out = Path(args.out)
    env = make_env(cfg, device=args.device, hires_camera=False)
    stage = sim_utils.get_current_stage()
    a, b = [], []
    for r in range(args.repeats):
        a.append(steps_per_s(env, args.steps, args.warmup))
        env.add_hires_camera(cfg.hires, cfg.env)
        b.append(steps_per_s(env, args.steps, args.warmup))
        print(f"repeat {r}: A (no paper camera) {a[-1]:.0f} env steps/s | B (paper camera prim, no render product) "
              f"{b[-1]:.0f} env steps/s", flush=True)
        if r + 1 < args.repeats:
            stage.RemovePrim(env.hires_prim)
            env.hires_prim = None
    rec = env.hires_recorder(cfg.hires, out, 0, fps=1.0 / env.step_dt)
    rec.start()
    c = steps_per_s(env, min(args.steps, env.max_episode_steps), args.warmup, capture=rec.capture)
    videos = rec.stop()
    res = {"num_envs": env.num_envs, "steps": args.steps, "repeats": args.repeats,
           "A_no_camera": a, "B_camera_prim_only": b, "C_recording": c,
           "A_median": statistics.median(a), "B_median": statistics.median(b)}
    res["training_slowdown_pct"] = 100.0 * (1.0 - res["B_median"] / res["A_median"])
    res["recording_slowdown_pct"] = 100.0 * (1.0 - c / res["A_median"])
    res["videos"] = {k: str(v) for k, v in videos.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=1))
    print(f"\ntraining env steps/s without the paper camera: {res['A_median']:.0f}, with it: {res['B_median']:.0f} "
          f"-> slowdown {res['training_slowdown_pct']:+.1f} % (requirement: < 5 %)")
    print(f"while recording (evaluation episodes only): {c:.0f} env steps/s ({res['recording_slowdown_pct']:+.1f} %)")
    print(f"saved {out / 'bench.json'}")
    env.close()


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
