"""Record demonstrations with the scripted controller (needle lift or two-arm handover).

    python record_demos.py --num_demos 50 --out demos/needle_demos.pt

Runs the scripted controller for env.task (surgical_env/scripted.py: the Isaac for Healthcare
pick-and-lift state machine, or the two-arm handover one) in every parallel env, keeps only episodes that end in success, and stores their
stereo images, actions and rewards in the format train.py loads with
train.demo_path=... . Use the same env/camera config as for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sim_app import launch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/needle_tissue.yaml")
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--max_episodes", type=int, default=1000, help="give up after this many attempts")
parser.add_argument("--out", default="demos/needle_demos.pt")
simulation_app, args, overrides = launch(parser)

import torch  # noqa: E402

import vis  # noqa: E402
from config import load_config  # noqa: E402
from surgical_env.env import make_env  # noqa: E402
from surgical_env.scripted import make_scripted_controller  # noqa: E402


def main() -> None:
    cfg = load_config(args.config, overrides)
    env = make_env(cfg, device=args.device)
    n = env.num_envs
    controller = make_scripted_controller(env, cfg)  # lift or handover, per env.task

    # Per-env lists of the running episode's steps (kept on the CPU).
    running = [dict(obs=[], action=[], reward=[], terminated=[], truncated=[], proprio=[]) for _ in range(n)]
    kept: list[dict] = []
    attempts, successes = 0, 0
    obs = env.reset()
    prop = env.proprio()
    controller.reset_idx()
    while len(kept) < args.num_demos and attempts < args.max_episodes:
        action = controller.act(env)
        next_obs, reward, term, trunc, info = env.step(action)
        obs_c, act_c, rew_c, prop_c = obs.cpu(), action.cpu(), reward.cpu(), prop.cpu()
        term_c, trunc_c = term.cpu(), trunc.cpu()
        for e in range(n):
            ep = running[e]
            ep["obs"].append(obs_c[e])
            ep["action"].append(act_c[e])
            ep["reward"].append(rew_c[e])
            ep["terminated"].append(term_c[e])
            ep["truncated"].append(trunc_c[e])
            ep["proprio"].append(prop_c[e])
        done = term | trunc
        if bool(done.any()):
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for e in ids.tolist():
                attempts += 1
                if bool(info["final_success"][e]) and len(kept) < args.num_demos:
                    successes += 1
                    kept.append({k: torch.stack(v) for k, v in running[e].items()})
                running[e] = dict(obs=[], action=[], reward=[], terminated=[], truncated=[], proprio=[])
            controller.reset_idx(ids)
            print(f"attempts {attempts:4d} | successes {successes:4d} | kept {len(kept)}/{args.num_demos}", flush=True)
        obs = next_obs
        prop = env.proprio()

    if not kept:
        env.close()
        raise SystemExit(f"the scripted controller succeeded in 0 of {attempts} episodes; nothing saved")

    first = []
    for ep in kept:
        f = torch.zeros(ep["obs"].shape[0], dtype=torch.bool)
        f[0] = True
        first.append(f)
    data = {
        "obs": torch.cat([ep["obs"] for ep in kept]),
        "action": torch.cat([ep["action"] for ep in kept]).float(),
        "reward": torch.cat([ep["reward"] for ep in kept]).float(),
        "terminated": torch.cat([ep["terminated"] for ep in kept]).bool(),
        "truncated": torch.cat([ep["truncated"] for ep in kept]).bool(),
        "proprio": torch.cat([ep["proprio"] for ep in kept]).float(),  # robot state at each obs (for agent.proprio)
        "first": torch.cat(first),
        "meta": {
            "num_episodes": len(kept),
            "attempts": attempts,
            "success_rate": successes / max(attempts, 1),
            "env": cfg.to_dict()["env"],
            "camera": cfg.to_dict()["camera"],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"saved {len(kept)} successful episodes ({data['obs'].shape[0]} transitions) to {out}; "
          f"scripted success rate {data['meta']['success_rate']:.2f}")
    frames = [vis.stereo_frame(o, scale=3) for o in kept[0]["obs"]]
    vis.save_mp4(frames, out.with_suffix(".mp4"), fps=1.0 / env.step_dt)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
