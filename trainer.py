"""Training loop for MV-MAE + DrQ-v2 on a vectorised stereo-image environment.

Kept independent of Isaac Sim: it works with any object that has the interface of
``surgical_env.env.StereoNeedleEnv`` (``reset``, ``step``, ``num_envs``,
``action_dim``, ``obs_shape``, ``max_episode_steps``, ``step_dt``, ``device``),
which is how tests/test_train_loop.py runs it on the CPU with a fake environment.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import vis
from agent.drqv2 import Batch, MVMAEDrQV2Agent
from agent.replay import FrameStacker, ReplayBuffer, concat_batches
from agent.utils import set_seed_everywhere


def scalar_logs(log: dict, prefix: str) -> dict[str, float]:
    """Keep the numeric scalar entries of an Isaac Lab extras["log"] dict."""
    out = {}
    for k, v in log.items():
        if torch.is_tensor(v) and v.numel() == 1:
            out[prefix + k] = float(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[prefix + k] = float(v)
    return out


def batch_to(batch: Batch, device) -> Batch:
    return Batch(*[t.to(device, non_blocking=True) for t in vars(batch).values()])


class MetricAverager:
    """Sums tensors on the GPU and syncs only when read (avoids a .item() per update)."""

    def __init__(self):
        self.sums: dict[str, torch.Tensor | float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, metrics: dict) -> None:
        for k, v in metrics.items():
            self.sums[k] = self.sums[k] + (v.detach().float() if torch.is_tensor(v) else float(v))
            self.counts[k] += 1

    def pop(self) -> dict[str, float]:
        out = {k: float(self.sums[k]) / self.counts[k] for k in self.sums}
        self.sums.clear()
        self.counts.clear()
        return out


@torch.no_grad()
def evaluate_policy(env, agent: MVMAEDrQV2Agent, env_step: int, record_video: bool, video_scale: int) -> tuple[dict, list]:
    """One deterministic episode in every parallel env. Resets the environment first.

    Returns (metrics, frames): frames are env 0's stereo views, left | right.
    """
    agent.train(False)
    n, dev = env.num_envs, env.device
    obs = env.reset()
    stacker = FrameStacker(n, agent.frame_stack, env.obs_shape, dev)
    stack = stacker.reset(obs)
    ret = torch.zeros(n, device=dev)
    length = torch.zeros(n, device=dev)
    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    success = torch.zeros(n, dtype=torch.bool, device=dev)
    success_any = torch.zeros(n, dtype=torch.bool, device=dev)
    goal_dist = torch.zeros(n, device=dev)
    frames = [vis.stereo_frame(obs[0], video_scale)] if record_video else []
    isaac_logs: list[dict] = []
    for _ in range(env.max_episode_steps + 1):
        action = agent.act(stack, env_step, eval_mode=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term | trunc
        alive = ~finished
        ret += reward * alive
        length += alive.float()
        success_any |= info["success_now"] & alive & ~done
        newly = done & alive
        success = torch.where(newly, info["final_success"], success)
        success_any |= newly & info["final_success"]
        goal_dist = torch.where(newly, info["final_goal_distance"], goal_dist)
        if info["log"] and bool(newly.any()):
            isaac_logs.append(info["log"])
        if record_video and not bool(finished[0]) and not bool(done[0]):
            frames.append(vis.stereo_frame(obs[0], video_scale))
        finished |= done
        stack = stacker.step(obs, done)
        if bool(finished.all()):
            break
    metrics = {
        "eval/episode_return": ret.mean().item(),
        "eval/episode_length": length.mean().item(),
        "eval/success": success.float().mean().item(),
        "eval/success_any": success_any.float().mean().item(),
        "eval/final_goal_distance": goal_dist.mean().item(),
    }
    per_key = defaultdict(list)
    for d in isaac_logs:
        for k, v in scalar_logs(d, "eval/isaac/").items():
            per_key[k].append(v)
    metrics.update({k: float(np.mean(v)) for k, v in per_key.items()})
    agent.train(True)
    return metrics, frames


class Trainer:
    def __init__(self, cfg, env, device, demo_data: dict | None = None):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)
        set_seed_everywhere(cfg.train.seed)
        t, a, m = cfg.train, cfg.agent, cfg.mvmae
        n = env.num_envs
        h, w = env.obs_shape[-2:]
        self.agent = MVMAEDrQV2Agent(a, m, (h, w), env.obs_shape[0], env.action_dim, self.device)
        replay_device = t.replay_device if (self.device.type == "cuda" or t.replay_device == "cpu") else "cpu"
        self.replay = ReplayBuffer(t.replay_capacity, n, env.obs_shape, env.action_dim, m.frame_stack, a.nstep, a.gamma,
                                   replay_device)
        self.demo = None
        if demo_data is not None and t.demo_ratio > 0:
            if tuple(demo_data["obs"].shape[1:]) != tuple(env.obs_shape):
                raise ValueError(f"demo images {tuple(demo_data['obs'].shape[1:])} != env images {env.obs_shape}")
            if demo_data["action"].shape[-1] != env.action_dim:
                raise ValueError("demo action size does not match the environment")
            self.demo = ReplayBuffer.from_episodes(demo_data, m.frame_stack, a.nstep, a.gamma, replay_device)
        self.stacker = FrameStacker(n, m.frame_stack, env.obs_shape, env.device)
        self.run_dir = Path(t.run_dir) / (cfg.log.run_name or time.strftime("%Y%m%d-%H%M%S"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if cfg.log.wandb_mode != "disabled":
            import wandb

            self.wandb = wandb.init(
                project=cfg.log.wandb_project,
                entity=cfg.log.wandb_entity or None,
                name=cfg.log.run_name or None,
                mode=cfg.log.wandb_mode,
                config=cfg.to_dict(),
                dir=str(self.run_dir),
            )
        print(f"MV-MAE tokens per sample: {self.agent.mvmae.L}, representation size: {self.agent.repr_dim}")
        n_params = sum(p.numel() for p in self.agent.mvmae.parameters())
        print(f"MV-MAE parameters: {n_params / 1e6:.2f} M | replay rows x envs: {self.replay.R} x {n} on {replay_device}"
              + (f" | demos: {self.demo.R} transitions" if self.demo is not None else ""))

    # ------------------------------------------------------------------ utils
    def log(self, data: dict, step: int) -> None:
        if self.wandb is not None and data:
            self.wandb.log(data, step=step)

    def sample(self) -> Batch:
        b = self.cfg.agent.batch_size
        if self.demo is not None:
            nd = int(round(b * self.cfg.train.demo_ratio))
            return concat_batches(self.replay.sample(b - nd), self.demo.sample(nd, is_demo=True), self.device)
        return batch_to(self.replay.sample(b), self.device)

    def save(self, env_steps: int, tag: str | None = None) -> Path:
        path = self.run_dir / f"ckpt_{tag or env_steps}.pt"
        torch.save({"agent": self.agent.state_dict(), "config": self.cfg.to_dict(), "env_steps": env_steps}, path)
        torch.save({"agent": self.agent.state_dict(), "config": self.cfg.to_dict(), "env_steps": env_steps},
                   self.run_dir / "ckpt_latest.pt")
        return path

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------- main
    def train(self) -> None:
        cfg, env, agent = self.cfg, self.env, self.agent
        t, lg = cfg.train, cfg.log
        n, dev = env.num_envs, env.device
        start = time.time()
        timers = defaultdict(float)
        metrics_avg = MetricAverager()
        env_steps, episodes, update_credit = 0, 0, 0.0
        next_eval, next_ckpt = t.eval_every_env_steps, t.checkpoint_every_env_steps
        mae_pretrained = t.mae_pretrain_updates <= 0
        last_log_updates, last_log_time, last_log_steps = 0, time.time(), 0
        last_batch_obs = None

        def fresh_start():
            o = env.reset()
            return o, self.stacker.reset(o), torch.ones(n, dtype=torch.bool, device=dev)

        obs, stack, first = fresh_start()
        ep_return = torch.zeros(n, device=dev)
        ep_length = torch.zeros(n, device=dev)
        ep_success_any = torch.zeros(n, dtype=torch.bool, device=dev)

        while env_steps < t.total_env_steps:
            # ---- act
            t0 = time.perf_counter()
            if env_steps < t.seed_env_steps:
                action = torch.rand(n, env.action_dim, device=dev) * 2.0 - 1.0
            else:
                action = agent.act(stack, env_steps, eval_mode=False)
            self._sync()
            t1 = time.perf_counter()
            next_obs, reward, term, trunc, info = env.step(action)
            self._sync()
            t2 = time.perf_counter()
            timers["act"] += t1 - t0
            timers["env"] += t2 - t1

            self.replay.add(obs, action, reward, term, trunc, first)
            done = term | trunc
            ep_return += reward
            ep_length += 1
            ep_success_any |= info["success_now"] & ~done
            env_steps += n

            if bool(done.any()):
                idx = done.nonzero(as_tuple=False).squeeze(-1)
                episodes += idx.numel()
                final = info["final_success"][idx]
                log = {
                    "train/episode_return": ep_return[idx].mean().item(),
                    "train/episode_length": ep_length[idx].mean().item(),
                    "train/success": final.float().mean().item(),
                    "train/success_any": (ep_success_any[idx] | final).float().mean().item(),
                    "train/final_goal_distance": info["final_goal_distance"][idx].mean().item(),
                    "train/terminated_frac": term[idx].float().mean().item(),
                    "train/episodes": episodes,
                }
                log.update(scalar_logs(info["log"], "train/isaac/"))
                self.log(log, env_steps)
                print(f"[{env_steps:>9d}] episodes {episodes:6d} | return {log['train/episode_return']:8.2f} | "
                      f"success {log['train/success']:.2f} | {time.time() - start:7.0f}s", flush=True)
                ep_return[idx] = 0.0
                ep_length[idx] = 0.0
                ep_success_any[idx] = False

            first = done
            stack = self.stacker.step(next_obs, first)
            obs = next_obs

            # ---- learn
            if env_steps >= t.seed_env_steps and self.replay.can_sample():
                t3 = time.perf_counter()
                if not mae_pretrained:
                    print(f"MV-MAE pre-training for {t.mae_pretrain_updates} updates ...", flush=True)
                    pre_avg = MetricAverager()
                    for i in range(t.mae_pretrain_updates):
                        batch = self.sample()
                        pre_avg.add(agent.update_mae_only(batch))
                        if (i + 1) % lg.log_every_updates == 0 or i + 1 == t.mae_pretrain_updates:
                            m = {f"pretrain/{k}": v for k, v in pre_avg.pop().items()}
                            print(f"  pretrain {i + 1}: recon {m.get('pretrain/mvmae/recon_loss', 0):.4f}", flush=True)
                            self.log(m, env_steps)
                    last_batch_obs = batch.obs
                    mae_pretrained = True
                update_credit += n * t.updates_per_env_step
                while update_credit >= 1.0:
                    batch = self.sample()
                    metrics_avg.add(agent.update(batch, env_steps))
                    update_credit -= 1.0
                    last_batch_obs = batch.obs
                    if agent.num_updates % lg.recon_every_updates == 0:
                        self._log_reconstruction(last_batch_obs, env_steps)
                self._sync()
                timers["update"] += time.perf_counter() - t3

            # ---- periodic logging of learner metrics / speed
            if agent.num_updates - last_log_updates >= lg.log_every_updates:
                now = time.time()
                log = {f"agent/{k}" if not k.startswith(("train/", "grad/", "mvmae/", "critic/", "actor/")) else k: v
                       for k, v in metrics_avg.pop().items()}
                total_t = sum(timers.values()) or 1.0
                log.update({
                    "train/env_steps": env_steps,
                    "train/updates": agent.num_updates,
                    "train/stddev": agent.stddev(env_steps),
                    "train/lr_encoder": agent.encoder_opt.param_groups[0]["lr"],
                    "train/replay_size": len(self.replay),
                    "speed/env_steps_per_s": (env_steps - last_log_steps) / max(now - last_log_time, 1e-6),
                    "speed/updates_per_s": (agent.num_updates - last_log_updates) / max(now - last_log_time, 1e-6),
                    "speed/frac_env": timers["env"] / total_t,
                    "speed/frac_act": timers["act"] / total_t,
                    "speed/frac_update": timers["update"] / total_t,
                    "time/wall_clock_s": now - start,
                })
                if self.device.type == "cuda":
                    log["system/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
                    log["system/gpu_mem_max_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
                self.log(log, env_steps)
                timers.clear()
                last_log_updates, last_log_time, last_log_steps = agent.num_updates, now, env_steps

            # ---- evaluation (resets every env; running training episodes are cut)
            if env_steps >= next_eval:
                next_eval += t.eval_every_env_steps
                self.replay.mark_last_truncated()
                metrics, frames = evaluate_policy(env, agent, env_steps, lg.video, lg.video_scale)
                print(f"[{env_steps:>9d}] EVAL return {metrics['eval/episode_return']:.2f} | "
                      f"success {metrics['eval/success']:.2f}", flush=True)
                if frames:
                    video_path = self.run_dir / "videos" / f"eval_{env_steps}.mp4"
                    vis.save_mp4(frames, video_path, fps=1.0 / env.step_dt)
                    if self.wandb is not None:
                        try:
                            metrics["eval/video"] = vis.wandb_video(video_path)
                        except Exception as e:  # a failed video must never stop training
                            print(f"video logging failed: {e}")
                self.log(metrics, env_steps)
                obs, stack, first = fresh_start()
                ep_return.zero_()
                ep_length.zero_()
                ep_success_any.zero_()

            if env_steps >= next_ckpt:
                next_ckpt += t.checkpoint_every_env_steps
                print(f"saved {self.save(env_steps)}", flush=True)

        print(f"saved {self.save(env_steps, tag='final')}", flush=True)
        if self.wandb is not None:
            self.wandb.finish()

    def _log_reconstruction(self, obs: torch.Tensor | None, env_steps: int) -> None:
        if obs is None:
            return
        img = vis.upscale(self.agent.reconstruction_image(obs).numpy(), 2)
        vis.save_png(img, self.run_dir / "reconstructions" / f"recon_{self.agent.num_updates}.png")
        if self.wandb is not None:
            import wandb

            self.log({"mvmae/reconstruction": wandb.Image(
                img, caption="rows: frames (oldest first) | cols: truth L, truth R, visible L, visible R, recon L, recon R")},
                env_steps)
