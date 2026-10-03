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


def build_agent(cfg, env, device) -> MVMAEDrQV2Agent:
    """The agent for this config and env (with the robot state as an extra input if agent.proprio)."""
    h, w = env.obs_shape[-2:]
    proprio_dim = env.proprio_dim if cfg.agent.proprio else 0
    return MVMAEDrQV2Agent(cfg.agent, cfg.mvmae, (h, w), env.obs_shape[0], env.action_dim, device, proprio_dim)


def robot_state(env, agent: MVMAEDrQV2Agent):
    """The env's robot state if the agent uses it, else None."""
    return env.proprio() if agent.proprio_dim > 0 else None


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


EVAL_BUCKETS = ("never_lifted", "lifted_dropped", "missed_goal", "glitch_cut")
CENTER_RADIUS_M = 0.015  # needle landing spots closer than this to the pad centre count as "center"


@torch.no_grad()
def evaluate_policy(env, agent: MVMAEDrQV2Agent, env_step: int, record_video: bool, video_scale: int):
    """One deterministic episode in every parallel env. Resets the environment first.

    Returns (metrics, frames, fail_frames), frames = one env's stereo views (left | right).
    `frames` shows the first env that succeeded, else the first that ever reached success,
    else the one that kept the needle lifted longest, else env 0 (metrics["eval/video_env"]
    and ["eval/video_env_success"] say which). `fail_frames` shows a failed episode from the
    most common failure bucket (empty if every env succeeded).

    Every failed episode is also put in a bucket, logged as eval/fail/<bucket> (fraction of
    all episodes): never_lifted (needle never above the lift height), lifted_dropped (lifted
    at some point but not at the end), missed_goal (still lifted at the end, not at the goal),
    glitch_cut (cut early by a blowup_* time-out). eval/success_center / eval/success_edge
    split the success rate by where the needle landed on the pad.
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
    history = [obs.clone()] if record_video else []  # (n, V, 3, H, W) uint8 per step, on the sim device
    valid = torch.ones(n, dtype=torch.long, device=dev)  # frames of history that belong to each env's episode
    lifted_steps = torch.zeros(n, device=dev)
    ever_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    lifted_prev = torch.zeros(n, dtype=torch.bool, device=dev)
    final_lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    cut_early = torch.zeros(n, dtype=torch.bool, device=dev)
    radius = None
    isaac_logs: list[dict] = []
    prop = robot_state(env, agent)
    for t in range(env.max_episode_steps + 1):
        if t == 5 and hasattr(env, "needle_xy_local"):  # the needle has landed on the pad
            radius = torch.linalg.vector_norm(env.needle_xy_local(), dim=-1)
        action = agent.act(stack, env_step, eval_mode=True, proprio=prop)
        obs, reward, term, trunc, info = env.step(action)
        done = term | trunc
        alive = ~finished
        live = alive & ~done  # still inside the same episode after this step
        ret += reward * alive
        length += alive.float()
        success_any |= info["success_now"] & live
        newly = done & alive
        success = torch.where(newly, info["final_success"], success)
        success_any |= newly & info["final_success"]
        goal_dist = torch.where(newly, info["final_goal_distance"], goal_dist)
        cut_early |= newly & trunc & (length < env.max_episode_steps)
        if info["log"] and bool(newly.any()):
            isaac_logs.append(info["log"])
        if record_video:
            history.append(obs.clone())
            valid += live.long()  # after `done` the returned frame already shows the reset scene
        if "needle_lifted" in info:
            lifted = info["needle_lifted"]
            lifted_steps += (lifted & live).float()
            ever_lifted |= lifted & live
            final_lifted |= newly & lifted_prev  # lifted on the last step inside the episode
            lifted_prev = torch.where(live, lifted, lifted_prev)
        finished |= done
        stack = stacker.step(obs, done)
        prop = robot_state(env, agent)
        if bool(finished.all()):
            break

    failed = ~success
    buckets = {
        "glitch_cut": failed & cut_early,
        "never_lifted": failed & ~cut_early & ~ever_lifted,
        "missed_goal": failed & ~cut_early & ever_lifted & final_lifted,
        "lifted_dropped": failed & ~cut_early & ever_lifted & ~final_lifted,
    }
    frames: list = []
    fail_frames: list = []
    if record_video:
        if bool(success.any()):
            pick, pick_ok = int(success.nonzero()[0]), 1.0
        elif bool(success_any.any()):
            pick, pick_ok = int(success_any.nonzero()[0]), 0.5
        else:
            pick, pick_ok = int(lifted_steps.argmax()) if bool((lifted_steps > 0).any()) else 0, 0.0
        frames = [vis.stereo_frame(history[i][pick], video_scale) for i in range(int(valid[pick]))]
        worst = max(EVAL_BUCKETS, key=lambda b: int(buckets[b].sum()))
        if bool(buckets[worst].any()):
            fail_pick = int(buckets[worst].nonzero()[0])
            fail_frames = [vis.stereo_frame(history[i][fail_pick], video_scale) for i in range(int(valid[fail_pick]))]
    metrics = {
        "eval/episode_return": ret.mean().item(),
        "eval/episode_length": length.mean().item(),
        "eval/success": success.float().mean().item(),
        "eval/success_any": success_any.float().mean().item(),
        "eval/final_goal_distance": goal_dist.mean().item(),
    }
    if "needle_lifted" in info:
        metrics.update({f"eval/fail/{b}": buckets[b].float().mean().item() for b in EVAL_BUCKETS})
    if radius is not None:
        center = radius < CENTER_RADIUS_M
        if bool(center.any()):
            metrics["eval/success_center"] = success[center].float().mean().item()
        if bool((~center).any()):
            metrics["eval/success_edge"] = success[~center].float().mean().item()
    if record_video:
        metrics["eval/video_env"] = float(pick)
        metrics["eval/video_env_success"] = pick_ok  # 1 = succeeded, 0.5 = reached success at some point, 0 = neither
        if fail_frames:
            metrics["eval/failure_video_bucket"] = float(EVAL_BUCKETS.index(worst))
    per_key = defaultdict(list)
    for d in isaac_logs:
        for k, v in scalar_logs(d, "eval/isaac/").items():
            per_key[k].append(v)
    metrics.update({k: float(np.mean(v)) for k, v in per_key.items()})
    agent.train(True)
    return metrics, frames, fail_frames


class Trainer:
    def __init__(self, cfg, env, device, demo_data: dict | None = None):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)
        self.best_score = None  # (success, success_any, return) of the best evaluation so far
        self.freeze_streak = 0  # consecutive evaluations at or above train.freeze_encoder_success
        set_seed_everywhere(cfg.train.seed)
        t, a, m = cfg.train, cfg.agent, cfg.mvmae
        n = env.num_envs
        self.agent = build_agent(cfg, env, self.device)
        pd = self.agent.proprio_dim
        replay_device = t.replay_device if (self.device.type == "cuda" or t.replay_device == "cpu") else "cpu"
        self.replay = ReplayBuffer(t.replay_capacity, n, env.obs_shape, env.action_dim, m.frame_stack, a.nstep, a.gamma,
                                   replay_device, pd)
        self.demo = None
        if demo_data is not None and t.demo_ratio > 0:
            if tuple(demo_data["obs"].shape[1:]) != tuple(env.obs_shape):
                raise ValueError(f"demo images {tuple(demo_data['obs'].shape[1:])} != env images {env.obs_shape}")
            if demo_data["action"].shape[-1] != env.action_dim:
                raise ValueError("demo action size does not match the environment")
            self.demo = ReplayBuffer.from_episodes(demo_data, m.frame_stack, a.nstep, a.gamma, replay_device, pd)
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
        enc = (f"MV-MAE tokens per sample: {self.agent.mvmae.L}" if self.agent.uses_mvmae
               else "BASELINE image encoder: DrQ-v2 conv net on the raw stereo frames (no MV-MAE)")
        print(f"{enc}, representation size: {self.agent.repr_dim}" + (f", robot state: {pd} values" if pd else ", images only"))
        n_params = sum(p.numel() for p in self.agent.encoder.parameters())
        print(f"encoder parameters: {n_params / 1e6:.2f} M | replay rows x envs: {self.replay.R} x {n} on {replay_device}"
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

    def save(self, env_steps: int, tag: str | None = None, update_latest: bool = True) -> Path:
        path = self.run_dir / f"ckpt_{tag or env_steps}.pt"
        state = {"agent": self.agent.state_dict(), "config": self.cfg.to_dict(), "env_steps": env_steps}
        torch.save(state, path)
        if update_latest:
            torch.save(state, self.run_dir / "ckpt_latest.pt")
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
        mae_pretrained = t.mae_pretrain_updates <= 0 or not agent.uses_mvmae  # nothing to pre-train in the baseline
        last_log_updates, last_log_time, last_log_steps = 0, time.time(), 0
        last_batch_obs = None

        def fresh_start():
            o = env.reset()
            return o, self.stacker.reset(o), torch.ones(n, dtype=torch.bool, device=dev), robot_state(env, agent)

        obs, stack, first, prop = fresh_start()
        ep_return = torch.zeros(n, device=dev)
        ep_length = torch.zeros(n, device=dev)
        ep_success_any = torch.zeros(n, dtype=torch.bool, device=dev)

        while env_steps < t.total_env_steps:
            # ---- act
            t0 = time.perf_counter()
            if env_steps < t.seed_env_steps:
                action = torch.rand(n, env.action_dim, device=dev) * 2.0 - 1.0
            else:
                action = agent.act(stack, env_steps, eval_mode=False, proprio=prop)
            self._sync()
            t1 = time.perf_counter()
            next_obs, reward, term, trunc, info = env.step(action)
            self._sync()
            t2 = time.perf_counter()
            timers["act"] += t1 - t0
            timers["env"] += t2 - t1

            self.replay.add(obs, action, reward, term, trunc, first, proprio=prop)
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
            prop = robot_state(env, agent)

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
                metrics, frames, fail_frames = evaluate_policy(env, agent, env_steps, lg.video, lg.video_scale)
                print(f"[{env_steps:>9d}] EVAL return {metrics['eval/episode_return']:.2f} | "
                      f"success {metrics['eval/success']:.2f}", flush=True)
                fails = ", ".join(f"{k.split('/')[-1]} {v:.2f}" for k, v in metrics.items() if k.startswith("eval/fail/"))
                if fails:
                    print(f"[{env_steps:>9d}] EVAL failures (fraction of episodes): {fails}", flush=True)
                for key, clip, name in (("eval/video", frames, "eval"), ("eval/failure_video", fail_frames, "eval_failure")):
                    if not clip:
                        continue
                    video_path = self.run_dir / "videos" / f"{name}_{env_steps}.mp4"
                    vis.save_mp4(clip, video_path, fps=1.0 / env.step_dt)
                    if self.wandb is not None:
                        try:
                            metrics[key] = vis.wandb_video(video_path)
                        except Exception as e:  # a failed video must never stop training
                            print(f"video logging failed: {e}")
                self.log(metrics, env_steps)
                self._maybe_freeze_encoder(metrics["eval/success_any"], env_steps)
                score = (metrics["eval/success"], metrics["eval/success_any"], metrics["eval/episode_return"])
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    print(f"new best eval (success {score[0]:.3f}, success_any {score[1]:.3f}): "
                          f"saved {self.save(env_steps, tag='best', update_latest=False)}", flush=True)
                obs, stack, first, prop = fresh_start()
                ep_return.zero_()
                ep_length.zero_()
                ep_success_any.zero_()

            if t.freeze_encoder_at > 0 and env_steps >= t.freeze_encoder_at and not agent.encoder_frozen:
                self._freeze_encoder(env_steps, f"reached train.freeze_encoder_at={t.freeze_encoder_at}")

            if env_steps >= next_ckpt:
                next_ckpt += t.checkpoint_every_env_steps
                print(f"saved {self.save(env_steps)}", flush=True)
                self._upload_checkpoints("latest", "best", "frozen")

        print(f"saved {self.save(env_steps, tag='final')}", flush=True)
        self._upload_checkpoints("final", "best", "frozen")
        if self.wandb is not None:
            self.wandb.finish()

    def _maybe_freeze_encoder(self, success_any: float, env_steps: int) -> None:
        t = self.cfg.train
        if self.agent.encoder_frozen or t.freeze_encoder_success <= 0:
            return
        self.freeze_streak = self.freeze_streak + 1 if success_any >= t.freeze_encoder_success else 0
        if self.freeze_streak >= max(1, t.freeze_encoder_evals):
            self._freeze_encoder(env_steps, f"eval success_any >= {t.freeze_encoder_success} in "
                                            f"{self.freeze_streak} evaluations in a row")

    def _freeze_encoder(self, env_steps: int, reason: str) -> None:
        self.agent.freeze_encoder()
        print(f"[{env_steps:>9d}] ENCODER FROZEN ({reason}); saved {self.save(env_steps, tag='frozen', update_latest=False)}",
              flush=True)
        self.log({"train/encoder_frozen_at": float(env_steps)}, env_steps)
        self._upload_checkpoints("frozen")

    def _upload_checkpoints(self, *tags: str) -> None:
        """Copy checkpoints to wandb (Artifacts tab) so they survive the machine: ckpt_<tag>.pt per tag.

        Get one back with: wandb artifact get <entity>/<project>/<run name>-ckpt-<tag>:latest --root <dir>
        """
        if self.wandb is None or not self.cfg.log.upload_checkpoints:
            return
        import wandb

        name = (self.cfg.log.run_name or self.wandb.id).replace("/", "-")
        for tag in tags:
            path = self.run_dir / f"ckpt_{tag}.pt"
            if not path.exists():
                continue
            try:
                art = wandb.Artifact(f"{name}-ckpt-{tag}", type="model")
                art.add_file(str(path), name=path.name)
                self.wandb.log_artifact(art)
            except Exception as e:  # an upload problem must never stop training
                print(f"checkpoint upload failed ({tag}): {e}")

    def _log_reconstruction(self, obs: torch.Tensor | None, env_steps: int) -> None:
        if obs is None or not self.agent.uses_mvmae:
            return
        img = vis.upscale(self.agent.reconstruction_image(obs).numpy(), 2)
        vis.save_png(img, self.run_dir / "reconstructions" / f"recon_{self.agent.num_updates}.png")
        if self.wandb is not None:
            import wandb

            self.log({"mvmae/reconstruction": wandb.Image(
                img, caption="rows: frames (oldest first) | cols: truth L, truth R, visible L, visible R, recon L, recon R")},
                env_steps)
