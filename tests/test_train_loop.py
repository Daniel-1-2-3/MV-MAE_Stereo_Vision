"""End-to-end run of the training loop on the CPU with a fake stereo environment.

Exercises everything except Isaac Sim: seeding phase, MV-MAE pre-training,
updates with demos, evaluation with video, reconstruction images, checkpoints,
and wandb logging (offline).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

import vis
from config import load_config
from trainer import Trainer, evaluate_policy


class FakeHiResRecorder:
    """Stand-in for env 0's paper camera: writes env 0's stereo view into a hires.width x hires.height video."""

    def __init__(self, env, path, hcfg, fps):
        self.env, self.path, self.hcfg, self.fps = env, Path(path), hcfg, fps
        self.writer = None

    def start(self):
        self.writer = vis.Mp4Writer(self.path, self.fps, self.hcfg.crf, self.hcfg.preset)

    def capture(self):
        img = vis.stereo_frame(self.env.images()[0], 1)
        frame = np.zeros((self.hcfg.height, self.hcfg.width, 3), dtype=np.uint8)
        h, w = min(img.shape[0], frame.shape[0]), min(img.shape[1], frame.shape[1])
        frame[:h, :w] = img[:h, :w]
        self.writer.append(frame)

    def stop(self):
        self.writer.close()
        return {"main": self.path} if self.writer.frames else {}


class FakeStereoEnv:
    """A coloured square the agent can move; the right eye sees it shifted by a fixed disparity."""

    def __init__(self, num_envs=4, size=32, episode=20):
        self.num_envs, self.size, self.max_episode_steps = num_envs, size, episode
        self.device = torch.device("cpu")
        self.action_dim = 7
        self.obs_shape = (2, 3, size, size)
        self.step_dt = 0.04
        self.pos = torch.zeros(num_envs, 2)
        self.t = torch.zeros(num_envs, dtype=torch.long)
        self.light, self.cam_shift = 1.0, 0  # changed by the robustness suite's fake scene changes
        self.steps = 0  # like Isaac Lab's common_step_counter
        self.reset_log: list[torch.Tensor] = []

    def _reset(self, ids):
        self.pos[ids] = torch.rand(len(ids), 2) * (self.size - 8)
        self.t[ids] = 0

    def images(self):
        img = torch.full((self.num_envs, 2, 3, self.size, self.size), 40, dtype=torch.uint8)
        for e in range(self.num_envs):
            x, y = self.pos[e].long().tolist()
            x = min(max(x + self.cam_shift, 0), self.size - 6)
            img[e, 0, 0, y:y + 6, x:x + 6] = 220
            img[e, 1, 0, y:y + 6, max(0, x - 2):x + 4] = 220
        return (img.float() * self.light).clamp(0, 255).to(torch.uint8)

    def reset(self):
        self._reset(torch.arange(self.num_envs))
        self.reset_log.append(self.pos.clone())
        return self.images()

    # ---- fake versions of StereoNeedleEnv's evaluation-time tools
    def training_state(self):
        return {"steps": self.steps}

    def restore_training_state(self, state):
        self.steps = state["steps"]

    def sim_perturbation(self, family, severity):
        if family == "light_intensity":
            self.light = 1.0 + severity
        elif family == "camera_azimuth":
            self.cam_shift = int(severity)  # px
        elif family == "shadow_light":
            return lambda: None  # "applied", but nothing visible changes: the suite must not report it
        else:
            raise NotImplementedError("fake env cannot tilt its camera")

        def restore():
            self.light, self.cam_shift = 1.0, 0

        return restore

    def hires_recorder(self, hcfg, video_dir, env_step, fps):
        return FakeHiResRecorder(self, Path(video_dir) / f"hires_eval_{env_step}.mp4", hcfg, fps)

    proprio_dim = 3

    def proprio(self):
        return torch.cat([self.pos / self.size, (self.t.float() / self.max_episode_steps)[:, None]], dim=-1)

    def needle_xy_local(self):
        return (self.pos - self.size / 2) / 1000.0

    def step(self, action):
        self.pos = (self.pos + action[:, :2] * 2).clamp(0, self.size - 8)
        self.t += 1
        self.steps += 1
        dist = (self.pos - self.size / 2).norm(dim=1)
        reward = 1.0 - torch.tanh(dist / 10)
        term = torch.rand(self.num_envs) < 0.01
        trunc = (self.t >= self.max_episode_steps) & ~term
        done = term | trunc
        success_now = dist < 3
        final_success = success_now.clone()
        final_goal = dist.clone()
        log = {"Episode_Reward/reach": torch.tensor(0.5), "Episode_Termination/time_out": 1.0, "not_scalar": [1, 2]}
        if done.any():
            self._reset(done.nonzero().squeeze(-1))
        info = {"final_success": final_success, "final_goal_distance": final_goal, "success_now": success_now,
                "needle_lifted": dist < 8, "log": log if bool(done.any()) else {}}
        return self.images(), reward, term, trunc, info


def test_training_loop_runs_end_to_end(tmp_path):
    os.environ["WANDB_SILENT"] = "true"
    cfg = load_config("configs/needle_tissue.yaml", [
        "env.num_envs=4", "camera.width=32", "camera.height=32", "mvmae.patch_size=8", "mvmae.embed_dim=32",
        "mvmae.encoder_depth=1", "mvmae.decoder_depth=1", "mvmae.decoder_dim=32", "agent.batch_size=16",
        "agent.hidden_dim=32", "agent.feature_dim=8", "agent.amp=false", "train.total_env_steps=600",
        "train.seed_env_steps=100", "train.mae_pretrain_updates=5", "train.updates_per_env_step=0.25",
        "train.replay_capacity=400", "train.replay_device=cpu", "train.eval_every_env_steps=400",
        "train.checkpoint_every_env_steps=500", "agent.proprio=true", "train.freeze_encoder_at=520", f"train.run_dir={tmp_path}", "log.log_every_updates=10",
        "log.recon_every_updates=20", "log.wandb_mode=offline", "log.run_name=test",
    ])
    env = FakeStereoEnv(num_envs=4, size=32)
    # demos in the recorder's format
    n = 60
    demo = dict(obs=torch.randint(0, 256, (n, 2, 3, 32, 32), dtype=torch.uint8), action=torch.rand(n, 7) * 2 - 1,
                reward=torch.rand(n), terminated=torch.zeros(n, dtype=torch.bool),
                truncated=torch.zeros(n, dtype=torch.bool), first=torch.zeros(n, dtype=torch.bool), proprio=torch.rand(n, 3),
                meta={"num_episodes": 3})
    demo["first"][[0, 20, 40]] = True
    demo["truncated"][[19, 39, 59]] = True
    trainer = Trainer(cfg, env, device="cpu", demo_data=demo)
    trainer.train()
    run = tmp_path / "test"
    assert (run / "ckpt_final.pt").exists() and (run / "ckpt_latest.pt").exists() and (run / "ckpt_best.pt").exists()
    assert trainer.best_score is not None
    assert trainer.agent.encoder_frozen and (run / "ckpt_frozen.pt").exists()
    assert list((run / "videos").glob("eval_*.mp4")), "no evaluation video written"
    m, frames, fail_frames = evaluate_policy(env, trainer.agent, 0, record_video=True, video_scale=1)
    fail = [m[f"eval/fail/{b}"] for b in ("never_lifted", "lifted_dropped", "missed_goal", "glitch_cut")]
    assert abs(sum(fail) + m["eval/success"] - 1.0) < 1e-6, "buckets must cover every failed episode exactly once"
    assert frames and ("eval/success_center" in m or "eval/success_edge" in m)
    assert bool(fail_frames) == (m["eval/success"] < 1.0)
    assert list((run / "reconstructions").glob("recon_*.png")), "no reconstruction image written"
    assert trainer.agent.num_updates > 0
    ckpt = torch.load(run / "ckpt_final.pt", map_location="cpu")
    assert ckpt["env_steps"] >= 600 and "mvmae" in ckpt["agent"]


def test_training_loop_with_the_raw_pixel_baseline(tmp_path):
    os.environ["WANDB_SILENT"] = "true"
    cfg = load_config("configs/needle_tissue.yaml", [
        "agent.encoder=pixels", "agent.pixel_downsample=2", "env.num_envs=4", "camera.width=32", "camera.height=32",
        "agent.batch_size=16", "agent.hidden_dim=32", "agent.feature_dim=8", "agent.amp=false",
        "train.total_env_steps=500", "train.seed_env_steps=100", "train.replay_capacity=400", "train.replay_device=cpu",
        "train.eval_every_env_steps=400", "train.checkpoint_every_env_steps=500", "agent.proprio=true",
        f"train.run_dir={tmp_path}", "log.log_every_updates=10", "log.recon_every_updates=20",
        "log.wandb_mode=offline", "log.run_name=pix",
    ])
    trainer = Trainer(cfg, FakeStereoEnv(num_envs=4, size=32), device="cpu")
    trainer.train()
    assert (tmp_path / "pix" / "ckpt_final.pt").exists() and trainer.agent.num_updates > 0
    assert trainer.agent.encoder_kind == "pixels" and trainer.agent.repr_dim == 3 * 2 * 3 * 16 * 16


SMALL = ["env.num_envs=4", "camera.width=32", "camera.height=32", "mvmae.patch_size=8", "mvmae.embed_dim=32",
         "mvmae.encoder_depth=1", "mvmae.decoder_depth=1", "mvmae.decoder_dim=32", "agent.batch_size=16",
         "agent.hidden_dim=32", "agent.feature_dim=8", "agent.amp=false", "train.total_env_steps=600",
         "train.seed_env_steps=100", "train.mae_pretrain_updates=5", "train.replay_capacity=400",
         "train.replay_device=cpu", "train.eval_every_env_steps=400", "train.checkpoint_every_env_steps=500",
         "agent.proprio=true", "log.log_every_updates=10", "log.recon_every_updates=20"]


def test_training_with_robustness_suite_and_hires_video(tmp_path):
    """Every evaluation output once, for both the MV-MAE agent and the CNN baseline: robust JSON, per-condition
    videos, figure, env 0's paper-camera video, and the wandb log (offline)."""
    os.environ["WANDB_SILENT"] = "true"
    for encoder in ("mvmae", "cnn"):
        cfg = load_config("configs/needle_tissue.yaml", SMALL + [
            f"agent.encoder={encoder}", "robust.every_env_steps=400", "robust.episodes_per_condition=4",
            "hires.width=128", "hires.height=64", f"train.run_dir={tmp_path}", "log.wandb_mode=offline",
            f"log.run_name={encoder}"])
        trainer = Trainer(cfg, FakeStereoEnv(num_envs=4, size=32), device="cpu")
        trainer.train()
        run = tmp_path / encoder
        res = json.loads((run / "robust_400.json").read_text())
        assert res["env_step"] == 400 and res["episodes_per_condition"] == 4
        c = res["conditions"]
        assert c["nominal"]["skipped"] is None and len(c) == 1 + len(trainer.robust_conditions)
        for name in ("view_drop_left", "gaussian_noise_0.10", "occlusion_both_40", "hue_shift_0.10", "blur_2",
                     "frame_drop", "frozen_frames_2", "camera_azimuth_-10", "light_dome_1.5"):
            assert c[name]["skipped"] is None and "success_rel" in c[name], name
        assert c["camera_elevation_+5"]["skipped"]  # the fake env cannot do it: skipped with the reason
        for name in ("gaussian_noise_0.10", "frozen_frames_2", "camera_azimuth_+10", "light_dome_0.5"):
            assert (run / "videos" / f"robust_{name}_400.mp4").exists(), name
        assert (run / "videos" / "hires_eval_400.mp4").exists()
        assert (run / "robust" / "robust_400_curves.png").exists() and (run / "robust" / "robust_400_bars.png").exists()
        assert list((run / "videos").glob("eval_400.mp4"))
        assert (run / "ckpt_final.pt").exists()
        assert trainer.agent.encoder_kind == encoder


def test_robustness_suite_and_hires_video_do_not_change_training(tmp_path):
    """Same seed with and without the suite + paper video: bit-identical weights afterwards."""
    weights = {}
    for name, extra in (("suite_off", ["robust.enabled=false", "hires.enabled=false"]),
                        ("suite_on", ["robust.every_env_steps=400", "robust.episodes_per_condition=8",
                                "hires.width=128", "hires.height=64"])):
        cfg = load_config("configs/needle_tissue.yaml", SMALL + extra + [
            f"train.run_dir={tmp_path}", "log.wandb_mode=disabled", f"log.run_name={name}"])
        trainer = Trainer(cfg, FakeStereoEnv(num_envs=4, size=32), device="cpu")
        trainer.train()
        weights[name] = trainer.agent.state_dict()
    assert (tmp_path / "suite_on" / "robust_400.json").exists() and not (tmp_path / "suite_off" / "robust_400.json").exists()
    for part in ("mvmae", "actor", "critic", "critic_target"):
        a, b = weights["suite_off"][part], weights["suite_on"][part]
        assert all(torch.equal(a[k], b[k]) for k in a), f"{part} differs when the robustness suite runs"
    assert weights["suite_off"]["num_updates"] == weights["suite_on"]["num_updates"]


def test_training_loop_with_the_cnn_baseline(tmp_path):
    os.environ["WANDB_SILENT"] = "true"
    cfg = load_config("configs/needle_tissue.yaml", SMALL + [
        "agent.encoder=cnn", "train.freeze_encoder_at=520", f"train.run_dir={tmp_path}", "log.wandb_mode=disabled",
        "log.run_name=cnn"])
    trainer = Trainer(cfg, FakeStereoEnv(num_envs=4, size=32), device="cpu")
    trainer.train()
    assert trainer.agent.encoder_kind == "cnn" and trainer.agent.num_updates > 0 and trainer.agent.encoder_frozen
    ckpt = torch.load(tmp_path / "cnn" / "ckpt_final.pt", map_location="cpu")
    assert ckpt["agent"]["encoder_kind"] == "cnn" and "encoder" in ckpt["agent"]
    assert not list((tmp_path / "cnn").glob("reconstructions/*.png"))  # no MV-MAE, no reconstruction images
