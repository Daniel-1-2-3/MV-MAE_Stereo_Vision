"""End-to-end run of the training loop on the CPU with a fake stereo environment.

Exercises everything except Isaac Sim: seeding phase, MV-MAE pre-training,
updates with demos, evaluation with video, reconstruction images, checkpoints,
and wandb logging (offline).
"""

import os

import torch

from config import load_config
from trainer import Trainer


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

    def _reset(self, ids):
        self.pos[ids] = torch.rand(len(ids), 2) * (self.size - 8)
        self.t[ids] = 0

    def images(self):
        img = torch.full((self.num_envs, 2, 3, self.size, self.size), 40, dtype=torch.uint8)
        for e in range(self.num_envs):
            x, y = self.pos[e].long().tolist()
            img[e, 0, 0, y:y + 6, x:x + 6] = 220
            img[e, 1, 0, y:y + 6, max(0, x - 2):x + 4] = 220
        return img

    def reset(self):
        self._reset(torch.arange(self.num_envs))
        return self.images()

    def step(self, action):
        self.pos = (self.pos + action[:, :2] * 2).clamp(0, self.size - 8)
        self.t += 1
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
                "log": log if bool(done.any()) else {}}
        return self.images(), reward, term, trunc, info


def test_training_loop_runs_end_to_end(tmp_path):
    os.environ["WANDB_SILENT"] = "true"
    cfg = load_config("configs/needle_tissue.yaml", [
        "env.num_envs=4", "camera.width=32", "camera.height=32", "mvmae.patch_size=8", "mvmae.embed_dim=32",
        "mvmae.encoder_depth=1", "mvmae.decoder_depth=1", "mvmae.decoder_dim=32", "agent.batch_size=16",
        "agent.hidden_dim=32", "agent.feature_dim=8", "agent.amp=false", "train.total_env_steps=600",
        "train.seed_env_steps=100", "train.mae_pretrain_updates=5", "train.updates_per_env_step=0.25",
        "train.replay_capacity=400", "train.replay_device=cpu", "train.eval_every_env_steps=400",
        "train.checkpoint_every_env_steps=500", "train.freeze_encoder_at=520", f"train.run_dir={tmp_path}", "log.log_every_updates=10",
        "log.recon_every_updates=20", "log.wandb_mode=offline", "log.run_name=test",
    ])
    env = FakeStereoEnv(num_envs=4, size=32)
    # demos in the recorder's format
    n = 60
    demo = dict(obs=torch.randint(0, 256, (n, 2, 3, 32, 32), dtype=torch.uint8), action=torch.rand(n, 7) * 2 - 1,
                reward=torch.rand(n), terminated=torch.zeros(n, dtype=torch.bool),
                truncated=torch.zeros(n, dtype=torch.bool), first=torch.zeros(n, dtype=torch.bool),
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
    assert list((run / "reconstructions").glob("recon_*.png")), "no reconstruction image written"
    assert trainer.agent.num_updates > 0
    ckpt = torch.load(run / "ckpt_final.pt", map_location="cpu")
    assert ckpt["env_steps"] >= 600 and "mvmae" in ckpt["agent"]
