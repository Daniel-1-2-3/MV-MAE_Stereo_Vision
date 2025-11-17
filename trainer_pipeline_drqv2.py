import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
# On the cluster we want EGL. training.sh already sets MUJOCO_GL/MUJOCO_PLATFORM/PYOPENGL_PLATFORM.
# Only set sensible defaults if they are missing (e.g., when running locally).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_PLATFORM", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from pathlib import Path
import numpy as np
import torch
import time
import argparse
from dm_env import specs

import DrQv2_Architecture.utils as utils
from DrQv2_Architecture.logger import Logger
from DrQv2_Architecture.replay_buffer import ReplayBufferStorage, make_replay_loader
from DrQv2_Architecture.video import VideoRecorder
from DrQv2_Architecture.drqv2 import DrQV2Agent
from DrQv2_Architecture.env_wrappers import ExtendedTimeStepWrapper, ActionRepeatWrapper, FrameStackWrapper
from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from gymnasium.spaces import Box

torch.backends.cudnn.benchmark = True

class Workshop:
    def __init__(
        self,
        # General variables
        device: torch.device | None = None,
        # RL training
        buffer_size: int = 100_000,
        total_timesteps: int = 500_000,
        learning_starts: int = 10_000, # Turn on grads at step and start training at step n
        num_expl_steps: int = 5000, # Random actions (not policy determined) until step n
        episode_horizon: int = 300, # Truncates episode after 300 steps
        batch_size: int = 64,
        critic_target_tau: float = 0.001, # Soft-update for target critic
        update_every_steps: int = 2,
        stddev_schedule: str = 'linear(1.0,0.1,500000)', # Type of scheduler, value taken from cfgs/task/medium.yaml, stddev for exploration noise
        stddev_clip: int = 0.3, # How much to clip sampled action noise
        use_tb: bool = True,
        lr: float = 1e-4,
        discount: float = 0.99,
        action_repeat: int = 2,
        # MVMAE variables
        nviews: int = 2,
        mvmae_patch_size: int = 8, 
        mvmae_encoder_embed_dim: int = 256, 
        mvmae_decoder_embed_dim: int = 128,
        mvmae_encoder_heads: int = 16, 
        mvmae_decoder_heads: int = 16,
        masking_ratio: float = 0.75,
        coef_mvmae: float = 0.005,
        # Actor + critic 
        feature_dim: int = 100,
        hidden_dim: int = 1024,
        # Image specs
        render_mode: str = "human",
        in_channels: int = 3 * 3, # Number of frames stacked * 3 (RGB)
        img_h_size: int = 64,
        img_w_size: int = 64,
    ):  
        # Overall obs space expects frame stacked obs, sawyer env expects single image
        self.overall_obs_space = Box(low=np.float32(-4.0), high=np.float32(4.0), shape=(img_h_size, nviews * img_w_size, in_channels), dtype=np.float32)
        self.sawyer_env_obs_space = Box(low=np.float32(-4.0), high=np.float32(4.0), shape=(img_h_size, nviews * img_w_size, in_channels // 3), dtype=np.float32)
        self.action_space = Box(np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]), dtype=np.float32)
        self.device = device if (device is not None) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.num_expl_steps = num_expl_steps
        self.episode_horizon = episode_horizon
        self.batch_size = batch_size
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.lr = lr
        self.discount = discount
        self.action_repeat = action_repeat
        
        self.nviews = nviews
        self.mvmae_patch_size = mvmae_patch_size
        self.mvmae_encoder_embed_dim = mvmae_encoder_embed_dim
        self.mvmae_decoder_embed_dim = mvmae_decoder_embed_dim
        self.mvmae_encoder_heads = mvmae_encoder_heads
        self.mvmae_decoder_heads = mvmae_decoder_heads
        self.masking_ratio = masking_ratio
        self.coef_mvmae = coef_mvmae
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.render_mode = render_mode
        self.in_channels = in_channels
        self.img_h_size = img_h_size
        self.img_w_size = img_w_size
        
        self.work_dir = Path.cwd()
        print(f'Workspace: {self.work_dir}')
        self._global_step = 0
        self._global_episode = 0
        
        self.seed = 1
        utils.set_seed_everywhere(self.seed)
        
        self.agent = self.make_agent() # Make agent
        self.setup() # Make envs and setup replay buffer
        self.timer = utils.Timer()
        
        # Evaluation
        self.eval_every_frames = 10000
        self.num_eval_episodes = 10
        
    def make_agent(self):
        # DrQv2 agent takes action_shape as (A, ) tuple
        return DrQV2Agent(
            action_shape = (self.action_space.shape[0], ),
            device = self.device,
            lr = self.lr,

            nviews = self.nviews,
            mvmae_patch_size = self.mvmae_patch_size,
            mvmae_encoder_embed_dim = self.mvmae_encoder_embed_dim,
            mvmae_decoder_embed_dim = self.mvmae_decoder_embed_dim,
            mvmae_encoder_heads = self.mvmae_encoder_heads,
            mvmae_decoder_heads = self.mvmae_decoder_heads,
            in_channels = self.in_channels,
            img_h_size = self.img_h_size,
            img_w_size = self.img_w_size,
            masking_ratio = self.masking_ratio,
            coef_mvmae = self.coef_mvmae,
            
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,

            critic_target_tau = self.critic_target_tau,
            num_expl_steps = self.num_expl_steps,
            update_every_steps = self.update_every_steps,
            stddev_schedule = self.stddev_schedule,
            stddev_clip = self.stddev_clip,
            use_tb = self.use_tb
        )
    
    # SawyerReachEnvV3 with wrappers for frame stacking, action repeat, time_step formatting
    def make_env(self, render_mode):
        env = SawyerReachEnvV3(
            render_mode=render_mode, 
            img_height=self.img_h_size,
            img_width=self.img_w_size,
            max_path_length=self.episode_horizon,
            obs_space=self.sawyer_env_obs_space,
            action_space=self.action_space,
            discount=self.discount
        )
        env = FrameStackWrapper(env, num_frames=self.in_channels // 3)
        env = ActionRepeatWrapper(env, num_repeats=self.action_repeat)
        env = ExtendedTimeStepWrapper(env)
        return env
    
    def setup(self):
        self.logger = Logger(self.work_dir, use_tb=True)
        self.train_env = self.make_env("rgb_array")
        self.eval_env = self.make_env("rgb_array")
        
        data_specs = (
            specs.Array(self.overall_obs_space.shape, self.overall_obs_space.dtype, name="observation"),
            specs.Array(self.action_space.shape, self.action_space.dtype, name="action"),
            specs.Array((1,), np.float32, name="reward"),
            specs.Array((1,), np.float32, name="discount")
        )
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.buffer_size,
            self.batch_size, 1, False, 3, self.discount) # Samples 3 consecutive steps, frame stacking
        self._replay_iter = None
        
        self.video_recorder = VideoRecorder(self.work_dir) # Video recorder for eval frames
    
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.action_repeat

    @property
    def replay_iter(self): # An iterator of replay buffer dataloader
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, self.global_step, eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += float(time_step.reward[0])
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        train_until_step = utils.Until(self.total_timesteps, self.action_repeat) # True until self.action_repeat * self._global_step > self.total_timesteps
        seed_until_step = utils.Until(self.learning_starts, self.action_repeat) # True until self.action_repeat * self._global_step > self.learning_starts
        eval_every_step = utils.Every(self.eval_every_frames, self.action_repeat) # True for every self.eval_every_frames / self.action_repeat frames

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset() # time_step here is a ExtendedTimeStep object due to wrappers
        self.replay_storage.add(time_step)

        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # Wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # Reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)

                episode_step = 0
                episode_reward = 0

            # Try to evaluate, evals at intervals
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            t0 = time.perf_counter()
            # Sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, self.global_step, eval_mode=False)

            # Try to update the agent, will only update past self.learning_starts
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            t1 = time.perf_counter()
            print(f"Training step: {1/(t1 - t0):.2f} steps/sec, step {self.global_step}")

            # Take env step
            time_step = self.train_env.step(action)
            episode_reward += float(time_step.reward[0])
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

def get_args():
    parser = argparse.ArgumentParser()

    # General variables
    parser.add_argument("--device", type=str, default=None, help="Torch device string, e.g. 'cuda' or 'cpu'")
    # RL training
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--learning_starts", type=int, default=10_000, help="Start training (and using grads) at this step")
    parser.add_argument("--num_expl_steps", type=int, default=5000, help="Number of random action steps before policy actions")
    parser.add_argument("--episode_horizon", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--critic_target_tau", type=float, default=0.001)
    parser.add_argument("--update_every_steps", type=int, default=2)
    parser.add_argument("--stddev_schedule", type=str, default="linear(1.0,0.1,500000)")
    parser.add_argument("--stddev_clip", type=float, default=0.3)
    parser.add_argument("--use_tb", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--action_repeat", type=int, default=2)
    # MV-MAE variables
    parser.add_argument("--nviews", type=int, default=2)
    parser.add_argument("--mvmae_patch_size", type=int, default=8)
    parser.add_argument("--mvmae_encoder_embed_dim", type=int, default=256)
    parser.add_argument("--mvmae_decoder_embed_dim", type=int, default=128)
    parser.add_argument("--mvmae_encoder_heads", type=int, default=16)
    parser.add_argument("--mvmae_decoder_heads", type=int, default=16)
    parser.add_argument("--masking_ratio", type=float, default=0.75)
    parser.add_argument("--coef_mvmae", type=float, default=0.005)
    # Actor + Critic
    parser.add_argument("--feature_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    # Image specs
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--in_channels", type=int, default=3 * 3)
    parser.add_argument("--img_h_size", type=int, default=64)
    parser.add_argument("--img_w_size", type=int, default=64)

    return parser.parse_args()

if __name__ == '__main__':
    root_dir = Path.cwd()
    args = get_args()
    workspace = Workshop(**vars(args))
    workspace.train()