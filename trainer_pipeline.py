from SawyerSim.stereo_env import StereoEnv
from stable_baselines3 import SAC

env = StereoEnv()
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000, log_interval=4)

