import os
os.environ["MUJOCO_GL"] = "egl"

from tqdm import tqdm
from SawyerSim.sawyer_stereo_env import SawyerReachEnvV3
from SawyerSim.custom_sac import SAC

