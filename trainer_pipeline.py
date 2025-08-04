from tqdm import tqdm
from SawyerSim.stereo_env import SawyerReachEnvV3
import numpy as np
from SawyerSim.sac import Agent

if __name__ == '__main__':
    env = SawyerReachEnvV3(render_mode="rgb_array")
    agent = Agent(env=env)

    for i in tqdm(range(100)):
        observation, info = env.reset()
        truncated = False
        while not truncated:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            agent.remember(observation, action, reward, observation_, truncated)
            agent.learn()
            
            observation = observation_
            env.render()
            
        print('Episode', i, 'Reward', reward)