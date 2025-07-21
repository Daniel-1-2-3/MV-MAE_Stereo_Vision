from franka import FrankaEnv  # Replace with actual import path
import numpy as np
import os

# Create the environment
env = FrankaEnv(
    model_path=os.path.join(os.getcwd(), "pick_place.xml"),  # <-- Update this path!
    render_mode="human",                   # Use "rgb_array" if no GUI
)

# Reset environment
obs, info = env.reset()
print("Initial observation:", obs)

# Run a short loop
for step in range(50):
    action = env.action_space.sample()  # Sample a random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step} | Reward: {reward} | Terminated: {terminated}")

    if terminated or truncated:
        print("Episode finished.")
        obs, info = env.reset()

env.close()
