import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, image_shape, state_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Split storage for image and state parts
        self.image_memory = np.zeros((self.mem_size, *image_shape), dtype=np.uint8)
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.new_image_memory = np.zeros((self.mem_size, *image_shape), dtype=np.uint8)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, obs, action, reward, obs_, done):
        """
        `obs` and `obs_` should be dicts with keys:
        - 'image_observations': np.ndarray of shape image_shape (e.g. (H, W, C))
        - 'state_observations': np.ndarray of shape (state_dim,)
        """
        index = self.mem_cntr % self.mem_size

        self.image_memory[index] = obs["image_observation"]
        self.state_memory[index] = obs["state_observation"]
        self.new_image_memory[index] = obs_["image_observation"]
        self.new_state_memory[index] = obs_["state_observation"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        # Reconstruct observation dicts
        obs = {
            "image_observation": self.image_memory[batch],
            "state_observation": self.state_memory[batch]
        }
        obs_ = {
            "image_observation": self.new_image_memory[batch],
            "state_observation": self.new_state_memory[batch]
        }

        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return obs, actions, rewards, obs_, dones
