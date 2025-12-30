import torch

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,   # (C, H, W) or (H, W, C) - choose one and be consistent
        act_dim: int,
        device: torch.device,
        *,
        store_on_gpu: bool = False,
        channels_last: bool = False,  # True if obs is HWC; False if CHW
        pin_memory: bool = True,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.channels_last = bool(channels_last)

        storage_device = device if store_on_gpu else torch.device("cpu")

        # Images: uint8
        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device=storage_device)
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device=storage_device)

        # Scalar/vector fields
        self.action = torch.empty((capacity, act_dim), dtype=torch.float32, device=storage_device)
        self.reward = torch.empty((capacity,), dtype=torch.float32, device=storage_device)
        self.discount = torch.empty((capacity,), dtype=torch.float32, device=storage_device)
        self.done = torch.empty((capacity,), dtype=torch.bool, device=storage_device)

        # Optional pinning for faster H2D copies
        self._pin = (pin_memory and storage_device.type == "cpu")
        if self._pin:
            self.obs = self.obs.pin_memory()
            self.next_obs = self.next_obs.pin_memory()
            self.action = self.action.pin_memory()
            self.reward = self.reward.pin_memory()
            self.discount = self.discount.pin_memory()
            self.done = self.done.pin_memory()

        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(self, obs_u8, action, reward, discount, next_obs_u8, done):
        """
        Insert a batch of transitions.
        Shapes:
          obs_u8:      [B, *obs_shape] uint8
          action:      [B, act_dim] float32
          reward:      [B] float32
          discount:    [B] float32
          next_obs_u8: [B, *obs_shape] uint8
          done:        [B] bool
        """
        B = obs_u8.shape[0]
        assert B > 0

        # Ensure types (avoid accidental float images)
        if obs_u8.dtype != torch.uint8:
            obs_u8 = obs_u8.to(torch.uint8)
        if next_obs_u8.dtype != torch.uint8:
            next_obs_u8 = next_obs_u8.to(torch.uint8)

        # Move to storage device if needed
        storage_dev = self.obs.device
        if obs_u8.device != storage_dev:
            obs_u8 = obs_u8.to(storage_dev, non_blocking=True)
            next_obs_u8 = next_obs_u8.to(storage_dev, non_blocking=True)
            action = action.to(storage_dev, non_blocking=True)
            reward = reward.to(storage_dev, non_blocking=True)
            discount = discount.to(storage_dev, non_blocking=True)
            done = done.to(storage_dev, non_blocking=True)

        end = self.ptr + B
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.obs[sl].copy_(obs_u8)
            self.next_obs[sl].copy_(next_obs_u8)
            self.action[sl].copy_(action)
            self.reward[sl].copy_(reward)
            self.discount[sl].copy_(discount)
            self.done[sl].copy_(done)
        else:
            first = self.capacity - self.ptr
            second = B - first

            sl1 = slice(self.ptr, self.capacity)
            sl2 = slice(0, second)

            self.obs[sl1].copy_(obs_u8[:first])
            self.next_obs[sl1].copy_(next_obs_u8[:first])
            self.action[sl1].copy_(action[:first])
            self.reward[sl1].copy_(reward[:first])
            self.discount[sl1].copy_(discount[:first])
            self.done[sl1].copy_(done[:first])

            self.obs[sl2].copy_(obs_u8[first:])
            self.next_obs[sl2].copy_(next_obs_u8[first:])
            self.action[sl2].copy_(action[first:])
            self.reward[sl2].copy_(reward[first:])
            self.discount[sl2].copy_(discount[first:])
            self.done[sl2].copy_(done[first:])

        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.capacity, self.size + B)

    @torch.no_grad()
    def sample(self, batch_size: int):
        """
        Returns batch on self.device for training.
        obs/next_obs returned as uint8 with shape [B, H, 2W, 3] (channels-last),
        matching pick_env_v4.render_pixels().
        """
        assert self.size > 0, "Replay is empty"
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs.device)

        obs_u8 = self.obs[idx]
        next_obs_u8 = self.next_obs[idx]

        action = self.action[idx]
        reward = self.reward[idx]
        discount = self.discount[idx]
        done = self.done[idx]

        # If storage is CPU, push to GPU non-blocking
        if obs_u8.device != self.device:
            obs_u8 = obs_u8.to(self.device, non_blocking=True)
            next_obs_u8 = next_obs_u8.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            reward = reward.to(self.device, non_blocking=True)
            discount = discount.to(self.device, non_blocking=True)
            done = done.to(self.device, non_blocking=True)

        return obs_u8, action, reward, discount, next_obs_u8, done
