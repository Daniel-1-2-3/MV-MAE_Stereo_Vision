import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,   # e.g. (H, 2W, 3) if channels_last True
        act_dim: int,
        device: torch.device,
        *,
        store_on_gpu: bool = False,
        channels_last: bool = False,
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

    @staticmethod
    def _ensure_batch_dim(x: torch.Tensor, want_ndim: int, name: str) -> torch.Tensor:
        """
        If x is missing a leading batch dim, add it.
        Example: obs (H,W,C) -> (1,H,W,C)
        """
        if x.ndim == want_ndim - 1:
            return x.unsqueeze(0)
        return x

    @staticmethod
    def _as_1d(x: torch.Tensor, name: str) -> torch.Tensor:
        """
        Ensure scalar fields are shape (B,), not (B,1) or ().
        """
        if x.ndim == 0:
            return x.view(1)
        if x.ndim == 2 and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x

    @torch.no_grad()
    def add_batch(self, obs_u8, action, reward, discount, next_obs_u8, done):
        """
        Insert a batch of transitions.

        Expected:
          obs_u8:      [B, *obs_shape] uint8
          next_obs_u8: [B, *obs_shape] uint8
          action:      [B, act_dim] float32
          reward:      [B] float32
          discount:    [B] float32
          done:        [B] bool
        """
        # Convert to tensors if caller passed numpy/etc.
        if not torch.is_tensor(obs_u8):      obs_u8 = torch.as_tensor(obs_u8)
        if not torch.is_tensor(next_obs_u8): next_obs_u8 = torch.as_tensor(next_obs_u8)
        if not torch.is_tensor(action):     action = torch.as_tensor(action)
        if not torch.is_tensor(reward):     reward = torch.as_tensor(reward)
        if not torch.is_tensor(discount):   discount = torch.as_tensor(discount)
        if not torch.is_tensor(done):       done = torch.as_tensor(done)

        # Normalize dtypes
        if obs_u8.dtype != torch.uint8:
            obs_u8 = obs_u8.to(torch.uint8)
        if next_obs_u8.dtype != torch.uint8:
            next_obs_u8 = next_obs_u8.to(torch.uint8)

        action = action.to(torch.float32)
        reward = reward.to(torch.float32)
        discount = discount.to(torch.float32)
        done = done.to(torch.bool)

        # Allow missing batch dim for convenience
        obs_u8 = self._ensure_batch_dim(obs_u8, want_ndim=self.obs.ndim, name="obs_u8")
        next_obs_u8 = self._ensure_batch_dim(next_obs_u8, want_ndim=self.next_obs.ndim, name="next_obs_u8")
        action = self._ensure_batch_dim(action, want_ndim=2, name="action")

        reward = self._as_1d(reward, "reward")
        discount = self._as_1d(discount, "discount")
        done = self._as_1d(done, "done")

        # Infer B from obs (since that's what we actually store)
        B = int(obs_u8.shape[0])
        if B <= 0:
            raise ValueError("add_batch got empty batch")

        # Hard shape checks (THIS is what will catch your current bug cleanly)
        def _shape(x): return tuple(x.shape)

        problems = []
        if next_obs_u8.shape[0] != B: problems.append(f"next_obs_u8 B={next_obs_u8.shape[0]} != obs_u8 B={B}")
        if action.shape[0] != B:      problems.append(f"action B={action.shape[0]} != obs_u8 B={B}")
        if reward.shape[0] != B:      problems.append(f"reward B={reward.shape[0]} != obs_u8 B={B}")
        if discount.shape[0] != B:    problems.append(f"discount B={discount.shape[0]} != obs_u8 B={B}")
        if done.shape[0] != B:        problems.append(f"done B={done.shape[0]} != obs_u8 B={B}")

        # Check obs shapes match buffer layout
        if tuple(obs_u8.shape[1:]) != tuple(self.obs.shape[1:]):
            problems.append(f"obs_u8 trailing shape {_shape(obs_u8)[1:]} != expected {tuple(self.obs.shape[1:])}")
        if tuple(next_obs_u8.shape[1:]) != tuple(self.next_obs.shape[1:]):
            problems.append(f"next_obs_u8 trailing shape {_shape(next_obs_u8)[1:]} != expected {tuple(self.next_obs.shape[1:])}")

        # Check action last dim
        if action.ndim != 2 or action.shape[1] != self.action.shape[1]:
            problems.append(f"action shape {_shape(action)} != expected (B,{self.action.shape[1]})")

        if problems:
            raise RuntimeError(
                "ReplayBuffer.add_batch got inconsistent shapes:\n"
                + "\n".join(f"  - {p}" for p in problems)
                + "\n\nGot:\n"
                + f"  obs_u8      {tuple(obs_u8.shape)} {obs_u8.dtype} {obs_u8.device}\n"
                + f"  next_obs_u8 {tuple(next_obs_u8.shape)} {next_obs_u8.dtype} {next_obs_u8.device}\n"
                + f"  action      {tuple(action.shape)} {action.dtype} {action.device}\n"
                + f"  reward      {tuple(reward.shape)} {reward.dtype} {reward.device}\n"
                + f"  discount    {tuple(discount.shape)} {discount.dtype} {discount.device}\n"
                + f"  done        {tuple(done.shape)} {done.dtype} {done.device}\n"
            )

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
        assert self.size > 0, "Replay is empty"
        idx = torch.randint(0, self.size, (batch_size,), device=self.obs.device)

        obs_u8 = self.obs[idx]
        next_obs_u8 = self.next_obs[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        discount = self.discount[idx]
        done = self.done[idx]

        if obs_u8.device != self.device:
            obs_u8 = obs_u8.to(self.device, non_blocking=True)
            next_obs_u8 = next_obs_u8.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            reward = reward.to(self.device, non_blocking=True)
            discount = discount.to(self.device, non_blocking=True)
            done = done.to(self.device, non_blocking=True)

        return obs_u8, action, reward, discount, next_obs_u8, done
