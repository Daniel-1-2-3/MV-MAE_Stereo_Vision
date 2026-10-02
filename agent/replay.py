"""Replay storage for vectorised environments, kept as PyTorch tensors.

Layout is time-major: row t holds, for every parallel env e,
    obs[t, e]    - the latest single stereo frame seen before acting (uint8, V x 3 x H x W)
    action[t, e] - the action taken
    reward[t, e] - the reward received for it
    term[t, e]   - the episode ended for real after this action (needle dropped)
    trunc[t, e]  - the episode was cut off after this action (time limit / evaluation);
                   the next row then belongs to a new episode, and the true next
                   frame is unknown
    first[t, e]  - obs[t, e] is the first frame of an episode
    proprio[t, e]- the robot's own state when obs[t, e] was taken (only with proprio_dim > 0)

Only single frames are stored; frame stacks are rebuilt at sampling time
(repeating the first frame of an episode, like a frame-stack wrapper does), so
memory is ~55 KB per transition at 2 x 96 x 96 RGB.

n-step targets never cross episode boundaries:
    * termination inside the window: rewards up to it, then discount 0;
    * truncation inside the window: stop *before* the truncated transition and
      bootstrap from the last frame that is known to belong to the episode;
    * a transition that is itself truncated has no usable successor and is
      never sampled as a start.
"""

from __future__ import annotations

import torch

from .drqv2 import Batch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        frame_stack: int,
        nstep: int,
        gamma: float,
        device,
        proprio_dim: int = 0,
    ):
        self.N = num_envs
        self.P = proprio_dim
        self.F = frame_stack
        self.n = nstep
        self.gamma = gamma
        self.device = torch.device(device)
        self.R = max(capacity // num_envs, frame_stack + nstep + 2)
        self.obs = torch.zeros((self.R, num_envs, *obs_shape), dtype=torch.uint8, device=self.device)
        self.action = torch.zeros((self.R, num_envs, action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((self.R, num_envs), dtype=torch.float32, device=self.device)
        self.term = torch.zeros((self.R, num_envs), dtype=torch.bool, device=self.device)
        self.trunc = torch.zeros((self.R, num_envs), dtype=torch.bool, device=self.device)
        self.first = torch.zeros((self.R, num_envs), dtype=torch.bool, device=self.device)
        self.proprio = torch.zeros((self.R, num_envs, proprio_dim), dtype=torch.float32, device=self.device)
        self.t = 0  # rows written so far (global index of the next row)

    # ------------------------------------------------------------- writing
    def add(self, obs, action, reward, term, trunc, first, proprio=None) -> None:
        i = self.t % self.R
        if self.P > 0:
            if proprio is None:
                raise ValueError("this replay buffer stores the robot state: pass proprio")
            self.proprio[i] = proprio.to(self.device, non_blocking=True).float()
        self.obs[i] = obs.to(self.device, non_blocking=True)
        self.action[i] = action.to(self.device, non_blocking=True).float()
        self.reward[i] = reward.to(self.device, non_blocking=True).float()
        self.term[i] = term.to(self.device, non_blocking=True).bool()
        self.trunc[i] = trunc.to(self.device, non_blocking=True).bool()
        self.first[i] = first.to(self.device, non_blocking=True).bool()
        self.t += 1

    def mark_last_truncated(self) -> None:
        """Cut every running episode after the last stored row (e.g. before an evaluation reset)."""
        if self.t > 0:
            self.trunc[(self.t - 1) % self.R] = True

    @classmethod
    def from_episodes(cls, data: dict, frame_stack: int, nstep: int, gamma: float, device,
                      proprio_dim: int = 0) -> "ReplayBuffer":
        """Static buffer (one 'env' column) from record_demos.py output."""
        obs = data["obs"]
        total = obs.shape[0]
        if proprio_dim > 0:
            if "proprio" not in data or data["proprio"].shape[-1] != proprio_dim:
                raise ValueError("these demos have no robot state (proprio); record them again with this version")
        buf = cls(total, 1, tuple(obs.shape[1:]), data["action"].shape[-1], frame_stack, nstep, gamma, device, proprio_dim)
        if proprio_dim > 0:
            buf.proprio[:, 0] = data["proprio"].to(buf.device).float()
        if buf.R != total:
            raise ValueError(f"demo file too short ({total} transitions)")
        buf.obs[:, 0] = obs.to(buf.device)
        buf.action[:, 0] = data["action"].to(buf.device).float()
        buf.reward[:, 0] = data["reward"].to(buf.device).float()
        buf.term[:, 0] = data["terminated"].to(buf.device).bool()
        buf.trunc[:, 0] = data["truncated"].to(buf.device).bool()
        buf.first[:, 0] = data["first"].to(buf.device).bool()
        buf.t = total
        return buf

    # ------------------------------------------------------------ sampling
    def __len__(self) -> int:
        return min(self.t, self.R) * self.N

    def _range(self) -> tuple[int, int]:
        """Inclusive range of global row indices usable as window starts."""
        oldest = max(0, self.t - self.R)
        return oldest + self.F - 1, self.t - 1 - self.n

    def can_sample(self) -> bool:
        lo, hi = self._range()
        return hi >= lo

    def _stack(self, rows: torch.Tensor, envs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame-stack row indices (B, F), oldest first; walks back until an episode's first frame."""
        idx = [rows]
        cur = rows
        for _ in range(self.F - 1):
            at_first = self.first[cur % self.R, envs]
            cur = torch.where(at_first, cur, cur - 1)
            idx.append(cur)
        idx = torch.stack(idx[::-1], dim=1)
        return idx, self.obs[idx % self.R, envs[:, None]]

    def sample(self, batch_size: int, is_demo: bool = False) -> Batch:
        lo, hi = self._range()
        if hi < lo:
            raise RuntimeError("not enough data in the replay buffer to sample")
        dev = self.device
        rows = torch.empty(0, dtype=torch.long, device=dev)
        envs = torch.empty(0, dtype=torch.long, device=dev)
        for _ in range(100):  # rejection sampling of starts that are themselves truncated (rare)
            need = batch_size - rows.numel()
            if need <= 0:
                break
            r = torch.randint(lo, hi + 1, (2 * need,), device=dev)
            e = torch.randint(0, self.N, (2 * need,), device=dev)
            ok = ~self.trunc[r % self.R, e]
            rows = torch.cat([rows, r[ok][:need]])
            envs = torch.cat([envs, e[ok][:need]])
        if rows.numel() < batch_size:
            raise RuntimeError("could not sample valid transitions (is every stored transition truncated?)")

        # n-step return
        ret = torch.zeros(batch_size, device=dev)
        disc = torch.ones(batch_size, device=dev)
        active = torch.ones(batch_size, dtype=torch.bool, device=dev)
        terminal = torch.zeros(batch_size, dtype=torch.bool, device=dev)
        m = torch.full((batch_size,), self.n, dtype=torch.long, device=dev)
        for k in range(self.n):
            i = (rows + k) % self.R
            te = self.term[i, envs]
            tr = self.trunc[i, envs] & ~te
            stop_before = active & tr
            m = torch.where(stop_before, torch.full_like(m, k), m)
            active = active & ~tr
            ret = ret + torch.where(active, disc * self.reward[i, envs], torch.zeros_like(ret))
            disc = torch.where(active, disc * self.gamma, disc)
            stop_after = active & te
            m = torch.where(stop_after, torch.full_like(m, k + 1), m)
            terminal = terminal | stop_after
            active = active & ~te
        discount = torch.where(terminal, torch.zeros_like(disc), disc)

        rows_stack, obs = self._stack(rows, envs)
        _, next_obs = self._stack(rows + m, envs)

        # Reward of the transition into each stacked frame (for MV-MAE reward prediction).
        oldest = max(0, self.t - self.R)
        prev = rows_stack - 1
        frame_valid = ~self.first[rows_stack % self.R, envs[:, None]] & (prev >= oldest)
        frame_reward = self.reward[prev % self.R, envs[:, None]]

        return Batch(
            obs=obs,
            action=self.action[rows % self.R, envs],
            reward=ret,
            discount=discount,
            next_obs=next_obs,
            is_demo=torch.full((batch_size,), is_demo, dtype=torch.bool, device=dev),
            frame_reward=frame_reward,
            frame_reward_valid=frame_valid,
            proprio=self.proprio[rows % self.R, envs],
            next_proprio=self.proprio[(rows + m) % self.R, envs],
        )


def concat_batches(a: Batch, b: Batch, device) -> Batch:
    return Batch(*[torch.cat([x.to(device), y.to(device)], dim=0) for x, y in zip(vars(a).values(), vars(b).values())])


class FrameStacker:
    """Keeps the last F frames per env for acting; pads a new episode with its first frame."""

    def __init__(self, num_envs: int, frame_stack: int, obs_shape: tuple[int, ...], device):
        self.F = frame_stack
        self.frames = torch.zeros((num_envs, frame_stack, *obs_shape), dtype=torch.uint8, device=device)

    def reset(self, obs: torch.Tensor) -> torch.Tensor:
        self.frames[:] = obs[:, None]
        return self.frames

    def step(self, obs: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        self.frames = torch.roll(self.frames, shifts=-1, dims=1)
        self.frames[:, -1] = obs
        if bool(first.any()):
            self.frames[first] = obs[first][:, None]
        return self.frames
