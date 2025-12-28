from __future__ import annotations

from typing import Any
from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class FrameStackInfo:
    frames: jax.Array  # (K,  H, W2, C)
    filled: jax.Array  # int32 scalar

@struct.dataclass
class FrameStackState:
    """Wrapper state that carries framestack without touching info."""
    inner: Any
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    info: Any
    fs: FrameStackInfo


class FrameStackWrapper:
    def __init__(self, env, num_frames: int):
        self._env = env
        self._K = int(num_frames)

    def _init_fs(self, obs: jnp.ndarray) -> FrameStackInfo:
        # obs comes in as (1,H,W2,C); store frames without the leading batch dim
        obs3 = obs[0]  # (H,W2,C)
        frames = jnp.broadcast_to(obs3, (self._K,) + obs3.shape)  # (K,H,W2,C)
        return FrameStackInfo(frames=frames, filled=jnp.int32(1))

    def _push(self, fs: FrameStackInfo, obs: jnp.ndarray) -> FrameStackInfo:
        # obs comes in as (1,H,W2,C); store without the leading batch dim
        obs3 = obs[0]  # (H,W2,C)
        frames = jnp.roll(fs.frames, shift=1, axis=0)
        frames = frames.at[0].set(obs3)
        filled = jnp.minimum(fs.filled + 1, jnp.int32(self._K))
        return FrameStackInfo(frames=frames, filled=filled)

    def _stack(self, fs: FrameStackInfo) -> jnp.ndarray:
        # fs.frames is (K,H,W2,C); output should stay (1,H,W2,K*C)
        x = jnp.concatenate(tuple(fs.frames[i] for i in range(self._K)), axis=-1)  # (H,W2,K*C)
        return x[None, ...]  # (1,H,W2,K*C)

    def reset(self, rng):
        st = self._env.reset(rng)
        fs = self._init_fs(st.obs)
        return FrameStackState(
            inner=st,
            obs=self._stack(fs),
            reward=st.reward,
            done=st.done,
            info=st.info,
            fs=fs,
        )

    def step(self, st: FrameStackState, action):
        st1 = self._env.step(st.inner, action)

        done1 = (st1.done > 0.5)
        fs1 = jax.lax.cond(
            done1,
            lambda: self._init_fs(st1.obs),
            lambda: self._push(st.fs, st1.obs),
        )

        return FrameStackState(
            inner=st1,
            obs=self._stack(fs1),
            reward=st1.reward,
            done=st1.done,
            info=st1.info,
            fs=fs1,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)