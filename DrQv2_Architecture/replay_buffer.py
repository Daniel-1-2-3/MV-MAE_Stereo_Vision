from __future__ import annotations
from dataclasses import dataclass
from flax import struct
from typing import Tuple

import jax
import jax.numpy as jnp

@struct.dataclass
class ReplayBufferState:
    # Ring storage: [capacity, ...]
    obs: jnp.ndarray        # [C, H, 2W, 3]?? NO: we store per-transition obs: [capacity, H, 2W, 3]
    action: jnp.ndarray     # [capacity, act_dim]
    reward: jnp.ndarray     # [capacity, 1] or [capacity,] (match your pipeline)
    discount: jnp.ndarray   # [capacity, 1] or [capacity,]
    done: jnp.ndarray       # [capacity,] bool

    # Ring pointers
    ptr: jnp.ndarray        # scalar int32: next write index (physical)
    size: jnp.ndarray       # scalar int32: number of valid items (<= capacity)
    capacity: int = struct.field(pytree_node=False) # static python int (kept in pytree as static via dataclass usage)

def rb_init(
    capacity: int,
    obs_shape: Tuple[int, int, int],     # (H, 2W, 3)
    action_shape: Tuple[int, ...],       # (act_dim,)
    reward_shape: Tuple[int, ...] = (1,),
    discount_shape: Tuple[int, ...] = (1,),
    # Pixels should be stored compactly (DrQ-style). Keep as uint8 in the buffer
    # and convert to float32 in [0,1] when sampling.
    obs_dtype=jnp.uint8,
    action_dtype=jnp.float32,
    reward_dtype=jnp.float32,
    discount_dtype=jnp.float32,
) -> ReplayBufferState:
    H, W2, C = obs_shape
    return ReplayBufferState(
        obs=jnp.zeros((capacity, H, W2, C), dtype=obs_dtype),
        action=jnp.zeros((capacity, *action_shape), dtype=action_dtype),
        reward=jnp.zeros((capacity, *reward_shape), dtype=reward_dtype),
        discount=jnp.ones((capacity, *discount_shape), dtype=discount_dtype),
        done=jnp.zeros((capacity,), dtype=jnp.bool_),
        ptr=jnp.array(0, jnp.int32),
        size=jnp.array(0, jnp.int32),
        capacity=capacity,
    )


def _float01_to_u8(x: jnp.ndarray) -> jnp.ndarray:
    """Convert float image in [0,1] to uint8 [0,255] safely."""
    x = jnp.clip(x * 255.0, 0.0, 255.0)
    return x.astype(jnp.uint8)


def _u8_to_float01(x: jnp.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    """Convert uint8 image [0,255] to float in [0,1]."""
    return (x.astype(dtype)) * (1.0 / 255.0)

def _ring_phys_index(rb: ReplayBufferState, t_logical: jnp.ndarray) -> jnp.ndarray:
    """
    Map logical time index t in [0, size) to physical ring index in [0, capacity).
    Oldest element is logical t=0.
    """
    # Oldest element sits at: start = (ptr - size) mod capacity
    start = (rb.ptr - rb.size) % rb.capacity
    return (start + t_logical) % rb.capacity

@jax.jit
def rb_add(
    rb: ReplayBufferState,
    obs_b: jnp.ndarray,       # [B, H, 2W, 3]  (your fused normalized output)
    action_b: jnp.ndarray,    # [B, act_dim]
    reward_b: jnp.ndarray,    # [B, ...]
    discount_b: jnp.ndarray,  # [B, ...]
    done_b: jnp.ndarray,      # [B] bool
) -> ReplayBufferState:
    """
    Fully jittable batched add. Writes B transitions into the ring.
    """
    B = obs_b.shape[0]
    idxs = (rb.ptr + jnp.arange(B, dtype=jnp.int32)) % rb.capacity

    # Store pixels compactly. Expect obs_b float in [0,1].
    obs_u8 = _float01_to_u8(obs_b)
    new_obs = rb.obs.at[idxs].set(obs_u8)
    new_action = rb.action.at[idxs].set(action_b)
    new_reward = rb.reward.at[idxs].set(reward_b)
    new_discount = rb.discount.at[idxs].set(discount_b)
    new_done = rb.done.at[idxs].set(done_b)

    new_ptr = (rb.ptr + B) % rb.capacity
    new_size = jnp.minimum(rb.size + B, rb.capacity)

    return ReplayBufferState(
        obs=new_obs,
        action=new_action,
        reward=new_reward,
        discount=new_discount,
        done=new_done,
        ptr=new_ptr,
        size=new_size,
        capacity=rb.capacity,
    )

def _valid_start(rb: ReplayBufferState, t: jnp.ndarray, nstep: int) -> jnp.ndarray:
    """
    Valid if:
      - t+nstep < size  (so next_obs exists)
      - no terminal inside [t, t+nstep-1] in the rollout window
    """
    # Must have indices t ... t+nstep (for next_obs at t+nstep)
    enough_room = (t + nstep) < rb.size

    # Check done in the rollout window [t, t+nstep-1]
    # We'll gather done via physical indices
    ts = t + jnp.arange(nstep, dtype=jnp.int32)  # length nstep
    phys = jax.vmap(lambda tt: _ring_phys_index(rb, tt))(ts)
    done_any = jnp.any(rb.done[phys])

    return enough_room & (~done_any)

def rb_sample(rb: ReplayBufferState, key: jax.Array, batch_size: int, nstep: int, gamma: float):
    """
    Fully jittable sampling.
    Returns (obs, action, reward_n, discount_n, next_obs) with batch dimension first.

    batch_size, nstep should be static for best compile behavior.
    """
    # If not enough data, return zeros (still jittable)
    min_needed = nstep + 1

    def empty():
        # shape-correct zero batch
        obs0 = jnp.zeros((batch_size,) + rb.obs.shape[1:], jnp.float32)
        act0 = jnp.zeros((batch_size,) + rb.action.shape[1:], rb.action.dtype)
        rew0 = jnp.zeros((batch_size,) + rb.reward.shape[1:], rb.reward.dtype)
        disc0 = jnp.ones((batch_size,) + rb.discount.shape[1:], rb.discount.dtype)
        nxt0 = jnp.zeros((batch_size,) + rb.obs.shape[1:], jnp.float32)
        return obs0, act0, rew0, disc0, nxt0

    def nonempty():
        # We do bounded retry inside jit by oversampling candidates and picking the first valid.
        # Oversample factor: 8 candidates per sample (tune if you have very short episodes).
        K = 8
        max_t = rb.size - min_needed  # max start logical index inclusive
        max_t = jnp.maximum(max_t, 0)

        # candidates: [batch_size, K]
        key1, key2 = jax.random.split(key)
        cand = jax.random.randint(key1, (batch_size, K), 0, max_t + 1, dtype=jnp.int32)

        # validity mask: [batch_size, K]
        valid = jax.vmap(lambda row: jax.vmap(lambda t: _valid_start(rb, t, nstep))(row))(cand)

        # pick first valid (or fallback to cand[:,0] if none valid)
        first_idx = jnp.argmax(valid, axis=1)  # returns 0 if all False
        has_any = jnp.any(valid, axis=1)

        chosen = cand[jnp.arange(batch_size), first_idx]
        chosen = jnp.where(has_any, chosen, cand[:, 0])

        # gather obs/action at start
        phys0 = jax.vmap(lambda t: _ring_phys_index(rb, t))(chosen)
        obs = _u8_to_float01(rb.obs[phys0], dtype=jnp.float32)
        action = rb.action[phys0]

        # n-step accumulate reward/discount exactly like your loop:
        #   reward += disc * step_reward
        #   disc *= step_discount * gamma
        def one_sample(t0):
            # gather sequences of length nstep for reward/discount
            ts = t0 + jnp.arange(nstep, dtype=jnp.int32)
            phys = jax.vmap(lambda tt: _ring_phys_index(rb, tt))(ts)

            r_seq = rb.reward[phys]      # [nstep, ...]
            d_seq = rb.discount[phys]    # [nstep, ...]

            r_acc = jnp.zeros_like(r_seq[0])
            d_acc = jnp.ones_like(d_seq[0])

            def body(i, carry):
                r, d = carry
                r = r + d * r_seq[i]
                d = d * d_seq[i] * gamma
                return (r, d)

            r_acc, d_acc = jax.lax.fori_loop(0, nstep, body, (r_acc, d_acc))
            return r_acc, d_acc

        reward_n, discount_n = jax.vmap(one_sample)(chosen)

        # next_obs at t+nstep
        phys_next = jax.vmap(lambda t: _ring_phys_index(rb, t + nstep))(chosen)
        next_obs = _u8_to_float01(rb.obs[phys_next], dtype=jnp.float32)

        return obs, action, reward_n, discount_n, next_obs

    return jax.lax.cond(rb.size < min_needed, empty, nonempty)

# Make batch_size and nstep static (recommended)
rb_sample_jit = jax.jit(rb_sample, static_argnames=("batch_size", "nstep"))

"""
# add transitions
rb = rb_add(rb, obs_fused, action, reward, discount, done)

# sample a training batch
rng, k = jax.random.split(rng)
obs_b, act_b, rew_b, disc_b, next_obs_b = rb_sample_jit(rb, k, batch_size=256, nstep=3, gamma=0.99)
"""