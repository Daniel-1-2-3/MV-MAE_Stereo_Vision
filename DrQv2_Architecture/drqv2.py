from MAE_Model.model import MAEModel

from typing import Tuple
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple

import jax
import optax
from flax.core import FrozenDict
from flax import struct

def _st_clamp(x, low=-1.0, high=1.0, eps=1e-6):
    clamped = jnp.clip(x, low + eps, high - eps)
    return x + jax.lax.stop_gradient(clamped - x)

def sample_tanh_gaussian(rng, loc, scale, clip=None, low=-1.0, high=1.0, eps=1e-6):
    noise = jax.random.normal(rng, loc.shape, dtype=loc.dtype)
    noise = noise * scale # scale
    if clip is not None:
        noise = jnp.clip(noise, -clip, clip)
    x = loc + noise
    x = _st_clamp(x, low=low, high=high, eps=eps) # straight-through clamp like Torch implementation
    return x

# obs for both actor and critic are encoder output z
class Actor(nn.Module):
    repr_dim: int
    action_shape: Tuple[int, ...]
    feature_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, std):
        # trunk: Linear -> LayerNorm -> Tanh
        h = nn.Dense(self.feature_dim, name="trunk_dense")(obs)
        h = nn.LayerNorm(name="trunk_ln")(h)
        h = jnp.tanh(h)

        # policy: Linear -> ReLU -> Linear -> ReLU -> Linear
        x = nn.Dense(self.hidden_dim, name="pi_fc1")(h)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="pi_fc2")(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_shape[0], name="pi_out")(x)

        mu = jnp.tanh(x)
        std = jnp.ones_like(mu) * std

        return mu, std

class Critic(nn.Module):
    repr_dim: int
    action_shape: Tuple[int, ...]
    feature_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        # trunk: Linear -> LayerNorm -> Tanh
        h = nn.Dense(self.feature_dim, name="trunk_dense")(obs)
        h = nn.LayerNorm(name="trunk_ln")(h)
        h = jnp.tanh(h)

        # torch.cat([h, action], dim=-1)
        h_action = jnp.concatenate([h, action], axis=-1)

        # Q1: Linear -> ReLU -> Linear -> ReLU -> Linear -> (1)
        q1 = nn.Dense(self.hidden_dim, name="q1_fc1")(h_action)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim, name="q1_fc2")(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1, name="q1_out")(q1)

        # Q2: Linear -> ReLU -> Linear -> ReLU -> Linear -> (1)
        q2 = nn.Dense(self.hidden_dim, name="q2_fc1")(h_action)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim, name="q2_fc2")(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1, name="q2_out")(q2)

        return q1, q2

@struct.dataclass
class AgentState:
    mvmae_params: FrozenDict
    actor_params: FrozenDict
    critic_params: FrozenDict
    critic_target_params: FrozenDict

    mvmae_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState

    rng: jax.Array
    
@struct.dataclass
class DrQV2Agent:
    # Static / config
    action_shape: Tuple[int, ...]

    nviews: int = 2
    mvmae_patch_size: int = 8
    mvmae_encoder_embed_dim: int = 256
    mvmae_decoder_embed_dim: int = 128
    mvmae_encoder_heads: int = 16
    mvmae_decoder_heads: int = 16
    in_channels: int = 9
    img_h_size: int = 64
    img_w_size: int = 64
    masking_ratio: float = 0.75

    feature_dim: int = 100
    hidden_dim: int = 1024
    lr: float = 1e-4

    # Non-pytree fields (module objects + optimizer transforms)
    mvmae: Any = struct.field(pytree_node=False, default=None)
    actor: Any = struct.field(pytree_node=False, default=None)
    critic: Any = struct.field(pytree_node=False, default=None)

    mvmae_tx: Any = struct.field(pytree_node=False, default=None)
    actor_tx: Any = struct.field(pytree_node=False, default=None)
    critic_tx: Any = struct.field(pytree_node=False, default=None)

    repr_dim: int = struct.field(pytree_node=False, default=0)

    @staticmethod
    def create(action_shape: Tuple[int, ...], **kwargs) -> "DrQV2Agent":
        agent = DrQV2Agent(action_shape=action_shape, **kwargs)

        total_patches = (agent.img_h_size // agent.mvmae_patch_size) * (
            (agent.nviews * agent.img_w_size) // agent.mvmae_patch_size
        )
        repr_dim = int(total_patches * agent.mvmae_encoder_embed_dim)

        mvmae = MAEModel(
            nviews=agent.nviews,
            patch_size=agent.mvmae_patch_size,
            encoder_embed_dim=agent.mvmae_encoder_embed_dim,
            decoder_embed_dim=agent.mvmae_decoder_embed_dim,
            encoder_heads=agent.mvmae_encoder_heads,
            decoder_heads=agent.mvmae_decoder_heads,
            in_channels=agent.in_channels,
            img_h_size=agent.img_h_size,
            img_w_size=agent.img_w_size,
            masking_ratio=agent.masking_ratio,
        )
        actor = Actor(
            repr_dim=repr_dim,
            action_shape=agent.action_shape,
            feature_dim=agent.feature_dim,
            hidden_dim=agent.hidden_dim,
        )
        critic = Critic(
            repr_dim=repr_dim,
            action_shape=agent.action_shape,
            feature_dim=agent.feature_dim,
            hidden_dim=agent.hidden_dim,
        )

        return agent.replace(
            mvmae=mvmae,
            actor=actor,
            critic=critic,
            mvmae_tx=optax.adam(agent.lr),
            actor_tx=optax.adam(agent.lr),
            critic_tx=optax.adam(agent.lr),
            repr_dim=repr_dim,
        )

    def init_state(self, seed: int) -> AgentState:
        rng = jax.random.PRNGKey(seed)
        rng, k_mae_p, k_mae_do, k_mae_mask, k_actor, k_critic = jax.random.split(rng, 6)

        dummy_obs = jnp.zeros(
            (1, self.img_h_size, self.nviews * self.img_w_size, self.in_channels),
            dtype=jnp.float32,
        )

        mvmae_vars = self.mvmae.init(
            {"params": k_mae_p, "dropout": k_mae_do, "mask": k_mae_mask},
            dummy_obs,
            deterministic=False,
        )
        mvmae_params = mvmae_vars["params"]

        dummy_z = jnp.zeros((1, self.repr_dim), dtype=jnp.float32)
        dummy_std = jnp.asarray(1.0, dtype=jnp.float32)
        actor_vars = self.actor.init({"params": k_actor}, dummy_z, dummy_std)
        actor_params = actor_vars["params"]

        dummy_action = jnp.zeros((1, self.action_shape[0]), dtype=jnp.float32)
        critic_vars = self.critic.init({"params": k_critic}, dummy_z, dummy_action)
        critic_params = critic_vars["params"]

        return AgentState(
            mvmae_params=mvmae_params,
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_params,
            mvmae_opt_state=self.mvmae_tx.init(mvmae_params),
            actor_opt_state=self.actor_tx.init(actor_params),
            critic_opt_state=self.critic_tx.init(critic_params),
            rng=rng,
        )

    @staticmethod
    def act(
        *, agent, state,
        obs: jnp.ndarray, step: jnp.ndarray,# scalar int32/int64 (jnp)
        eval_mode: jnp.ndarray,  # bool scalar (jnp.bool_)
        num_expl_steps: int, # exploration / schedule (Torch parity)
        std_start: float,
        std_end: float,
        std_duration: int,
        clip_pre_tanh: float | None = None,
        deterministic_encoder: bool = False, # encoder behavior (Torch used mask_x=False)
    ):
        """
        Torch parity:
            z = mvmae.encoder(obs, mask_x=False)
            stddev = linear schedule
            if eval_mode: action = mean
            else: action ~ policy
                if step < num_expl_steps: uniform(-1,1)
        Returns:
          action: (action_dim,)
          new_state: state with updated rng
          stddev: scalar jnp.float32 (useful for logging)
        """
        # ensure batch dim (env sometimes gives (1,H,W,C); replay might give (H,W,C))
        obs_b = obs if obs.ndim == 4 else obs[None, ...]
        # jittable linear stddev schedule: linear(std_start, std_end, std_duration)
        t = jnp.clip(step.astype(jnp.float32) / jnp.asarray(std_duration, jnp.float32), 0.0, 1.0)
        stddev = jnp.asarray(std_start, jnp.float32) + t * (jnp.asarray(std_end, jnp.float32) - jnp.asarray(std_start, jnp.float32))

        rng = state.rng
        rng, rng_enc, rng_act, rng_uni = jax.random.split(rng, 4)

        z = agent.mvmae.apply(
            {"params": state.mvmae_params},
            obs_b,
            deterministic=deterministic_encoder,
            method=MAEModel.encoder_no_masking,
            rngs={"dropout": rng_enc} if not deterministic_encoder else {},
        )
        z = z.reshape(z.shape[0], -1)  # (1, repr_dim)
        mu, std = agent.actor.apply({"params": state.actor_params}, z, stddev)

        def do_eval():
            return mu # mean action

        def do_sample():
            action = sample_tanh_gaussian(rng_act, mu, std, clip=clip_pre_tanh)
            return action
        
        action = jax.lax.cond(eval_mode, do_eval, do_sample)  # (1, action_dim)

        # first-N exploration override (Torch: uniform(-1,1) during exploration steps)
        def do_uniform_override(a_in):
            a_uni = jax.random.uniform(rng_uni, shape=a_in.shape, minval=-1.0, maxval=1.0, dtype=a_in.dtype)
            return a_uni

        # Only override when NOT eval_mode and step < num_expl_steps
        use_uniform = jnp.logical_and(jnp.logical_not(eval_mode), step < jnp.asarray(num_expl_steps))
        action = jax.lax.cond(use_uniform, do_uniform_override, lambda x: x, action)

        new_state = state.replace(rng=rng)
        return action[0], new_state, stddev
    
    @staticmethod
    def update_critic(
        *, mvmae, actor, critic, # modules (Flax Module objects, used with .apply)
        mvmae_params, actor_params, critic_params, critic_target_params, mvmae_opt_state, critic_opt_state, # params + opt states
        mvmae_tx: optax.GradientTransformation, critic_tx: optax.GradientTransformation, # optimizer related 
        obs, action, reward, discount, next_obs, # batch (already on-device jnp arrays)
        rng: jax.Array, step: jnp.ndarray, update_mvmae: jnp.ndarray, coef_mvmae: float, critic_target_tau: float, stddev: jnp.ndarray, stddev_clip: float,  # flags/hparams
    ) -> tuple[tuple[Any, ...], dict]:
        """
            Returns:
                (new_mvmae_params, new_critic_params, new_critic_target_params, new_mvmae_opt_state, new_critic_opt_state, new_rng), metrics dict (jnp arrays)
        """
        rng, rng_z, rng_znext, rng_nextact, rng_mask, rng_drop = jax.random.split(rng, 6)
        
        def encode_obs_with_grads():
            z = mvmae.apply(
                {"params": mvmae_params}, # learned weights and biases
                obs, deterministic=False, # arguments into the chosen method
                method=MAEModel.encoder_no_masking, rngs={"dropout": rng_z}
            )
            return z
        
        def encode_obs_no_grads():
            return jax.lax.stop_gradient(encode_obs_with_grads())
    
        z = jax.lax.cond(update_mvmae, encode_obs_with_grads, encode_obs_no_grads) # Only use grads on encoder when update_mvmae is true
        z = z.reshape(z.shape[0], -1)
        
        # Calculate z_next always with no grads
        z_next = mvmae.apply(
            {"params": mvmae_params},
            next_obs,
            deterministic=False,
            method=MAEModel.encoder_no_masking,
            rngs={"dropout": rng_znext},
        )
        z_next = jax.lax.stop_gradient(z_next).reshape(z_next.shape[0], -1)
        
        # Compute target Q
        mu_next, std_next = actor.apply({"params": actor_params}, z_next, stddev)
        next_action = sample_tanh_gaussian(rng_nextact, mu_next, std_next, clip=stddev_clip)
        target_q1, target_q2 = critic.apply({"params": critic_target_params}, z_next, next_action)
        target_v = jnp.minimum(target_q1, target_q2)
        target_q = reward + discount * target_v

        def recon_loss_fn(mv_params):
            out, mask, _ = mvmae.apply( # Default forward __call__
                {"params": mv_params},
                obs, deterministic=False,
                rngs={"mask": rng_mask, "dropout": rng_drop},
            )
            return mvmae.compute_loss(out, obs, mask) # compute_loss is pure JAX in MAEModel

        def total_loss_fn(crit_params, mv_params):
            q1, q2 = critic.apply({"params": crit_params}, z, action)
            critic_loss = jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)

            # Only add on reconstruction loss to the total loss if update_mvmae (every n steps)
            recon_loss = jax.lax.cond( 
                update_mvmae,
                lambda: recon_loss_fn(mv_params),
                lambda: jnp.asarray(0.0, jnp.float32),
            )
            total = critic_loss + coef_mvmae * recon_loss
            return total, (critic_loss, recon_loss, q1, q2)

        # Both forward() call to get loss, and then backward() to get grads
        (total_loss, (critic_loss, recon_loss, q1, q2)), (crit_grads, mv_grads) = jax.value_and_grad(
            total_loss_fn, argnums=(0, 1), has_aux=True
        )(critic_params, mvmae_params)
        
        # Critic update, stepping optimizer
        critic_updates, new_critic_opt_state = critic_tx.update(crit_grads, critic_opt_state, critic_params)
        new_critic_params = optax.apply_updates(critic_params, critic_updates)
        
        # Mvmae update only when update_mvmae is true (every n steps)
        def apply_mvmae():
            mv_updates, new_mv_opt = mvmae_tx.update(mv_grads, mvmae_opt_state, mvmae_params)
            new_mv_params = optax.apply_updates(mvmae_params, mv_updates)
            return new_mv_params, new_mv_opt

        new_mvmae_params, new_mvmae_opt_state = jax.lax.cond(
            update_mvmae,
            apply_mvmae,
            lambda: (mvmae_params, mvmae_opt_state),
        )

        # Soft update target critic
        new_critic_target_params = jax.tree_util.tree_map(
            lambda tp, p: tp * (1.0 - critic_target_tau) + p * critic_target_tau,
            critic_target_params,
            new_critic_params,
        )
        
        metrics = {
            "critic_loss": critic_loss,
            "recon_loss": recon_loss,
            "total_loss": total_loss,
            "q1_mean": jnp.mean(q1),
            "q2_mean": jnp.mean(q2),
            "target_q_mean": jnp.mean(target_q),
        }

        return (
            new_mvmae_params, new_critic_params, new_critic_target_params, new_mvmae_opt_state, new_critic_opt_state, rng
        ), metrics
    
    @staticmethod
    def update_actor(
        *, actor,critic, # Models
        actor_params, critic_params, actor_opt_state, actor_tx: optax.GradientTransformation, # params + optimizer related
        z: jnp.ndarray, # (B, repr_dim) already computed
        rng: jax.Array, stddev: jnp.ndarray, stddev_clip: float
    ):
        """
            Returns:
                (new_actor_params, new_actor_opt_state, new_rng), metrics
        """
        rng, rng_act = jax.random.split(rng, 2)

        def actor_loss_fn(a_params):
            mu, std = actor.apply({"params": a_params}, z, stddev)
            action = sample_tanh_gaussian(rng_act, mu, std, clip=stddev_clip) # Sample action
            q1, q2 = critic.apply({"params": critic_params}, z, action)
            q = jnp.minimum(q1, q2)
            loss = -jnp.mean(q)
            return loss, (mu, std, q)

        # Forward pass for loss, then backward pass for grads
        (loss, (mu, std, q)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)

        # Step optimizer 
        updates, new_opt_state = actor_tx.update(grads, actor_opt_state, actor_params)
        new_actor_params = optax.apply_updates(actor_params, updates)

        metrics = {
            "actor_loss": loss,
            "actor_mu_mean": jnp.mean(mu),
            "actor_std_mean": jnp.mean(std),
            "actor_q_mean": jnp.mean(q),
        }

        return (new_actor_params, new_opt_state, rng), metrics
    
    @staticmethod
    def update_step(
        *,
        agent,              # DrQV2Agent module instance (static in JIT context)
        state,              # AgentState (pytree)
        batch,              # (obs, action, reward, discount, next_obs) as jnp arrays
        step: jnp.ndarray,  # scalar int
        update_every_steps: int,
        update_mvmae_every_steps: int,
        coef_mvmae: float,
        critic_target_tau: float,
        stddev: jnp.ndarray,      # scalar
        stddev_clip: float,
    ):
        """
        Returns: (new_state, metrics)
        If step not update step: returns (state, empty_metrics)
        """
        obs, action, reward, discount, next_obs = batch

        do_update = (step % update_every_steps) == 0
        update_mvmae = (step % update_mvmae_every_steps) == 0
        
        def no_update():
            z = jnp.asarray(0.0, jnp.float32)
            metrics = {
                "critic_loss": z,
                "recon_loss": z,
                "total_loss": z,
                "q1_mean": z,
                "q2_mean": z,
                "target_q_mean": z,
                "actor_loss": z,
                "actor_mu_mean": z,
                "actor_std_mean": z,
                "actor_q_mean": z,
            }
            return state, metrics

        def do_update_fn():
            rng = state.rng
            (new_mvmae_params, new_critic_params, new_critic_target_params, new_mvmae_opt_state, new_critic_opt_state, rng), metrics_c = DrQV2Agent.update_critic(
                mvmae=agent.mvmae,
                actor=agent.actor,
                critic=agent.critic,
                mvmae_params=state.mvmae_params,
                actor_params=state.actor_params,
                critic_params=state.critic_params,
                critic_target_params=state.critic_target_params,
                mvmae_opt_state=state.mvmae_opt_state,
                critic_opt_state=state.critic_opt_state,
                mvmae_tx=agent.mvmae_tx,
                critic_tx=agent.critic_tx,
                obs=obs,
                action=action,
                reward=reward,
                discount=discount,
                next_obs=next_obs,
                rng=rng,
                step=step,
                update_mvmae=update_mvmae,
                coef_mvmae=coef_mvmae,
                critic_target_tau=critic_target_tau,
                stddev=stddev,
                stddev_clip=stddev_clip,
            )

            # Recompute z for actor update, using updated mvmae params, no gradients
            z_for_actor = agent.mvmae.apply(
                {"params": new_mvmae_params},
                obs, deterministic=False,
                method=MAEModel.encoder_no_masking,
                rngs={},
            )
            z_for_actor = jax.lax.stop_gradient(z_for_actor).reshape(z_for_actor.shape[0], -1)

            # Actor update (uses updated critic params)
            (new_actor_params, new_actor_opt_state, rng), metrics_a = DrQV2Agent.update_actor(
                actor=agent.actor,
                critic=agent.critic,
                actor_params=state.actor_params,
                critic_params=new_critic_params,
                actor_opt_state=state.actor_opt_state,
                actor_tx=agent.actor_tx,
                z=z_for_actor,
                rng=rng,
                stddev=stddev,
                stddev_clip=stddev_clip,
            )

            new_state = state.replace(
                mvmae_params=new_mvmae_params,
                actor_params=new_actor_params,
                critic_params=new_critic_params,
                critic_target_params=new_critic_target_params,
                mvmae_opt_state=new_mvmae_opt_state,
                actor_opt_state=new_actor_opt_state,
                critic_opt_state=new_critic_opt_state,
                rng=rng,
            )

            # merge metrics
            metrics = {**metrics_c, **metrics_a}
            return new_state, metrics

        return jax.lax.cond(do_update, do_update_fn, no_update)