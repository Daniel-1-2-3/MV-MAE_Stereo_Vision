from stable_baselines3.sac.sac import SAC
import torch
from torch import nn

# Custom SAC that implements the mv-mae loss
class CustomSAC(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self, gradient_steps: int, batch_size: int = 32):
        for _ in range(gradient_steps):
            # Sample a batch from replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                next_obs_features = self.actor.features_extractor(replay_data.next_observations)
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                
                qf0_next_target = self.critic_target.qf0(torch.cat([next_obs_features, next_actions], dim=1))
                qf1_next_target = self.critic_target.qf1(torch.cat([next_obs_features, next_actions], dim=1))

                min_qf_next_target = torch.min(qf0_next_target, qf1_next_target) - self.log_ent_coef.exp().detach() * next_log_prob.unsqueeze(-1)
                next_q_value = replay_data.rewards + (1 - replay_data.dones) * self.gamma * min_qf_next_target

            obs_features = self.actor.features_extractor(replay_data.observations)
            current_q0 = self.critic.qf0(torch.cat([obs_features, replay_data.actions], dim=1))
            current_q1 = self.critic.qf1(torch.cat([obs_features, replay_data.actions], dim=1))

            critic_loss = nn.functional.mse_loss(current_q0, next_q_value) + nn.functional.mse_loss(current_q1, next_q_value)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # Actor update
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            obs_features = self.actor.features_extractor(replay_data.observations)
            qf0_pi = self.critic.qf0(torch.cat([obs_features, actions_pi], dim=1))
            qf1_pi = self.critic.qf1(torch.cat([obs_features, actions_pi], dim=1))
            min_qf_pi = torch.min(qf0_pi, qf1_pi)
            actor_loss = (self.log_ent_coef.exp().detach() * log_prob.unsqueeze(-1) - min_qf_pi).mean()
        
            # MV-MAE loss
            mvmae_loss = self.actor.features_extractor.last_mvmae_loss
            total_loss = actor_loss + (mvmae_loss if mvmae_loss is not None else 0)

            print(f'mvmae: {mvmae_loss.item() if mvmae_loss is not None else "None"}, '
                f'actor: {actor_loss.item()}, total: {total_loss.item()}')

            self.actor.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            
            # Entropy coefficient update
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef = self.log_ent_coef.exp().detach()
