from stable_baselines3.sac.sac import SAC
import torch
from torch import nn
import csv, os

# Custom SAC that implements the mv-mae loss
class CustomSAC(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create CSV header
        self.log_file = "training_log.csv"
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["MVMAE_L", "Policy_L", "Total_Loss", "Batch_Reward_Mean"])
        
    def train(self, gradient_steps: int, batch_size: int = 32):
        for _ in range(gradient_steps):
            # Sample a batch from replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Move all replay data tensors to the correct device
            obs = {k: v.to(self.device) for k, v in replay_data.observations.items()}
            next_obs = {k: v.to(self.device) for k, v in replay_data.next_observations.items()}
            actions = replay_data.actions.to(self.device)
            rewards = replay_data.rewards.to(self.device)
            dones = replay_data.dones.to(self.device)

            with torch.no_grad():
                next_obs_features = self.actor.features_extractor(next_obs)
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)

                qf0_next_target = self.critic_target.qf0(torch.cat([next_obs_features, next_actions], dim=1))
                qf1_next_target = self.critic_target.qf1(torch.cat([next_obs_features, next_actions], dim=1))

                min_qf_next_target = torch.min(qf0_next_target, qf1_next_target) - self.log_ent_coef.exp().detach() * next_log_prob.unsqueeze(-1)
                next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target

            obs_features = self.actor.features_extractor(obs)
            current_q0 = self.critic.qf0(torch.cat([obs_features, actions], dim=1))
            current_q1 = self.critic.qf1(torch.cat([obs_features, actions], dim=1))

            critic_loss = nn.functional.mse_loss(current_q0, next_q_value) + nn.functional.mse_loss(current_q1, next_q_value)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # Actor update
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            obs_features = self.actor.features_extractor(obs)
            qf0_pi = self.critic.qf0(torch.cat([obs_features, actions_pi], dim=1))
            qf1_pi = self.critic.qf1(torch.cat([obs_features, actions_pi], dim=1))
            min_qf_pi = torch.min(qf0_pi, qf1_pi)
            
            policy_loss = (self.log_ent_coef.exp().detach() * log_prob.unsqueeze(-1) - min_qf_pi).mean()
            mvmae_loss = self.actor.features_extractor.last_mvmae_loss # MV-MAE loss
            total_loss =  policy_loss + mvmae_loss * 0.01 # Gradients from mvmae are much stronger

            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    mvmae_loss.item() if mvmae_loss is not None else None,
                    policy_loss.item(),
                    total_loss.item(),
                    rewards.mean().item(),
                    self.log_ent_coef.exp().detach(),
            ])
            print(f'mvmae: {mvmae_loss.item()}, policy: {policy_loss.item()}, total_loss: {total_loss.item()}, rewards: {rewards.mean().item()}')

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
