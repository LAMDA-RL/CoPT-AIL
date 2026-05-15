import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import hydra

from module.discriminator import ShapedDiscriminator, ShapedStdDiscriminator
from module.net import soft_update, hard_update
from module.critic import DoubleQCritic
from module.rot import Actor, ActorStd, Critic
from utils.utils import get_concat_samples, wd_param_groups, LinearSchedule


class FOIL(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.name = "foil"
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.args = args
        self.step = 0
        self.bc_transit = args.agent.bc_transit
        self.bc_weight = LinearSchedule(0.99, 0, 100000)
        self.disc_reg = args.agent.disc_reg
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.disc_update_frequency = agent_cfg.disc_update_frequency
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        reward_scale = 1 / action_dim
        self.discriminator = ShapedStdDiscriminator(obs_dim, action_dim, 50, 1024, scale=reward_scale).to(self.device)

        self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)
        self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = ActorStd(obs_dim, action_dim, 50, 1024).to(self.device)
        self.actor_bc = ActorStd(obs_dim, action_dim, 50, 1024).to(self.device)
        for param in self.actor_bc.parameters():
            param.requires_grad = False

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        disc_param_groups = wd_param_groups(self.discriminator, weight_decay=1e-4)
        self.disc_optimizer = AdamW(disc_param_groups,
                                   lr=agent_cfg.disc_lr,
                                   betas=agent_cfg.disc_betas)
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    @torch.no_grad()
    def infer_r(self, state, action):
        return self.discriminator(state, action)

    def load_state(self, path, actor=True, disc=True):
        state = torch.load(path, map_location=self.device)
        if actor:
            self.actor.load_state_dict(state)
            self.actor_bc.load_state_dict(state)
        if disc:
            self.discriminator.load_state_dict(state)

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        if torch.isnan(action).any():
            print(action)
        return action.detach().cpu().numpy()[0]

    def update(self, policy_buffer, expert_buffer, logger, step):
        policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
        expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

        losses = {}

        if self.step % self.disc_update_frequency == 0:
            losses.update(self.update_discriminator(policy_batch, expert_batch))

        expert_ratio = 0.2
        expert_size = int(self.batch_size * expert_ratio)
        mix_batch = get_concat_samples(expert_buffer.get_samples(expert_size, self.device),
                                       policy_buffer.get_samples(self.batch_size - expert_size, self.device))

        losses.update(self.update_critic(mix_batch))

        if self.actor and self.step % self.actor_update_frequency == 0:

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(expert_batch, mix_batch)
            losses.update(actor_alpha_losses)

        if self.step % self.critic_target_update_frequency == 0:
            if self.args.train.soft_update:
                soft_update(self.critic_net, self.critic_target_net,
                            self.critic_tau)
            else:
                hard_update(self.critic_net, self.critic_target_net)

        if step % 100 == 0:
            logger.log_train(losses)

        self.step += 1

        return losses

    def update_discriminator(self, policy_batch, expert_batch):
        loss_dict = dict()
        policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
        expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

        expert_reward = self.discriminator(expert_obs, expert_action)
        policy_reward = self.discriminator(policy_obs, policy_action)
        policy_reward = torch.clamp(policy_reward, min=-5)

        disc_loss = policy_reward.mean() + torch.exp(-policy_reward).mean() * self.disc_reg - expert_reward.mean()
        loss = disc_loss


        is_inf = torch.isinf(torch.exp(-policy_reward))
        if is_inf.any():
            print(torch.exp(-policy_reward))
            raise RuntimeError('Loss divergence not finite')

        self.disc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.disc_optimizer.step()

        if self.step == 0:
            print(f'expert_reward: {expert_reward.mean().item()}, policy_reward: {policy_reward.mean().item()}')

        loss_dict['expert_reward'] = expert_reward.mean().item()
        loss_dict['policy_reward'] = policy_reward.mean().item()
        loss_dict['discriminator_loss'] = disc_loss.item()
        if hasattr(self.discriminator, 'std'):
            loss_dict['disc_std'] = self.discriminator.std

        return loss_dict

    def update_critic(self, mix_batch):
        obs, next_obs, action, reward, done = mix_batch[:5]
        reward = self.infer_r(obs, action)

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = (reward + (1 - done) * self.gamma * target_V).clip(-100, 100)

        if isinstance(self.critic, DoubleQCritic):
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        else:
            current_Q = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        loss_dict = dict()
        loss_dict['critic_loss'] = critic_loss.item()
        loss_dict['target_Q'] = target_Q.mean().item()
        if isinstance(self.critic, DoubleQCritic):
            loss_dict['current_Q'] = current_Q1.mean().item()
        else:
            loss_dict['current_Q'] = current_Q.mean().item()
        return loss_dict

    def update_actor_and_alpha(self, expert_batch, mix_batch):
        expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
        obs = mix_batch[0]
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        if self.bc_transit:
            bc_weight = self.bc_weight(self.step)
            u_target = self.actor.atanh(expert_action)
            mu, log_std = self.actor.pre_tanh_params(expert_obs)
            nll = -self.actor.log_prob(expert_obs, expert_action).mean()
            pre_mse = F.mse_loss(mu, u_target)
            bc_loss = nll
            actor_loss = bc_weight * bc_loss * 0.03 + (1 - bc_weight) * actor_loss
        else:
            bc_weight = 0
            bc_loss = -self.actor.log_prob(expert_obs, expert_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'actor_loss': actor_loss.item(),
            'actor_entropy': -log_prob.mean().item(),
            'actor_Q': actor_Q.mean().item(),
            'bc_weight': bc_weight,
            'bc_loss': bc_loss.item(),
        }
        if hasattr(self.actor, 'std'):
            losses['actor_std'] = self.actor.std

        return losses
