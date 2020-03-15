# -*- encoding: utf-8 -*-
"""
@File           :   ddpg_agent.py
@Time           :   2020_01_26-20:08:22
@Author         :   zhenwang
@Description    :
  - Version 1.0.0: File created.
"""
import os
import copy
import random

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from config import *
from memory import ReplyBuffer
from model import Actor, Critic


class Agent(object):
    """
    Interacts with and learns from the environment.
    """

    def __init__(self, state_space, hidden_size, action_size, num_agents,
                 seed=0, buffer_size=int(1e6),
                 actor_lr=1e-4, actor_hidden_sizes=(128, 256), actor_weight_decay=0,
                 critic_lr=1e-4, critic_hidden_sizes=(128, 256, 128), critic_weight_decay=0,
                 batch_size=128, gamma=0.99, tau=1e-3):
        """
        Initialize an Agent object.

        Params
        ======
            state_space (tuple): dimension of each states
            hidden_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents to train
            seed (int): random seed, default value is 0
            buffer_size (int): buffer size of experience memory, default value is 100000

            actor_lr (float): learning rate of actor model, default value is 1e-4
            actor_lr (float): learning rate of actor model, default value is 1e-4
            actor_hidden_sizes (tuple): size of hidden layer of actor model, default value is (128, 256)
            critic_lr (float): learning rate of critic model, default value is 1e-4
            critic_hidden_sizes (tuple): size of hidden layer of critic model, default value is (128, 256, 128)

            batch_size (int): mini-batch size
            gamma (float): discount factor
            tau (float): interpolation parameter
        """
        self.state_space = state_space
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = seed

        self.batch_size = batch_size  # mini-batch size
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters

        # Actor Network
        self.actor_local = Actor(state_space, hidden_size, action_size, seed,
                                 hidden_units=actor_hidden_sizes).to(DEVICE)
        self.actor_target = Actor(state_space, hidden_size, action_size, seed,
                                  hidden_units=actor_hidden_sizes).to(DEVICE)
        self.actor_target.eval()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=actor_lr,
                                          weight_decay=actor_weight_decay)

        # Critic Network
        self.critic_local = Critic(state_space, hidden_size, action_size, seed,
                                   hidden_units=critic_hidden_sizes).to(DEVICE)
        self.critic_target = Critic(state_space, hidden_size, action_size, seed,
                                    hidden_units=critic_hidden_sizes).to(DEVICE)
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=critic_lr,
                                           weight_decay=critic_weight_decay)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), seed)

        # Replay memory
        self.memory = ReplyBuffer(buffer_size=buffer_size, seed=seed)

        # copy parameters of the local model to the target model
        self.soft_update(self.critic_local, self.critic_target, 1.)
        self.soft_update(self.actor_local, self.actor_target, 1.)

        self.seed = random.seed(seed)
        np.random.seed(seed)

        self.reset()

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        state = np.asarray([state])

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1., 1.)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """

        # Save experience / reward
        #  for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(batch_size=self.batch_size)
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """
        Update policy and experiences parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-experiences

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        actions, rewards, dones = torch.from_numpy(actions).float().to(DEVICE), \
                                  torch.from_numpy(rewards).float().to(DEVICE), \
                                  torch.from_numpy(dones).to(DEVICE)

        # ------- update critic ------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_targets = q_targets.detach()

        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        assert q_expected.shape == q_targets.shape
        critic_loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)  # clip the gradient (Udacity)
        self.critic_optimizer.step()

        # ------- update actor ------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #  update target networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.detach_()
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self):
        """
        Save model state
        """
        torch.save(self.actor_local.state_dict(), "checkpoints/checkpoint_actor.pth")
        torch.save(self.actor_target.state_dict(), "checkpoints/checkpoint_actor_target.pth")

        torch.save(self.critic_local.state_dict(), "checkpoints/checkpoint_critic.pth")
        torch.save(self.critic_target.state_dict(), "checkpoints/checkpoint_critic_target.pth")

    def load(self):
        """
        Load model state
        """
        if not os.path.exists("checkpoints/checkpoint_actor.pth") or \
                not os.path.exists("checkpoints/checkpoint_actor_target.pth") or \
                not os.path.exists("checkpoints/checkpoint_critic.pth") or \
                not os.path.exists("checkpoints/checkpoint_critic_target.pth"):
            return

        self.actor_local.load_state_dict(torch.load("checkpoints/checkpoint_actor.pth"))
        self.actor_target.load_state_dict(torch.load("checkpoints/checkpoint_actor_target.pth"))

        self.critic_local.load_state_dict(torch.load("checkpoints/checkpoint_critic.pth"))
        self.critic_target.load_state_dict(torch.load("checkpoints/checkpoint_critic_target.pth"))

    def __str__(self):
        return f"{str(self.actor_local)}\n{str(self.critic_local)}"


class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
