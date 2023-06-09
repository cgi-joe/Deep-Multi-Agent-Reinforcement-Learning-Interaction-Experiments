# This code is generated by Bing and is not tested or guaranteed to work
# It is based on the ppo_mpe.py script from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_mpe.py
# You might need to install some dependencies and adjust some hyperparameters before running it

import os
import time
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.mpe import simple_tag_v2

# Create a multi-agent environment
env = simple_tag_v2.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
env.reset()

# Get the observation and action spaces
obs_space = env.observation_spaces[env.agents[0]]
act_space = env.action_spaces[env.agents[0]]

# Define some hyperparameters
seed = 1 # random seed
total_timesteps = 1e6 # total number of timesteps to train
gamma = 0.99 # discount factor
lamda = 0.95 # GAE parameter
clip_ratio = 0.2 # PPO clip ratio
pi_lr = 3e-4 # learning rate for policy network
vf_lr = 1e-3 # learning rate for value network
train_pi_iters = 80 # number of policy gradient steps per epoch
train_v_iters = 80 # number of value function steps per epoch
target_kl = 0.01 # target KL divergence for early stopping
epochs = int(total_timesteps / env.num_agents) # number of epochs to train
save_freq = 10 # frequency of saving model and video

# Set random seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)

# Define a function to compute discounted rewards-to-go
def discount_cumsum(x, discount):
    return torch.tensor([x[i] * (discount ** i) for i in range(len(x))])

# Define a function to compute advantages using Generalized Advantage Estimation (GAE)
def compute_advantages(rewards, values, gamma, lamda):
    deltas = rewards + gamma * values[1:] - values[:-1]
    advantages = discount_cumsum(deltas, gamma * lamda)
    return advantages

# Define a function to compute the log probability of an action given a categorical distribution and an action
def compute_logp(pi, act):
    return Categorical(pi).log_prob(act)

# Define a function to compute the entropy of a categorical distribution
def compute_entropy(pi):
    return Categorical(pi).entropy()

# Define a function to compute the KL divergence between two categorical distributions
def compute_kl(pi1, pi2):
    return Categorical(pi1).kl_divergence(Categorical(pi2))

# Define a neural network class for policy and value functions
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi(obs)
            v = self.v(obs).squeeze(-1)
            a = Categorical(pi).sample()
            logp_a = compute_logp(pi, a)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def get_pi(self, obs):
        return self.pi(obs)

    def get_v(self, obs):
        return self.v(obs).squeeze(-1)

# Create a neural network for each agent
models = {agent: ActorCritic(obs_space.shape[0], act_space.n) for agent in env.agents}

# Create an optimizer for each network
optimizers = {agent: [optim.Adam(model.pi.parameters(), lr=pi_lr), optim.Adam(model.v.parameters(), lr=vf_lr)] for agent, model in models.items()}

# Create a summary writer for logging
writer = SummaryWriter()

#Training Loop
global_step = 0
for epoch in range(epochs):
    obs = env.reset()
    done = {agent: False for agent in env.agents}
    epoch_rewards = {agent: 0 for agent in env.agents}
    epoch_length = 0
    
    while not all(done.values()):
        epoch_length += 1
        global_step += 1
        actions, values, logp_olds = {}, {}, {}
        for agent in env.agents:
            # Get the action, value, and log probability for the current agent's observation
            a, v, logp_old = models[agent].step(torch.as_tensor(obs[agent], dtype=torch.float32))
            actions[agent] = a
            values[agent] = v
            logp_olds[agent] = logp_old

        # Step the environment with the actions
        next_obs, rewards, done, _ = env.step(actions)

        # Update the total rewards for each agent
        for agent in env.agents:
            epoch_rewards[agent] += rewards[agent]

        # Calculate the advantages for each agent
        advantages = {}
        for agent in env.agents:
            advantages[agent] = compute_advantages(torch.tensor([rewards[agent]]), torch.tensor([values[agent], models[agent].get_v(torch.as_tensor(next_obs[agent], dtype=torch.float32))]), gamma, lamda)

        # Update the models using PPO
        for agent in env.agents:
            for i in range(train_pi_iters):
                pi = models[agent].get_pi(torch.as_tensor(obs[agent], dtype=torch.float32))
                logp = compute_logp(pi, torch.as_tensor(actions[agent], dtype=torch.int64))
                ratio = torch.exp(logp - torch.as_tensor(logp_olds[agent], dtype=torch.float32))
                clipped_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages[agent]
                loss_pi = -(torch.min(ratio * advantages[agent], clipped_adv)).mean()

                # Calculate the approximate KL and early stopping if the KL is too large
                kl = compute_kl(pi, torch.as_tensor(pi.detach().numpy(), dtype=torch.float32)).mean().item()
                if kl > 1.5 * target_kl:
                    break

                # Update the policy
                optimizers[agent][0].zero_grad()
                loss_pi.backward()
                optimizers[agent][0].step()

            for _ in range(train_v_iters):
                v = models[agent].get_v(torch.as_tensor(obs[agent], dtype=torch.float32))
                loss_v = ((v - torch.as_tensor(values[agent], dtype=torch.float32)) ** 2).mean()

                # Update the value function
                optimizers[agent][1].zero_grad()
                loss_v.backward()
                optimizers[agent][1].step()

        obs = next_obs

    # Log the rewards and epoch length
    for agent in env.agents:
        writer.add_scalar(f'Reward/{agent}', epoch_rewards[agent], global_step)
    writer.add_scalar('Epoch Length', epoch_length, global_step)

    # Save the models and a video of the agents' performance
    if (epoch + 1) % save_freq == 0:
        for agent in env.agents:
            torch.save(models[agent].state_dict(), f'./models/{agent}_epoch{epoch+1}.pth')
        env.reset()
        env.render(mode='rgb_array', filename=f'./videos/epoch{epoch+1}.mp4')
