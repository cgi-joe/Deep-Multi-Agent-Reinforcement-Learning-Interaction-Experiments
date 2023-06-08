"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from pettingzoo.butterfly import pistonball_v6
import sys
import pickle
import argparse
import os
from collections import deque
# from tensorboardX import SummaryWriter

# log_dir = 'tensorboard_logs'
# os.makedirs(log_dir, exist_ok=True)
# writer = SummaryWriter(log_dir, flush_secs=10)

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x
print(f"CUDA = {torch.cuda.is_available()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pistonball PPO training and evaluation")
    # parser.add_argument("--evaluate", action="store_true", help="Evaluate a saved model instead of training")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model file")
    args = parser.parse_args()
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.3
    gamma = 0.99
    batch_size = 64
    stack_size = 4
    frame_size = (64, 64)
    max_cycles = 125
    total_episodes = 200

    """ ENV SETUP """
    env = pistonball_v6.parallel_env(
        render_mode="rgb_array", continuous=False, max_cycles=max_cycles
    )
    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.0001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)
    
    default_filename = f"trained_{total_episodes}_epochs.pk1"
    model_path = args.model_path if args.model_path else default_filename
    if args.model_path:
        if os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path))
            agent.eval()
        else:
            print(f"No saved model found: {model_path}")
            sys.exit(1)
    else:
        best_episodic_return = -np.inf
        last_10_episodic_returns = deque(maxlen=10)
        last_10_episodes = deque(maxlen=10)

        """ TRAINING LOGIC """
        # train for n number of episodes
        for episode in range(total_episodes):
            if episode == 150:
                    optimizer = optim.Adam(agent.parameters(), lr=0.00001, eps=1e-5)

            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs = env.reset(seed=None)
                # reset the episodic return
                total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, max_cycles):
                    # rollover the observation
                    obs = batchify_obs(next_obs, device)

                    # get action from the agent
                    actions, logprobs, _, values = agent.get_action_and_value(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = env.step(
                        unbatchify(actions, env)
                    )

                    # add to episode storage
                    rb_obs[step] = obs
                    rb_rewards[step] = batchify(rewards, device)
                    rb_terms[step] = batchify(terms, device)
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()

                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        bonus_reward = min(1, 5 / end_step)
                        total_episodic_return += bonus_reward
                        break
                

            # bootstrap value if not done
            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t]
                        + gamma * rb_values[t + 1] * rb_terms[t + 1]
                        - rb_values[t]
                    )
                    rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
                rb_returns = rb_advantages + rb_values

            # convert our episodes to batch of individual transitions
            b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
            b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
            b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
            b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
            b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
            b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(3):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[batch_index]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[batch_index] * ratio
                    pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            # last_10_episodic_returns.append(np.mean(total_episodic_return)/end_step)
            # last_10_episodes.append(episode)

            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(total_episodic_return)}")
            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            
            # if np.mean(total_episodic_return/end_step) > max(last_10_episodic_returns):
            #     best_episode = episode
            #     torch.save(agent.state_dict(), f"best_trained_model_{best_episode}.pk1")
            #     print(f"\nNew best model saved at episode {best_episode}")

            # if episode % 10 == 0 and episode is not 0:
            #     best_episode = last_10_episodes[np.argmax(last_10_episodic_returns)]
            #     if os.path.exists(f"best_trained_model_{best_episode}.pk1"):
            #         agent.load_state_dict(torch.load(f"best_trained_model_{best_episode}.pk1"))
            #         print(f"\nLoaded best model from episode {best_episode}")
                        
            print("\n-------------------------------------------\n")

            # Log metrics to TensorBoard
            # writer.add_scalar('Episodic Return', np.mean(total_episodic_return), episode)
            # writer.add_scalar('Episode Length', end_step, episode)
            # writer.add_scalar('Value Loss', v_loss.item(), episode)
            # writer.add_scalar('Policy Loss', pg_loss.item(), episode)
            # writer.add_scalar('Old Approx KL', old_approx_kl.item(), episode)
            # writer.add_scalar('Approx KL', approx_kl.item(), episode)
            # writer.add_scalar('Clip Fraction', np.mean(clip_fracs), episode)
            # writer.add_scalar('Explained Variance', explained_var.item(), episode)
      
        # save the trained model

    if not args.model_path:
        torch.save(agent.state_dict(), default_filename)



def process_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False

""" RENDER THE POLICY """
env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=4)

agent.eval()
action_hist = {0:None, 1:None, 2:None, 3:None, 4:None}
with torch.no_grad():
    # render 5 episodes out
    for episode in range(5):
        obs = batchify_obs(env.reset(seed=None), device)
        terms = [False]
        truncs = [False]
   
        while not any(terms) and not any(truncs):
            # action_hist.append(action_hist)
            actions, logprobs, _, values = agent.get_action_and_value(obs)
            # print(actions)
            if action_hist[episode] == None:
                action_hist[episode] = actions.unsqueeze(1)
            else:
                action_hist[episode] = torch.cat([action_hist[episode],actions.unsqueeze(1)], dim=1) 
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            obs = batchify_obs(obs, device)
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]
        # print(action_hist[episode])

        pygame.display.flip()
        pygame.time.wait(20)

            # Process Pygame events
        if process_pygame_events():
            pygame.quit()
            sys.exit()

# path to the pickle file
path_to_file = f"actions_hist_trained{total_episodes}.pkl"

# save the action_hist dictionary in a pickle file
with open(path_to_file, 'wb') as f:
    pickle.dump(action_hist, f)

    