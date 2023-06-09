"""This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.

Author: Will (https://github.com/WillDudley)

# Python version used: 3.8.10

# Requirements:
# pettingzoo == 1.22.0
# git+https://github.com/thu-ml/tianshou
# """

# from tianshou.data import Collector
# from tianshou.env import DummyVectorEnv, PettingZooEnv
# from tianshou.policy import MultiAgentPolicyManager, RandomPolicy


# from pettingzoo.classic import rps_v2
from pettingzoo.mpe import simple_tag_v2
import pickle
import tianshou as ts
from supersuit.multiagent_wrappers.padding_wrappers import pad_observations_v0, pad_action_space_v0

# if __name__ == "__main__":
#     # Step 1: Load the PettingZoo environment
#     env = simple_tag_v2.env(render_mode="human",
#                             num_good=1, 
#                             num_adversaries=3, 
#                             num_obstacles=2, 
#                             max_cycles=100, 
#                             continuous_actions=False)
#     env = pad_observations_v0(env)

#     # Step 2: Wrap the environment for Tianshou interfacing
#     env = PettingZooEnv(env)

#     # Step 3: Define policies for each agent
#     policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy(), RandomPolicy(), RandomPolicy()], env)

#     # Step 4: Convert the env to vector format
#     env = DummyVectorEnv([lambda: env])

#     # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
#     collector = Collector(policies, env)

#     # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
#     result = collector.collect(n_episode=1, render=0.05)
    
"""This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from pettingzoo.classic import tictactoe_v3


def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space.shape
            or observation_space["observation"].n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )
        

    # if agent_opponent is None:
    #     agent_opponent = RandomPolicy()

    agents = [agent_learn, agent_learn, agent_learn, RandomPolicy()]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    env = simple_tag_v2.env(render_mode="human",
                        num_good=1, 
                        num_adversaries=3, 
                        num_obstacles=2, 
                        max_cycles=500, 
                        continuous_actions=False)
    env = pad_observations_v0(env)

    return PettingZooEnv(env)


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[3]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 50

    def train_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.1)
        policy.policies[agents[1]].set_eps(0.1)
        policy.policies[agents[2]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[0]].set_eps(0.05)
        policy.policies[agents[1]].set_eps(0.05)
        policy.policies[agents[2]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 0]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )
    file_path = "mpe_agent_trained_policies.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(policy, f)

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")