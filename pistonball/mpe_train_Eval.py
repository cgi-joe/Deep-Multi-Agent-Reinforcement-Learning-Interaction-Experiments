"""This is a minimal example to show how to use Tianshou with a PettingZoo environment. No training of agents is done here.

Author: Will (https://github.com/WillDudley)

# Python version used: 3.8.10

# Requirements:
# pettingzoo == 1.22.0
# git+https://github.com/thu-ml/tianshou
# """

from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy
import pickle

from pettingzoo.classic import rps_v2
from pettingzoo.mpe import simple_tag_v2

from supersuit.multiagent_wrappers.padding_wrappers import pad_observations_v0, pad_action_space_v0

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = simple_tag_v2.env(render_mode="human",
                            num_good=1, 
                            num_adversaries=3, 
                            num_obstacles=2, 
                            max_cycles=100, 
                            continuous_actions=False)
    env = pad_observations_v0(env)

    file_path = "D:\Projects\correlation\mpe_trained_policies10.pkl"
    agent_path = "D:\Projects\correlation\mpe_agent_trained_policies.pkl"
    with open(file_path, "rb") as f:
        trained_policy = pickle.load(f)
    with open(agent_path, "rb") as f:
        agent_path = pickle.load(f)
    # test = MultiAgentPolicyManager
    agent_idx = list(trained_policy.agent_idx.keys())
    # Load the saved policy
    
    # print(policy)
    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([agent_path.policies[agent_idx[0]], 
                                        agent_path.policies[agent_idx[1]], 
                                        agent_path.policies[agent_idx[2]], 
                                        agent_path.policies[agent_idx[3]]], 
                                        env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.01)
    env.close()
    