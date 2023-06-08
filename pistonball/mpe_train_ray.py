import ray
from ray import tune
from ray.rllib.agents import ppo
from pettingzoo.mpe import simple_tag_v2
import supersuit as ss
from ray.tune.progress_reporter import CLIReporter
from ray.rllib.algorithms.callbacks import DefaultCallbacks

ray.init()

def env_creator(env_config):
    env = simple_tag_v2.env()
    # env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.frame_skip_v0(env, 2)
    # env = ss.pettingzoo_env_to_vec_env_v0(env)
    # env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')
    return env

tune.registry.register_env("simple_tag_v2", env_creator)

config = ppo.DEFAULT_CONFIG.copy()
config["env"] = "simple_tag_v2"
config["num_workers"] = 1
config["num_gpus"] = 0  # Set this to 1 if you have a GPU available
config["framework"] = "torch"  # You can use "tf" for TensorFlow

reporter = CLIReporter(metric_columns=["episode_reward_mean", "episode_len_mean", "timesteps_total"])

class RenderEvery25Episodes(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        if episode.episode_id % 25 == 0:
            base_env.get_unwrapped()[0].render()

results = tune.run(
    "PPO",
    config=config,
    stop={"timesteps_total": 1000000},
    checkpoint_freq=10,
    progress_reporter=reporter,
    # callbacks=RenderEvery25Episodes()
)

best_trial = max(results.trials, key=lambda trial: trial.last_result["episode_reward_mean"])
best_checkpoint = best_trial.checkpoint.value

def run_trained_model(checkpoint_path):
    cls = get_trainable_cls("PPO")
    agent = cls(config)
    agent.restore(checkpoint_path)

    env_creator = _global_registry.get(ENV_CREATOR, "simple_tag_v2")
    env = env_creator({})

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = {}
        for agent_id in obs:
            action[agent_id] = agent.compute_action(obs[agent_id], policy_id="default_policy")
        obs, reward, done, _ = env.step(action)
        total_reward += sum(reward.values())
        env.render()

    env.close()
    print("Total reward:", total_reward)

run_trained_model(best_checkpoint)
