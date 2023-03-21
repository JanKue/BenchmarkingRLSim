import gym
import numpy as np

from stable_baselines3 import SAC, PPO, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

import __init__


def main(env_name: str, path: str, total_steps: int, algorithm: str, seed: int = 1, **kwargs):

    # logging, paths setup
    logger = configure(path, ["stdout", "log", "csv", "tensorboard"])
    model_path = path + "/final_model"
    eval_path = path

    # environments and model setup
    if algorithm == 'PPO':
        env = make_vec_env(env_name, n_envs=8, seed=seed)  # vector env (PPO)
        env = VecNormalize(venv=env, norm_obs=True, norm_reward=True, training=True)
        eval_env = make_vec_env(env_name, n_envs=1)
        eval_env = VecNormalize(venv=eval_env, norm_obs=True, norm_reward=False, training=False)

        model = PPO("MlpPolicy", env=env, verbose=1, seed=seed)

    else:
        env = gym.make(env_name)  # regular env (SAC, DDPG, TD3)
        eval_env = gym.make(env_name)  # regular env for evaluation

        # action noise, optional for SAC
        action_noise = None
        if algorithm != 'SAC' or ('sac_action_noise' in kwargs and kwargs['sac_action_noise']):
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        selected_algorithm = {"SAC": SAC, "DDPG": DDPG, "TD3": TD3}[algorithm]
        model = selected_algorithm("MlpPolicy", env=env, verbose=1, seed=seed, action_noise=action_noise)

    # learning
    model.set_logger(logger)
    model.learn(total_timesteps=total_steps,
                eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10, eval_log_path=eval_path)
    model.save(model_path)


if __name__ == "__main__":
    main(env_name="DoorOpenEnv-v1", path="../outcomes/local/door_open/ddpg", total_steps=2_500_000, algorithm='SAC',
         sac_action_noise=True)
