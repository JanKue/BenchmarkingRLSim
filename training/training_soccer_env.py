import gym
import numpy as np
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize

import __init__

algorithms_dict = {"SAC": SAC, "PPO": PPO, "DDPG": DDPG, "TD3": TD3, "A2C": A2C}


def main(env_name: str, path: str, algorithm: str, total_steps: int = 2_500_000, seed: int = 1, **kwargs):

    # setup paths
    logger = configure(path, ["stdout", "log", "csv", "tensorboard"])
    model_path = path + "/final_model"
    eval_path = path

    # setup envs, model
    if algorithm == 'PPO':
        env = make_vec_env(env_name, n_envs=8, seed=seed)  # vector env (PPO)
        env = VecNormalize(venv=env, norm_obs=True, norm_reward=True, training=True)
        eval_env = make_vec_env(env_name, n_envs=1)
        eval_env = VecNormalize(venv=eval_env, norm_obs=True, norm_reward=False, training=False)

        model = PPO("MlpPolicy", env=env, verbose=1, seed=seed)

    else:
        env = gym.make(env_name)  # regular env (SAC, DDPG, TD3)
        eval_env = gym.make(env_name)  # regular env for evaluation

        # optional action noise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        selected_algorithm = algorithms_dict[algorithm]
        model = selected_algorithm("MlpPolicy", env=env, verbose=1, seed=seed, action_noise=action_noise)

    # training
    model.set_logger(logger)
    model.learn(total_timesteps=total_steps,
                eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10, eval_log_path=eval_path)
    model.save(model_path)


if __name__ == "__main__":
    main(env_name="SoccerEnv-v3", algorithm='SAC', path="../outcomes/local/soccer_random/sac")
