import gym
import numpy as np

from stable_baselines3 import SAC, PPO, TD3, DDPG, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import __init__


algorithms_dict = {"SAC": SAC, "PPO": PPO, "DDPG": DDPG, "TD3": TD3, "A2C": A2C}


def main(env_name: str, path: str, total_steps: int, algorithm: str, seed: int = 1, **kwargs):

    # setup environments
    env = gym.make(env_name)  # regular env (SAC, DDPG)
    # env = make_vec_env(env_name, n_envs=8, seed=seed)  # vector env (PPO)
    # env = VecNormalize(venv=env, norm_obs=True, norm_reward=True)
    eval_env = gym.make(env_name)  # regular env for evaluation
    # eval_env = make_vec_env(env_name, n_envs=1)
    # eval_env = VecNormalize(venv=eval_env, norm_reward=False, training=False)

    # optional action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    logger = configure(path, ["stdout", "log", "tensorboard"])
    model_path = path + "/final_model"
    eval_path = path

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    selected_algorithm = algorithms_dict[algorithm]

    model = selected_algorithm("MlpPolicy", env=env, verbose=1, seed=seed, action_noise=action_noise)
    model.set_logger(logger)
    model.learn(total_timesteps=total_steps,
                eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10, eval_log_path=eval_path)
    model.save(model_path)


if __name__ == "__main__":
    main(env_name="ReachEnv-v2", algorithm="TD3", total_steps=3_000_000, path="../outcomes/local/reach/random_goal/td3")
