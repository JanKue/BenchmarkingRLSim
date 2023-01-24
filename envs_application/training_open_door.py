import gym
import numpy as np

from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

import __init__


def main(env_name: str, path: str, total_steps: int = 3_000_000, seed: int = 1, learning_rate: float = 3e-4, **kwargs):

    # setup
    env = gym.make(env_name)  # regular env (SAC)
    # env = make_vec_env(env_name, n_envs=8)  # vector env (PPO)
    eval_env = gym.make(env_name)  # regular env for evaluation
    logger = configure(path, ["stdout", "tensorboard"])
    model_path = path + "/final_model"
    eval_path = path

    # optional action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1, action_noise=action_noise, seed=seed, learning_rate=learning_rate)
    # model = SAC.load(path=model_path, env=env, force_reset=True)
    model.set_logger(logger)
    model.learn(total_timesteps=total_steps,
                eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10, eval_log_path=eval_path)
    model.save(model_path)


if __name__ == "__main__":
    main(env_name="DoorOpenEnv-v1", path="../outcomes/local/door_open/sac")
