import gym
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

import __init__

if __name__ == "__main__":

    # setup parameters

    env_name = "ReachEnv-v2"  # should be  ReachEnv-v0, -v2, or -v4
    # env = gym.make(env_name)  # regular env (SAC)
    env = make_vec_env(env_name, n_envs=8)  # vector env (PPO)
    env = VecNormalize(venv=env, norm_obs=True, norm_reward=True)
    # eval_env = gym.make(env_name)  # regular env for evaluation
    eval_env = make_vec_env(env_name, n_envs=1)
    eval_env = VecNormalize(venv=eval_env)
    logger = configure("../outcomes/tensorboard_log/simple_reach/ppo/random_goal/exp_reward", ["stdout", "tensorboard"])
    model_path = "../outcomes/models/ppo_simple_reach_randomgoal"

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = PPO("MlpPolicy", env=env, verbose=1)
    # model = PPO.load(path=model_path, env=env, force_reset=True)
    model.set_logger(logger)
    model.learn(total_timesteps=10_000_000, eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10,
                eval_log_path="../outcomes/evaluation/simple_reach/ppo")
    model.save(model_path)

