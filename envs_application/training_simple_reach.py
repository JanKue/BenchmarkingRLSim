import gym
from stable_baselines3 import SAC, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

import __init__

if __name__ == "__main__":

    # setup parameters

    env = gym.make("ReachEnv-v0")
    logger = configure("../tensorboard_log/simple_reach", ["stdout", "tensorboard"])
    model_path = "../models/sac_simple_reach"

    # print("begin checking env")
    check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1)
    # model = SAC.load(path=model_path, env=env, force_reset=True)
    model.set_logger(logger)
    model.learn(total_timesteps=2_000_000, eval_env=env, eval_freq=5000, eval_log_path="../evaluation/simple_reach")
    model.save(model_path)

