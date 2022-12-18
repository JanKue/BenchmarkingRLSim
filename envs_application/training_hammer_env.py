import gym
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import __init__


def main():

    # setup parameters

    env_name = "HammerEnv-v1"  # should be  HammerEnv-v1
    env = gym.make(env_name)  # regular env (SAC)
    # env = make_vec_env(env_name, n_envs=8)  # vector env (PPO)
    eval_env = gym.make(env_name)  # regular env for evaluation
    logger = configure("../outcomes/tensorboard_log/hammer/sac", ["stdout", "tensorboard"])
    model_path = "../outcomes/models/sac_hammer"

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1, seed=1)
    # model = SAC.load(path=model_path, env=env, force_reset=True)
    model.set_logger(logger)
    model.learn(total_timesteps=2_500_000, eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10,
                eval_log_path="../outcomes/evaluation/hammer/sac")
    model.save(model_path)


if __name__ == "__main__":
    main()
