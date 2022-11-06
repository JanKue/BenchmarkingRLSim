import gym
from stable_baselines3 import SAC, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

import __init__

if __name__ == "__main__":

    # setup parameters

    env = gym.make("DoorOpenEnv-v1")
    logger = configure("../outcomes/tensorboard_log/open_door/sac", ["stdout", "tensorboard"])
    model_path = "../outcomes/models/sac_open_door"

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1)
    # model = SAC.load(path=model_path, env=env, force_reset=True)
    model.set_logger(logger)
    model.learn(total_timesteps=1_500_000,
                eval_env=env, eval_freq=10_000, n_eval_episodes=10, eval_log_path="../outcomes/evaluation/open_door")
    model.save(model_path)

