import gym
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

import __init__


def main(env_name: str, path: str, total_steps: int = 3_000_000, seed: int = 1, **kwargs):

    # setup environments
    env = gym.make(env_name)  # regular env (SAC)
    # env = make_vec_env(env_name, n_envs=8, seed=seed)  # vector env (PPO)
    # env = VecNormalize(venv=env, norm_obs=True, norm_reward=True)
    eval_env = gym.make(env_name)  # regular env for evaluation
    # eval_env = make_vec_env(env_name, n_envs=1, seed=0)
    # eval_env = VecNormalize(venv=eval_env, norm_reward=False, training=False)
    logger = configure(path + "/tensorboard_log", ["stdout", "tensorboard"])
    model_path = path + "/models/model"
    eval_path = path + "/evaluation"

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1, seed=seed)
    # model = PPO.load(path=model_path, env=env, force_reset=True)
    model.set_logger(logger)
    model.learn(total_timesteps=total_steps,
                eval_env=eval_env, eval_freq=10_000, n_eval_episodes=10, eval_log_path=eval_path)
    model.save(model_path)


if __name__ == "__main__":
    main(env_name="ReachEnv-v2", path="../outcomes/local/reach/random_goal/sac")
