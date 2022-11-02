import gym
from stable_baselines3 import SAC, TD3, A2C

from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.sims.SimFactory import SimRepository

from envs.meta_reach_env import MetaReachEnv

import __init__

if __name__ == "__main__":

    # create scene and environment

    env = gym.make("ReachEnv-v5")
    file_path = "../evaluation/simple_reach/best_model"

    # load trained model and run it

    model = SAC.load(path=file_path, env=env)
    print("Loaded model.")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print("action:", action, "/ observation:", obs, "/ reward:", rewards, "/ info:", info)
        if done:
            obs = env.reset()