import gym
from stable_baselines3 import SAC, PPO, TD3, A2C

import __init__


def main():

    # create scene and environment

    env = gym.make("ReachEnv-v5")
    file_path = "../outcomes/local/evaluation/simple_reach/sac/best_model_random_init.zip"
    # file_path = "../outcomes/models/ppo_simple_reach_randomgoal"

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


if __name__ == "__main__":
    main()
