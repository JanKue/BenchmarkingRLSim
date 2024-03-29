import __init__

import gym.spaces


def main():

    env = gym.make("HammerEnv-v0")

    ctrl = env.controller

    env.reset()
    for i in range(10_000):
        action = ctrl.action_space().sample()
        # print("action:", action)
        step_obs, step_reward, step_done, _ = env.step(action)
        # print("%d: Position " % (i), robot.current_c_pos)
        print("Latest step reward:", step_reward)
        # print("observation: ", step_obs)
        # print("valid observation: ", env.observation_space.contains(step_obs))

        if step_done or (i + 1) % 250 == 0:
            print("reset!")
            env.reset()


if __name__ == "__main__":
    main()
