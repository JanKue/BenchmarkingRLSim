import __init__

import gym.spaces


def main():

    env = gym.make("DoorOpenEnv-v0")
    ctrl = env.controller

    step_done = False
    for i in range(10_000):
        if step_done or i % 250 == 0:
            print("reset!")
            env.reset()

        action = ctrl.action_space().sample()
        # print("action:", action)
        step_obs, step_reward, step_done, _ = env.step(action)
        # print("%d: Position " % (i), robot.current_c_pos)
        print("Latest step reward:", step_reward)
        # print("observation: ", step_obs)
        # print("valid observation: ", env.observation_space.contains(step_obs))


if __name__ == "__main__":
    main()
