import gym.spaces

import __init__


def main():

    env = gym.make("ReachEnv-v1")
    scene = env.scene
    robot = env.robot
    ctrl = env.controller

    env.start()

    scene.start_logging()
    env.reset()
    for i in range(200):
        action = ctrl.action_space().sample()
        print("action:", action)
        step_obs, step_reward, step_done, _ = env.step(action)
        print("%d: Position " % (i), robot.current_c_pos)
        print("Latest step reward:", step_reward)
        print("observation: ", step_obs)
        print("valid observation: ", env.observation_space.contains(step_obs))

        if step_done or (i + 1) % 50 == 0:
            print("reset!")
            env.reset()

    scene.stop_logging()


if __name__ == "__main__":
    main()
