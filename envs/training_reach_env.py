import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.sims.SimFactory import SimRepository

from envs.custom_reach_env import CustomReachEnv

from alr_sim.core.logger import RobotPlotFlags

if __name__ == "__main__":

    # create scene and environment

    sim_factory = SimRepository.get_factory("mujoco")

    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)

    # ctrl = GymCartesianVelController(
    #     robot,
    #     fixed_orientation=np.array([0, 1, 0, 0]),
    #     max_cart_vel=0.1,
    #     use_spline=False,
    # )
    # robot.cartesianPosQuatTrackingController.neglect_dynamics = False

    ctrl = GymTorqueController(robot)
    env = CustomReachEnv(scene=scene, robot=robot, controller=ctrl, max_steps=200)

    env.start()
    scene.start_logging()

    # create model

    # check_env(env)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200)
    # model.save("a2c_reach_model")

    # run training

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print("action", action, "observation", obs, "reward", rewards)
        if done:
            obs = env.reset()

