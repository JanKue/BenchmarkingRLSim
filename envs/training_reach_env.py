import numpy as np

from stable_baselines3 import A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecCheckNan

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

    random_env = False
    env = CustomReachEnv(scene=scene, robot=robot, controller=ctrl, max_steps=200, random_env=random_env)

    env.start()
    scene.start_logging()

    # check_env(env)

    model = SAC("MlpPolicy", env, verbose=1)

    iterations = 10
    iteration_steps = 100

    for i in range(iterations):
        if random_env:
            # model = SAC.load("sac_reach_model_random")
            # print("Loaded random model.")
            model.learn(total_timesteps=iteration_steps)
            model.save("sac_reach_model_random")
            print("Saved random model.")
        else:
            # model = SAC.load("sac_reach_model_norandom")
            # print("Loaded static model.")
            model.learn(total_timesteps=iteration_steps)
            model.save("sac_reach_model_norandom")
            print("Saved static model.")



    #####

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     print("action", action, "observation", obs, "reward", rewards)
    #     if done:
    #         obs = env.reset()
