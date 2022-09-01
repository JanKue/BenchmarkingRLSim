from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.sims.SimFactory import SimRepository

from envs.custom_reach_env import CustomReachEnv

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

    random_env = True
    total_steps = 4000
    env = CustomReachEnv(scene=scene, robot=robot, controller=ctrl, max_steps=200, random_env=random_env)
    logger = configure("../tensorboard_log/", ["stdout", "tensorboard"])

    random_path = "random" if random_env else "norandom"
    file_path = "sac_reach_model_" + random_path

    env.start()
    env.reset()
    scene.start_logging()

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    # model = SAC("MlpPolicy", env=env, verbose=1)

    model = SAC.load(path=file_path, env=env)
    print("Loaded " + random_path + " model.")

    model.learn(total_timesteps=total_steps,
                eval_env=env, eval_freq=100, tb_log_name="SAC_2022-09-01_01", eval_log_path="../evaluation/")

    model.save(file_path)
    print("Saved " + random_path + " model.")
