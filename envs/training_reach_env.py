from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.core.Scene import Scene

from envs.custom_reach_env import CustomReachEnv

if __name__ == "__main__":

    # setup parameters

    random_env = False
    total_steps = 200_000
    episode_steps = 100

    # create scene and environment

    sim_factory = SimRepository.get_factory("mujoco")

    scene = sim_factory.create_scene()
    scene.render_mode = Scene.RenderMode.BLIND
    robot = sim_factory.create_robot(scene)
    ctrl = GymTorqueController(robot)
    env = CustomReachEnv(scene=scene, robot=robot, controller=ctrl, max_steps=episode_steps, random_env=random_env)

    logger = configure("../tensorboard_log/4", ["stdout", "tensorboard"])
    random_path = "random" if random_env else "static"
    file_path = "../models/sac_reach_model_low_" + random_path

    env.start()
    env.reset()
    scene.start_logging()

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1)

    # model = SAC.load(path=file_path, env=env)
    print("Loaded " + random_path + " model.")

    model.set_logger(logger)

    model.learn(total_timesteps=total_steps, eval_env=env, eval_freq=10000, eval_log_path="../evaluation/")

    model.save(file_path)
    print("Saved " + random_path + " model.")
