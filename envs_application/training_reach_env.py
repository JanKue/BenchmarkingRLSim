from stable_baselines3 import SAC, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.core.Scene import Scene

from envs.meta_reach_env import MetaReachEnv

if __name__ == "__main__":

    # setup parameters

    random_env = False
    total_steps = 200_000
    episode_steps = 100

    # create scene and environment

    sim_factory = SimRepository.get_factory("mujoco")

    scene = sim_factory.create_scene(render=Scene.RenderMode.BLIND)
    robot = sim_factory.create_robot(scene)
    robot.gravity_comp = False
    ctrl = GymTorqueController(robot)
    env = MetaReachEnv(scene=scene, robot=robot, controller=ctrl, max_steps=episode_steps, random_env=random_env)

    logger = configure("../tensorboard_log/reach", ["stdout", "tensorboard"])
    random_path = "random" if random_env else "static"
    file_path = "../models/sac_reach_noexp_ctrlaction_model_" + random_path

    env.start()
    env.reset()
    scene.start_logging()

    # print("begin checking env")
    # check_env(env)
    # print("finished checking env")

    model = SAC("MlpPolicy", env=env, verbose=1, ent_coef=0.1, learning_starts=100)

    # model = SAC.load(path=file_path, env=env, force_reset=True)
    print("Loaded " + random_path + " model.")

    model.set_logger(logger)

    model.learn(total_timesteps=total_steps, eval_env=env, eval_freq=5000, eval_log_path="../evaluation/reach")

    model.save(file_path)
    print("Saved " + random_path + " model.")
