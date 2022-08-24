from stable_baselines3 import SAC

from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.sims.SimFactory import SimRepository

from envs.custom_reach_env import CustomReachEnv

if __name__ == "__main__":

    # create scene and environment

    sim_factory = SimRepository.get_factory("mujoco")

    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)

    ctrl = GymTorqueController(robot)

    random_env = False
    env = CustomReachEnv(scene=scene, robot=robot, controller=ctrl, max_steps=1000, random_env=random_env)

    random_path = "random" if random_env else "norandom"
    file_path = "sac_reach_model_" + random_path

    env.start()
    scene.start_logging()

    # load trained model and run it

    model = SAC.load(path=file_path, env=env)
    print("Loaded " + random_path + " model.")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print("action", action, "observation", obs, "reward", rewards)
        if done:
            obs = env.reset()
