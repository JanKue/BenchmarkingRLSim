import gym
import gym.spaces as spaces

import numpy as np

from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_controllers import GymController
from alr_sim.core import RobotBase, Scene
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere


class CustomReachEnv(gym.Env):

    def __init__(self,
                 scene: Scene,
                 robot: RobotBase,
                 controller: GymController,
                 max_steps):

        self.scene = scene
        self.robot = robot
        self.controller = controller
        self.max_steps = max_steps

        self.goal = Sphere(
            name="goal",
            size=[0.01],
            init_pos=[0.5, 0, 0.3],
            init_quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            static=True,
            visual_only=True
        )
        self.scene.add_object(self.goal)

        self.action_space = controller.action_space()
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(10,))

        self.terminated = False
        self.step_counter = 0
        self.episode_counter = 0

    def _get_observation(self):
        goal_position = self.scene.get_obj_pos(self.goal)
        robot_position = self.robot.current_c_pos
        pos_difference = robot_position - goal_position
        distance = np.linalg.norm(pos_difference)

        observation = np.concatenate([goal_position, robot_position, pos_difference, [distance]], dtype='float32')

        assert self.observation_space.contains(observation)

        return observation

    def _get_reward(self):
        goal_position = self.scene.get_obj_pos(self.goal)
        robot_position = self.robot.current_c_pos

        # reward function per Yu et al. 2020
        distance = np.linalg.norm(robot_position - goal_position)
        reward = 1000 * np.exp((distance**2) / 0.01)

        return -reward

    def _is_done(self):
        goal_position = self.scene.get_obj_pos(self.goal)
        robot_position = self.robot.current_c_pos
        distance = np.linalg.norm(robot_position - goal_position)

        reached_goal = (distance <= 0.05)

        if self.terminated or (self.step_counter > self.max_steps) or reached_goal:
            return True
        else:
            return False

    def start(self):
        self.scene.start()

    def step(self, action):

        self.controller.set_action(action)
        self.controller.execute_action(n_time_steps=500)

        self.step_counter += 1

        reward = self._get_reward()
        observation = self._get_observation()
        done = self._is_done()

        return observation, reward, done, {}

    def reset(self):
        self.terminated = False
        self.step_counter = 0
        self.episode_counter += 1
        self.scene.reset()
        return self._get_observation()

    def render(self):
        print(self._get_observation())

