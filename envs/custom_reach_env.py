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
                 max_steps,
                 random_env=False):

        self.scene = scene
        self.robot = robot
        self.controller = controller
        self.max_steps = max_steps
        self.random_env = random_env

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

        self.goal_space = spaces.Box(low=np.array([0.2, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.5]))
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float64)
        # normalize action space
        self.ctrl_action_space = controller.action_space()
        self.norm_factors = self.ctrl_action_space.high * 2
        self.norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self.ctrl_action_space.shape)
        self.action_space = self.norm_action_space  # self.ctrl_action_space

        self.terminated = False
        self.step_counter = 0
        self.episode_counter = 0

    def _get_observation(self):

        # per Yu et al. 2020, observation should be 9-dimensional and contain cartesian positions
        #  of end-effector, object (not applicable here), and goal

        self.robot.receiveState()
        tcp_pos = self.robot.current_c_pos  # end effector position
        goal_position = self.scene.get_obj_pos(self.goal)   # goal position

        observation = np.concatenate([tcp_pos, goal_position, goal_position])

        # assert self.observation_space.contains(observation)

        if not self.observation_space.contains(observation):
            print("Observation not in observation space!")
            print(observation)

        return observation

    def _get_reward(self):
        self.robot.receiveState()
        robot_position = self.robot.current_c_pos
        goal_position = self.scene.get_obj_pos(self.goal)

        # reward function per Yu et al. 2020
        distance = np.linalg.norm(robot_position - goal_position)
        reward = np.exp((distance**2) / 0.1, dtype=np.float32)

        if reward > np.finfo(np.float32).max:
            reward = np.finfo(np.float32).max

        return -reward

    def _is_done(self):
        goal_position = self.scene.get_obj_pos(self.goal)
        robot_position = self.robot.current_c_pos
        distance = np.linalg.norm(robot_position - goal_position)

        reached_goal = (distance <= 0.05)   # success metric from Yu et al. 2020

        if self.terminated or (self.step_counter > self.max_steps) or reached_goal:
            return True
        else:
            return False

    def start(self):
        self.scene.start()

    def step(self, action):

        # de-normalize action
        ctrl_action = action * 2 / self.norm_factors

        self.controller.set_action(ctrl_action)
        self.controller.execute_action(n_time_steps=20)

        self.step_counter += 1

        reward = self._get_reward()
        observation = self._get_observation()
        done = self._is_done()

        return observation, reward, done, {}

    def reset(self):
        self.terminated = False
        self.step_counter = 0
        self.episode_counter += 1

        if self.random_env:
            new_goal = [self.goal, self.goal_space.sample()]
            self.scene.reset([new_goal])
        else:
            self.scene.reset()

        return self._get_observation()

    def render(self):
        self.scene.render()
        print(self._get_observation())

