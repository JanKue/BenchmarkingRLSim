import gym
import gym.spaces as spaces

import numpy as np

from alr_sim.gyms.gym_controllers import GymController
from alr_sim.core import RobotBase, Scene
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere, Cylinder


class PushEnv(gym.Env):

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
            init_pos=[0.5, 0.8, 0.01],
            init_quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            static=True,
            visual_only=True
        )
        self.scene.add_object(self.goal)

        self.puck = Cylinder(
            name="puck",
            size=None,
            init_pos=[0.5, -0.5, 0.1],
            init_quat=[0, 0, 0, 0],
            rgba=[0, 0, 1, 1],
            static=False,
            visual_only=False
        )
        self.scene.add_object(self.puck)

        self.goal_space = spaces.Box(low=np.array([0.2, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.5]))
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(9,))
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
        goal_position = self.scene.get_obj_pos(self.goal)  # goal position
        puck_position = self.scene.get_obj_pos(self.puck) # object (puck) position

        observation = np.concatenate([tcp_pos, puck_position, goal_position], dtype='float32')

        if not self.observation_space.contains(observation):
            print("Observation not in observation space!")
            print(observation)

        return observation

    def _get_reward(self):
        self.robot.receiveState()
        robot_position = self.robot.current_c_pos
        goal_position = self.scene.get_obj_pos(self.goal)
        puck_position = self.scene.get_obj_pos(self.puck)

        robot_object_distance = np.linalg.norm(robot_position - puck_position)
        robot_goal_distance = np.linalg.norm(robot_position - goal_position)
        exp_distance = 1000 * np.exp((robot_goal_distance ** 2) / 0.01)
        indicator = 1 if robot_object_distance < 0.05 else 0

        reward = - robot_object_distance + indicator * exp_distance

        return -reward

    def _is_done(self):
        puck_position = self.scene.get_obj_pos(self.puck)
        goal_position = self.scene.get_obj_pos(self.goal)
        distance = np.linalg.norm(puck_position - goal_position)

        reached_goal = (distance <= 0.07)   # success metric from Yu et al. 2020

        if self.terminated or (self.step_counter > self.max_steps) or reached_goal:
            return True
        else:
            return False

    def start(self):
        self.scene.start()

    def step(self, action):

        ctrl_action = action * 2 / self.norm_factors

        self.controller.set_action(ctrl_action)
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

        # if self.random_env:
        #     new_goal = [self.goal, self.goal_space.sample()]
        #     self.scene.reset([new_goal])
        # else:
        #     self.scene.reset()
        self.scene.reset()

        return self._get_observation()

    def render(self):
        self.scene.render()
        print(self._get_observation())