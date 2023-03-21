from abc import ABC

import numpy as np
from alr_sim.core import Scene
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere

from objects.soccer_objects import SoccerGoal

from gym.spaces import Box


class SoccerEnv(GymEnvWrapper, ABC):
    """
    Implementation of soccer environment based on Meta-World with support for randomizing ball position.

    Args:
        random_ball_pos (bool): toggle random sampling of ball's starting position
        render (bool): whether to set the render mode to HUMAN or BLIND
        reward_dist_weight (int): reward function hyperparameter
        reward_touching_weight (int): reward function hyperparameter
        reward_behind_goal (int): reward function hyperparameter
    """

    def __init__(
            self,
            n_substeps: int = 10,
            max_steps_per_episode: int = 625,
            debug: bool = True,
            random_ball_pos: bool = False,
            render: bool = False,
            reward_dist_weight: int = 50,
            reward_touching_weight: int = 15,
            reward_behind_goal: int = 50
    ):
        sim_factory = SimRepository.get_factory("mujoco")
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(render=render_mode)
        robot = sim_factory.create_robot(scene)
        controller = GymTorqueController(robot)
        robot.cartesianPosQuatTrackingController.neglect_dynamics = False
        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.random_ball_pos = random_ball_pos
        self.ball_pos_space = Box(low=np.array([0.2, -0.5, 0.01]), high=np.array([0.6, 0.3, 0.01]), dtype=np.float64)
        self.soccer_ball = Sphere(name="soccer_ball", rgba=[0.8, 0.8, 0.8, 1],
                             init_pos=[0.4, -0.1, 0.1], init_quat=[0, 0, 0, 0],
                             size=[0.026], mass=0.04)
        self.scene.add_object(self.soccer_ball)

        self.soccer_goal_object = SoccerGoal(name="soccer_goal_object")
        self.scene.add_object(self.soccer_goal_object)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float64)
        self.action_space = self.controller.action_space()
        self.reward_range = (-np.inf, 0)

        self.goal_bounds_min = np.array([0.3, 0.44, -0.02])
        self.goal_bounds_max = np.array([0.5, 0.54, 0.13])
        self.goal_box = Box(low=np.array([0.3, 0.44, -0.02]), high=np.array([0.5, 0.54, 0.13]), dtype=np.float64)

        self.reward_dist_weight = reward_dist_weight
        self.reward_touching_weight = reward_touching_weight
        self.reward_behind_goal = reward_behind_goal

        self.start()

    def get_observation(self) -> np.ndarray:
        robot_state = self.robot_state()

        tcp_pos = self.robot.current_c_pos
        ball_pos = self.scene.get_obj_pos(self.soccer_ball)
        goal_pos = self.scene.sim.data.get_body_xpos("soccer_goal")
        tcp_ball_dist, tcp_ball_rel_dist = obj_distance(tcp_pos, ball_pos)
        ball_goal_dist, ball_goal_rel_dist = obj_distance(ball_pos, goal_pos)

        env_state = np.concatenate([tcp_pos, ball_pos, goal_pos,
                                    [tcp_ball_dist, ball_goal_dist], tcp_ball_rel_dist, ball_goal_rel_dist,
                                    self.goal_bounds_min, self.goal_bounds_max])

        return np.concatenate([robot_state, env_state])

    def get_reward(self):

        # first: check if ball is in goal -> terminate immediately with max reward
        ball_pos = self.scene.get_obj_pos(self.soccer_ball)
        if self.goal_box.contains(ball_pos):
            self.terminated = True
            return 0

        self.robot.receiveState()
        tcp_pos = self.robot.current_c_pos

        goal_center_pos = self.scene.sim.data.get_body_xpos("soccer_goal")
        tcp_ball_dist, _ = obj_distance(tcp_pos, ball_pos)
        ball_goal_dist, _ = obj_distance(ball_pos, goal_center_pos)

        ball_touching_penalty = self.reward_touching_weight * tcp_ball_dist if tcp_ball_dist > 0.1 else 0
        behind_goal_penalty = self.reward_behind_goal if ball_pos[1] > 0.44 else 0

        reward = - self.reward_dist_weight * ball_goal_dist - ball_touching_penalty - behind_goal_penalty

        return np.minimum(reward, 0)

    def _check_early_termination(self) -> bool:
        ball_pos = self.scene.get_obj_pos(self.soccer_ball)
        if self.goal_box.contains(ball_pos):
            self.terminated = True
            return True
        else:
            return False

    def _reset_env(self):
        if self.random_ball_pos:
            new_ball = [self.soccer_ball, self.ball_pos_space.sample()]
            self.scene.reset([new_ball])
        else:
            self.scene.reset()

    def reset(self):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self._reset_env()
        return self.get_observation()

    def debug_msg(self) -> dict:
        ball_pos = self.scene.get_obj_pos(self.soccer_ball)
        return {"is_success": self.goal_box.contains(ball_pos)}
