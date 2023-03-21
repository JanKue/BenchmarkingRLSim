from abc import ABC

import numpy as np
from gym.spaces import Box as SamplingSpace
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.core.Scene import Scene


class ReachEnv(GymEnvWrapper, ABC):
    """
    Implementation of reaching environment with support for randomizing initial robot position and goal position.

    Args:
        random_goal (bool): toggle random sampling of goal
        random_init (bool): toggle random initial position
        render (bool): whether to set the render mode to HUMAN or BLIND
    """

    def __init__(
        self,
        n_substeps: int = 10,
        max_steps_per_episode: int = 250,
        debug: bool = True,
        random_goal: bool = False,
        random_init: bool = False,
        render=False
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

        self.random_goal = random_goal

        self.goal = Sphere(
            name="goal",
            size=[0.01],
            init_pos=[0.5, 0, 0.1],
            init_quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            static=True,
        )
        self.goal_space = SamplingSpace(
            low=np.array([0.2, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.5])
        )
        self.scene.add_object(self.goal)

        self.target_min_dist = 0.025

        self.random_init = random_init
        init_range = np.array([0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0, 0])
        self.init_space = SamplingSpace(low=-init_range, high=init_range)

        self.observation_space = SamplingSpace(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float64)
        self.action_space = self.controller.action_space()
        self.reward_range = (-np.inf, 0)

        self.start()
        self.orig_init_qpos = self.scene.init_qpos

    def get_observation(self) -> np.ndarray:
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, rel_goal_tcp_pos = obj_distance(goal_pos, tcp_pos)

        env_state = np.concatenate([goal_pos, [dist_tcp_goal], rel_goal_tcp_pos])
        robot_state = self.robot_state()
        return np.concatenate([robot_state, env_state])

    def get_reward(self):
        tcp = self.robot.current_c_pos
        target = self.scene.get_obj_pos(self.goal)

        reward = -np.exp(np.linalg.norm(tcp - target)**2)

        return reward

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, _ = obj_distance(goal_pos, tcp_pos)

        if dist_tcp_goal <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
            return True
        return False

    def _reset_env(self):
        if self.random_init:
            self.scene.init_qpos = self.orig_init_qpos + self.init_space.sample()

        if self.random_goal:
            new_goal = [self.goal, self.goal_space.sample()]
            self.scene.reset([new_goal])
        else:
            self.scene.reset()

    def reset(self):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self._reset_env()
        return self.get_observation()

    def debug_msg(self) -> dict:
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, _ = obj_distance(goal_pos, tcp_pos)
        success = dist_tcp_goal <= self.target_min_dist
        return {"is_success": success}

