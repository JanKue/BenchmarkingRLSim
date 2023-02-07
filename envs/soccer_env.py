import numpy as np
from alr_sim.core import Scene
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere

from envs.objects.soccer_objects import SoccerObjects

from gym.spaces import Box


class SoccerEnv(GymEnvWrapper):
    def __init__(
            self,
            simulator: str = "mujoco",
            n_substeps: int = 10,
            max_steps_per_episode: int = 625,
            debug: bool = True,
            random_init: bool = False,
            random_ball_pos: bool = False,
            render=False
    ):
        sim_factory = SimRepository.get_factory(simulator)
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
        self.ball_pos_space = Box(low=np.array([0.2, -0.5, 0.01]), high=np.array([0.6, 0.3, 0.01]))
        self.soccer_ball = Sphere(name="soccer_ball", rgba=[0.8, 0.8, 0.8, 1],
                             init_pos=[0.4, -0.1, 0.1], init_quat=[0, 0, 0, 0],
                             size=[0.026], mass=0.04)
        self.scene.add_object(self.soccer_ball)

        self.soccer_goal_object = SoccerObjects(name="soccer_goal_object")
        self.scene.add_object(self.soccer_goal_object)

        self.random_init = random_init
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float64)
        self.action_space = self.controller.action_space()
        self.reward_range = (-np.inf, 0)

        self.goal_bounds_min = np.array([0.3, 0.44, -0.02])
        self.goal_bounds_max = np.array([0.5, 0.54, 0.13])
        self.goal_box = Box(low=np.array([0.3, 0.44, -0.02]), high=np.array([0.5, 0.54, 0.13]))

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

        # first: check if ball is in goal
        ball_pos = self.scene.get_obj_pos(self.soccer_ball)

        if self.goal_box.contains(ball_pos):
            return 0

        self.robot.receiveState()
        tcp_pos = self.robot.current_c_pos
        tcp_quat = self.robot.current_c_quat

        goal_center_pos = self.scene.sim.data.get_body_xpos("soccer_goal")
        tcp_ball_dist, _ = obj_distance(tcp_pos, ball_pos)
        ball_goal_dist, _ = obj_distance(ball_pos, goal_center_pos)

        ball_touching_penalty = 15 * tcp_ball_dist if tcp_ball_dist > 0.1 else 0
        behind_goal_penalty = 100 if ball_pos[1] > 0.44 else 0

        reward = - 50 * ball_goal_dist - ball_touching_penalty - behind_goal_penalty

        return np.minimum(reward, 0)

    def _check_early_termination(self) -> bool:

        ball_pos = self.scene.get_obj_pos(self.soccer_ball)

        if self.goal_box.contains(ball_pos):
            self.terminated = True
            return True

        return False

    def _reset_env(self):

        if self.random_ball_pos:
            new_ball = [self.soccer_ball, self.ball_pos_space.sample()]
            self.scene.reset([new_ball])
        else:
            self.scene.reset()

    def debug_msg(self) -> dict:
        ball_pos = self.scene.get_obj_pos(self.soccer_ball)
        return {"is_success": self.goal_box.contains(ball_pos)}