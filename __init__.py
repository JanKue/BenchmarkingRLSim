from gym.envs.registration import register
from envs.reach_env import ReachEnv
from envs.door_open_env import DoorOpenEnv
from envs.hammer_env import HammerEnv
from envs.soccer_env import SoccerEnv

register(
    id="ReachEnv-v0",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "random_init": False,
            "render": False}
)

register(
    id="ReachEnv-v1",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "random_init": False,
            "render": True}
)

register(
    id="ReachEnv-v2",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": False,
            "render": False}
)

register(
    id="ReachEnv-v3",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": False,
            "render": True}
)

register(
    id="ReachEnv-v4",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": True,
            "render": False}
)

register(
    id="ReachEnv-v5",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": True,
            "render": True}
)

register(
    id="DoorOpenEnv-v0",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": True}
)

register(
    id="DoorOpenEnv-v1",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": False}
)

register(
    id="DoorOpenEnv-v2",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": True,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 30,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v3",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 30,
            "reward_hinge_exp": True}
)

register(
    id="HammerEnv-v0",
    entry_point="envs.hammer_env:HammerEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": True}
)

register(
    id="HammerEnv-v1",
    entry_point="envs.hammer_env:HammerEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": False}
)

register(
    id="SoccerEnv-v0",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=500,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": True,
            "random_ball_pos": False}
)

register(
    id="SoccerEnv-v1",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=500,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": False,
            "random_ball_pos": False}
)

register(
    id="SoccerEnv-v2",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=500,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": True,
            "random_ball_pos": True}
)

register(
    id="SoccerEnv-v3",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=500,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "render": False,
            "random_ball_pos": True}
)