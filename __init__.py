
from gym.envs.registration import register
from envs.simple_reach_env import ReachEnv
from envs.door_open_env import DoorOpenEnv

register(
    id="ReachEnv-v0",
    entry_point="envs.simple_reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v1",
    entry_point="envs.simple_reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="ReachEnv-v2",
    entry_point="envs.simple_reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v3",
    entry_point="envs.simple_reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="ReachEnv-v4",
    entry_point="envs.simple_reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": True,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v5",
    entry_point="envs.simple_reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": True,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="DoorOpenEnv-v0",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="DoorOpenEnv-v1",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False}
)
