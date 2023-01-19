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
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v1",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": False,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="ReachEnv-v2",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v3",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="ReachEnv-v4",
    entry_point="envs.reach_env:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_goal": True,
            "random_init": True,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="ReachEnv-v5",
    entry_point="envs.reach_env:ReachEnv",
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
    max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="DoorOpenEnv-v1",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="HammerEnv-v0",
    entry_point="envs.hammer_env:HammerEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True}
)

register(
    id="HammerEnv-v1",
    entry_point="envs.hammer_env:HammerEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False}
)

register(
    id="SoccerEnv-v0",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=500,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True,
            "random_ball_pos": False}
)

register(
    id="SoccerEnv-v1",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=500,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False,
            "random_ball_pos": False}
)

register(
    id="SoccerEnv-v2",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": True,
            "random_ball_pos": True}
)

register(
    id="SoccerEnv-v3",
    entry_point="envs.soccer_env:SoccerEnv",
    max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "random_init": False,
            "simulator": 'mujoco',
            "render": False,
            "random_ball_pos": True}
)

# testing different env reward options

register(
    id="DoorOpenEnv-v10",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 4,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v11",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 10,
            "reward_hand_penalty_ratio": 4,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v12",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 1,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v13",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 10,
            "reward_hand_penalty_ratio": 1,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v14",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 10,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v15",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 10,
            "reward_hand_penalty_ratio": 10,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v22",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 10,
            "reward_hand_penalty_ratio": 20,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v23",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 20,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v24",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 10,
            "reward_hand_penalty_ratio": 50,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v25",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 10,
            "reward_hand_penalty_ratio": 15,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v26",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 25,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v27",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 25,
            "reward_hand_penalty_ratio": 30,
            "reward_hinge_exp": True}
)

register(
    id="DoorOpenEnv-v28",
    entry_point="envs.door_open_env:DoorOpenEnv",
    max_episode_steps=625,
    kwargs={"render": False,
            "reward_multiplier": 30,
            "reward_hand_penalty_ratio": 25,
            "reward_hinge_exp": True}
)
