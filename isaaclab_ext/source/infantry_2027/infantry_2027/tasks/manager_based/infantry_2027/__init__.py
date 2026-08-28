import gymnasium as gym

from . import agents


gym.register(
    id="Infantry-2027-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_env_cfg:Infantry2027FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027PPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_terrain_env_cfg:Infantry2027TerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainPPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Terrain-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_terrain_env_cfg:Infantry2027TerrainPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainPPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_env_cfg:Infantry2027FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027PPORunnerCfg",
    },
)

gym.register(
    id="Infantry-2027-Flat-Compatible-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027FlatCompatibleEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027FlatCompatiblePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Flat-Compatible-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027FlatCompatiblePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027FlatCompatiblePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Terrain-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027TerrainV1EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainV1PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Terrain-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v1_env_cfg:Infantry2027TerrainV1PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainV1PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Flat-Stable-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v2_env_cfg:Infantry2027FlatStableEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027FlatStablePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Flat-Stable-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v2_env_cfg:Infantry2027FlatStablePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027FlatStablePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Terrain-Stable-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v2_terrain_env_cfg:Infantry2027TerrainStableEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainStablePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Terrain-Stable-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v2_terrain_env_cfg:Infantry2027TerrainStablePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027TerrainStablePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Joint-Terrain-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v3_env_cfg:Infantry2027JointTerrainEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027JointTerrainPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Joint-Terrain-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v3_env_cfg:Infantry2027JointTerrainPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027JointTerrainPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Joint-Fudan-Terrain-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v4_env_cfg:Infantry2027JointFudanTerrainEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027JointFudanTerrainPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Infantry-2027-Joint-Fudan-Terrain-Play-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.infantry_2027_v4_env_cfg:Infantry2027JointFudanTerrainPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Infantry2027JointFudanTerrainPPORunnerCfg"
        ),
    },
)
