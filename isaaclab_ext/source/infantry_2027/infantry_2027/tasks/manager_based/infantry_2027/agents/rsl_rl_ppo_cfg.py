"""Fudan PPO and supervised velocity-estimator configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class EstimatorActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "InfantryEstimatorActorCritic"
    encoder_hidden_dims: list[int] = [128, 64]
    history_length: int = 5
    single_frame_dim: int = 25
    latent_dim: int = 3


@configclass
class EstimatorPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "InfantryEstimatorPPO"
    extra_learning_rate: float = 1.0e-3


@configclass
class Infantry2027PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "InfantryOnPolicyRunner"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    num_steps_per_env = 48
    max_iterations = 5000
    save_interval = 100
    experiment_name = "infantry_2027_v0_flat"
    empirical_normalization = False
    clip_actions = 100.0
    policy = EstimatorActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        encoder_hidden_dims=[128, 64],
        history_length=5,
        single_frame_dim=25,
        latent_dim=3,
        activation="elu",
    )
    algorithm = EstimatorPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        extra_learning_rate=1.0e-3,
    )


@configclass
class Infantry2027TerrainPPORunnerCfg(Infantry2027PPORunnerCfg):
    experiment_name = "infantry_2027_v0_terrain"
    # Terrain simulation has a larger host-memory footprint.  More frequent
    # checkpoints make a run recoverable when another GPU application forces
    # Windows to page out or terminate Kit; this does not alter optimization.
    save_interval = 20


@configclass
class Infantry2027FlatCompatiblePPORunnerCfg(Infantry2027PPORunnerCfg):
    """Two-thousand-update flat precursor with Fudan's final PPO settings."""

    experiment_name = "infantry_2027_v1_flat_compatible"
    max_iterations = 2000
    save_interval = 50
    policy = EstimatorActorCriticCfg(
        # Fudan's actual from-scratch entry point inherits 0.5 from
        # LeggedRobotCfgPPO.policy.  The value 1.0 only appears in several
        # later terrain/recovery log snapshots and is not the flat bootstrap.
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        encoder_hidden_dims=[128, 64],
        history_length=5,
        single_frame_dim=25,
        latent_dim=3,
        activation="elu",
    )


@configclass
class Infantry2027TerrainV1PPORunnerCfg(Infantry2027FlatCompatiblePPORunnerCfg):
    experiment_name = "infantry_2027_v1_terrain"
    max_iterations = 5000
    save_interval = 20


@configclass
class Infantry2027FlatStablePPORunnerCfg(Infantry2027FlatCompatiblePPORunnerCfg):
    """Stable five-thousand-update flat precursor for the VMC action contract."""

    experiment_name = "infantry_2027_v2_flat_stable"
    max_iterations = 5000
    save_interval = 50


@configclass
class Infantry2027TerrainStablePPORunnerCfg(Infantry2027FlatStablePPORunnerCfg):
    """Terrain continuation from the accepted Flat-Stable-v2 checkpoint."""

    experiment_name = "infantry_2027_v2_terrain_stable"
    max_iterations = 5000
    save_interval = 20
