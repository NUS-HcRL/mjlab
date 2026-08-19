"""RL configuration for PM1 falling task."""

from mjlab.rl import (
  RslRlAmpAlgorithmCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def pm1_falling_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PM1 falling task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pm1_falling",
    save_interval=2000,
    num_steps_per_env=24,
    max_iterations=40_000,
  )

def pm1_falling_amp_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PM1 falling task with AMP (MimicKit-style for speed)."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=0.1,
      noise_std_type="log",
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlAmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.15,
      entropy_coef=0.001,
      num_learning_epochs=4,
      num_mini_batches=2,
      learning_rate=7.5e-5,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      task_reward_weight=1.0,
      disc_reward_weight=1.0,
      reward_mix_mode="ema_balance",
      reward_mix_ema_decay=0.99,
      reward_mix_scale_clip=(0.05, 4.0),
      disc_reward_scale=2.0,
      disc_epochs=1,
      disc_batch_size_scale=2.0 / 32.0,
      disc_replay_samples=1000,
      disc_replay_buffer_size=200000,
      disc_lr=1.0e-4,
      disc_grad_penalty=10.0,
      disc_logit_reg=0.01,
      disc_input_noise_std=0.02,
      disc_hidden_dims=(512, 512),
    ),
    experiment_name="pm1_falling_amp",
    save_interval=5000,
    num_steps_per_env=30,
    max_iterations=40_000,
  )


def pm1_falling_amp_protective_finetune_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create AMP runner config for PM1 protective-contact finetuning."""
  cfg = pm1_falling_amp_runner_cfg()
  assert isinstance(cfg.algorithm, RslRlAmpAlgorithmCfg)
  cfg.algorithm.learning_rate = 2.0e-5
  cfg.algorithm.clip_param = 0.1
  cfg.algorithm.desired_kl = 0.006
  cfg.algorithm.use_clipped_value_loss = True
  cfg.algorithm.kl_early_stop = True
  cfg.algorithm.kl_early_stop_multiplier = 2.0
  cfg.algorithm.actor_freeze_iterations = 300
  cfg.algorithm.freeze_discriminator = True
  cfg.experiment_name = "pm1_falling_amp_protective_finetune"
  cfg.save_interval = 1000
  cfg.max_iterations = 10_000
  cfg.load_optimizer = False
  return cfg
