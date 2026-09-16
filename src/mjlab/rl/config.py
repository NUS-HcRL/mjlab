"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class RslRlPpoActorCriticCfg:
  """Config for the PPO actor-critic networks."""

  init_noise_std: float = 1.0
  """The initial noise standard deviation of the policy."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """The type of noise standard deviation for the policy. Default is scalar."""
  actor_obs_normalization: bool = False
  """Whether to normalize the observation for the actor network. Default is False."""
  critic_obs_normalization: bool = False
  """Whether to normalize the observation for the critic network. Default is False."""
  actor_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the actor network."""
  critic_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the critic network."""
  activation: str = "elu"
  """The activation function to use in the actor and critic networks."""
  class_name: str = "ActorCritic"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  class_name: str = "PPO"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO + AMP config for use with amp-rsl-rl (algorithm class from that package)."""

  class_name: str = "amp_rsl_rl.algorithms.amp_ppo.AMP_PPO"
  """Algorithm class for amp-rsl-rl. Adjust if your amp-rsl-rl version uses a different path."""

  # AMP weights
  task_reward_weight: float = 1.0
  """Weight for the task (environment) reward."""
  disc_reward_weight: float = 1.0
  """Weight for the discriminator-based style reward."""
  disc_reward_scale: float = 2.0
  """Scale for the disc reward: -log(1 - D(s)) * scale."""
  reward_mix_mode: Literal["legacy", "ema_balance"] = "legacy"
  """How to mix task/style rewards in runner. `ema_balance` auto-matches magnitudes."""
  reward_mix_ema_decay: float = 0.99
  """EMA decay for reward magnitude tracking used by `ema_balance`."""
  reward_mix_scale_clip: Tuple[float, float] = (0.2, 5.0)
  """Clamp range for style auto-scale ratio in `ema_balance`."""
  reward_mix_task_abs_clip: float = 5.0
  """Upper bound applied to task-reward magnitudes used by the mixing EMA."""
  invalid_physics_termination_term: str | None = None
  """Termination term whose samples are excluded from reward mixing statistics."""
  invalid_physics_terminal_reward: float = -5.0
  """Final PPO reward assigned to invalid-physics terminal transitions."""

  # Discriminator training (MimicKit-style: fewer epochs, small batch, limited replay)
  disc_epochs: int = 2
  """Number of discriminator update epochs per PPO update (MimicKit uses 2)."""
  disc_batch_size_scale: float = 2.0 / 24.0
  """Disc batch size = this * num_envs (MimicKit: 2*num_envs; with 24 steps ~2/24)."""
  disc_replay_samples: int = 1000
  """Max samples from replay buffer per disc update (MimicKit: 1000; 0 = use all)."""
  disc_replay_buffer_size: int = 200000
  """Replay buffer size for past agent disc_obs (MimicKit: 200000)."""
  disc_lr: float = 2.5e-4
  """Learning rate for the discriminator optimizer (MimicKit: 2.5e-4)."""
  disc_grad_penalty: float = 5.0
  """Gradient penalty coefficient for discriminator (MimicKit: 5)."""
  disc_logit_reg: float = 0.01
  """L2 regularization on discriminator logit weights (MimicKit: 0.01)."""
  disc_hidden_dims: Tuple[int, ...] = (1024, 1024)
  """Hidden layer sizes for the discriminator MLP (MimicKit: 2x1024)."""
  disc_input_noise_std: float = 0.05
  """Std of Gaussian instance noise added to discriminator inputs during training."""
  disc_eval_batch_size: int = 0
  """Minibatch size for disc reward eval (0 = no minibatch)."""
  kl_early_stop: bool = False
  """Whether to stop PPO minibatch updates early when KL exceeds the configured limit."""
  kl_early_stop_multiplier: float = 2.0
  """Early-stop threshold multiplier applied to desired_kl."""
  actor_freeze_iterations: int = 0
  """Number of finetune updates that train only the critic/value loss."""
  freeze_discriminator: bool = False
  """Whether to keep the AMP discriminator fixed during training."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"policy": ("policy",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """The experiment name."""
  run_name: str = ""
  """The run name. Default is empty string."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  load_optimizer: bool = True
  """Whether to load optimizer state when resuming from a checkpoint."""
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  policy: RslRlPpoActorCriticCfg = field(default_factory=RslRlPpoActorCriticCfg)
  """The policy configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""
