from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from amp_rsl_rl.networks import Discriminator
from amp_rsl_rl.storage import ReplayBuffer
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class _CombinedOptimizer:
  """Compatibility wrapper so runner save/load can handle two optimizers."""

  def __init__(self, policy_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer):
    self.policy_optimizer = policy_optimizer
    self.disc_optimizer = disc_optimizer

  @property
  def param_groups(self):
    return self.policy_optimizer.param_groups

  def state_dict(self) -> dict[str, Any]:
    return {
      "policy_optimizer": self.policy_optimizer.state_dict(),
      "disc_optimizer": self.disc_optimizer.state_dict(),
    }

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    if "policy_optimizer" in state_dict:
      self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
    if "disc_optimizer" in state_dict:
      self.disc_optimizer.load_state_dict(state_dict["disc_optimizer"])


class MjlabAmpPPO:
  """AMP PPO optimized for large batched training.

  Key differences from amp_rsl_rl.AMP_PPO:
  - PPO and discriminator are updated in separate loops.
  - Discriminator batch size is decoupled from PPO mini-batch size.
  - Config knobs such as disc_epochs/disc_batch_size_scale now actually apply.
  """

  actor_critic: ActorCritic
  _MIN_STD = 1e-6
  _MAX_STD = 10.0
  _MIN_LOG_STD = -12.0
  _MAX_LOG_STD = 2.0

  def __init__(
    self,
    actor_critic: ActorCritic,
    discriminator: Discriminator,
    amp_data,
    num_learning_epochs: int = 1,
    num_mini_batches: int = 1,
    clip_param: float = 0.2,
    gamma: float = 0.998,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.0,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "fixed",
    desired_kl: float = 0.01,
    amp_replay_buffer_size: int | None = None,
    disc_replay_buffer_size: int = 100000,
    use_smooth_ratio_clipping: bool = False,
    device: str = "cpu",
    task_reward_weight: float = 1.0,
    disc_reward_weight: float = 1.0,
    disc_epochs: int = 2,
    disc_batch_size_scale: float = 2.0 / 24.0,
    disc_replay_samples: int = 1000,
    disc_lr: float = 2.5e-4,
    disc_grad_penalty: float = 5.0,
    disc_logit_reg: float = 0.0,
    disc_input_noise_std: float = 0.0,
    disc_eval_batch_size: int = 0,
    normalize_advantage_per_mini_batch: bool = False,
    kl_early_stop: bool = False,
    kl_early_stop_multiplier: float = 2.0,
    actor_freeze_iterations: int = 0,
    freeze_discriminator: bool = False,
  ) -> None:
    self.device = device
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate

    self.discriminator = discriminator.to(self.device)
    self.amp_transition = RolloutStorage.Transition()
    obs_dim = self.discriminator.input_dim // 2
    buffer_size = amp_replay_buffer_size or disc_replay_buffer_size
    self.amp_storage = ReplayBuffer(obs_dim=obs_dim, buffer_size=buffer_size, device=device)
    self.amp_data = amp_data

    self.actor_critic = actor_critic.to(self.device)
    self.storage: Optional[RolloutStorage] = None
    self.transition = RolloutStorage.Transition()

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss
    self.use_smooth_ratio_clipping = use_smooth_ratio_clipping
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
    self.kl_early_stop = kl_early_stop
    self.kl_early_stop_multiplier = kl_early_stop_multiplier
    self.actor_freeze_iterations = actor_freeze_iterations
    self.freeze_discriminator = freeze_discriminator
    self._policy_update_count = 0

    self.task_reward_weight = task_reward_weight
    self.disc_reward_weight = disc_reward_weight
    self.disc_epochs = disc_epochs
    self.disc_batch_size_scale = disc_batch_size_scale
    self.disc_replay_samples = disc_replay_samples
    self.disc_grad_penalty = disc_grad_penalty
    self.disc_logit_reg = disc_logit_reg
    self.disc_input_noise_std = disc_input_noise_std
    self.disc_eval_batch_size = disc_eval_batch_size

    self.policy_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    disc_params = [
      {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-4},
      {"params": self.discriminator.linear.parameters(), "weight_decay": 1e-4},
    ]
    self.disc_optimizer = optim.Adam(disc_params, lr=disc_lr)
    self.optimizer = _CombinedOptimizer(self.policy_optimizer, self.disc_optimizer)
    self._sanitize_policy_noise()

  def _sanitize_policy_noise(self) -> None:
    """Keep policy noise parameters finite and in a safe range."""
    noise_std_type = getattr(self.actor_critic, "noise_std_type", "scalar")
    if noise_std_type == "log" and hasattr(self.actor_critic, "log_std"):
      log_std = self.actor_critic.log_std.data
      log_std.nan_to_num_(
        nan=math.log(0.05),
        posinf=self._MAX_LOG_STD,
        neginf=self._MIN_LOG_STD,
      )
      log_std.clamp_(min=self._MIN_LOG_STD, max=self._MAX_LOG_STD)
    elif hasattr(self.actor_critic, "std"):
      std = self.actor_critic.std.data
      std.nan_to_num_(
        nan=0.05,
        posinf=self._MAX_STD,
        neginf=self._MIN_STD,
      )
      std.clamp_(min=self._MIN_STD, max=self._MAX_STD)

  def init_storage(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    observations: TensorDict,
    action_shape: Tuple[int, ...],
  ) -> None:
    self.storage = RolloutStorage(
      training_type="rl",
      num_envs=num_envs,
      num_transitions_per_env=num_transitions_per_env,
      obs=observations,
      actions_shape=action_shape,
      device=self.device,
    )

  def test_mode(self) -> None:
    self.actor_critic.eval()
    self.discriminator.eval()

  def train_mode(self) -> None:
    self.actor_critic.train()
    self.discriminator.train()

  def act(self, obs: TensorDict) -> torch.Tensor:
    if self.actor_critic.is_recurrent:
      self.transition.hidden_states = self.actor_critic.get_hidden_states()
    self.transition.actions = self.actor_critic.act(obs).detach()
    self.transition.values = self.actor_critic.evaluate(obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    self.transition.observations = obs
    return self.transition.actions

  def act_amp(self, amp_obs: torch.Tensor) -> None:
    self.amp_transition.observations = amp_obs

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: Dict[str, Any],
  ) -> None:
    assert self.storage is not None
    self.actor_critic.update_normalization(obs)
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
      )
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor_critic.reset(dones)

  def process_amp_step(self, amp_obs: torch.Tensor) -> None:
    self.amp_storage.insert(self.amp_transition.observations, amp_obs)
    self.amp_transition.clear()

  def predict_style_reward(
    self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor
  ) -> torch.Tensor:
    if self.disc_eval_batch_size <= 0 or amp_obs.shape[0] <= self.disc_eval_batch_size:
      return self.discriminator.predict_reward(amp_obs, next_amp_obs)

    rewards = []
    for start in range(0, amp_obs.shape[0], self.disc_eval_batch_size):
      end = start + self.disc_eval_batch_size
      rewards.append(
        self.discriminator.predict_reward(amp_obs[start:end], next_amp_obs[start:end])
      )
    return torch.cat(rewards, dim=0)

  def compute_returns(self, obs: TensorDict) -> None:
    assert self.storage is not None
    last_values = self.actor_critic.evaluate(obs).detach()
    self.storage.compute_returns(
      last_values,
      self.gamma,
      self.lam,
      normalize_advantage=not self.normalize_advantage_per_mini_batch,
    )

  def _update_policy(self) -> tuple[float, float, float, float, float, float, float, float]:
    assert self.storage is not None
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_kl_divergence = 0.0
    mean_ratio = 0.0
    mean_ratio_std = 0.0
    mean_old_logp = 0.0
    mean_new_logp = 0.0
    mean_actor_grad_norm = 0.0
    kl_checks = 0
    actor_frozen = self._policy_update_count < self.actor_freeze_iterations

    if self.actor_critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    num_updates = 0
    for sample in generator:
      self._sanitize_policy_noise()
      (
        obs_batch,
        actions_batch,
        target_values_batch,
        advantages_batch,
        returns_batch,
        old_actions_log_prob_batch,
        old_mu_batch,
        old_sigma_batch,
        hidden_states_batch,
        masks_batch,
      ) = sample

      hidden_state_actor, hidden_state_critic = (None, None)
      if hidden_states_batch is not None:
        hidden_state_actor, hidden_state_critic = hidden_states_batch

      self.actor_critic.act(
        obs_batch, masks=masks_batch, hidden_states=hidden_state_actor
      )
      actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
      value_batch = self.actor_critic.evaluate(
        obs_batch, masks=masks_batch, hidden_states=hidden_state_critic
      )
      mu_batch = self.actor_critic.action_mean
      sigma_batch = self.actor_critic.action_std
      entropy_batch = self.actor_critic.entropy

      with torch.inference_mode():
        kl = torch.sum(
          torch.log((sigma_batch + 1.0e-5) / (old_sigma_batch + 1.0e-5))
          + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
          / (2.0 * torch.square(sigma_batch) + 1.0e-5)
          - 0.5,
          dim=-1,
        )
        kl_mean = torch.mean(kl)
        mean_kl_divergence += kl_mean.item()
        kl_checks += 1
        if (
          self.kl_early_stop
          and self.desired_kl is not None
          and kl_mean > self.desired_kl * self.kl_early_stop_multiplier
        ):
          break
        if self.desired_kl is not None and self.schedule == "adaptive":
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.policy_optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      if self.normalize_advantage_per_mini_batch:
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
          advantages_batch.std() + 1e-8
        )

      new_logp = actions_log_prob_batch.reshape(-1)
      old_logp = old_actions_log_prob_batch.reshape(-1)
      ratio = torch.exp(new_logp - old_logp)
      min_ = 1.0 - self.clip_param
      max_ = 1.0 + self.clip_param
      if self.use_smooth_ratio_clipping:
        clipped_ratio = (
          1
          / (1 + torch.exp((-(ratio - min_) / (max_ - min_) + 0.5) * 4))
          * (max_ - min_)
          + min_
        )
      else:
        clipped_ratio = torch.clamp(ratio, min_, max_)

      adv_batch = advantages_batch.reshape(-1)
      surrogate = -adv_batch * ratio
      surrogate_clipped = -adv_batch * clipped_ratio
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (
          value_batch - target_values_batch
        ).clamp(-self.clip_param, self.clip_param)
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      if actor_frozen:
        loss = self.value_loss_coef * value_loss
      else:
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_batch.mean()
        )

      self.policy_optimizer.zero_grad()
      loss.backward()
      actor_grad_norm = nn.utils.clip_grad_norm_(
        self.actor_critic.parameters(), self.max_grad_norm
      )
      self.policy_optimizer.step()
      self._sanitize_policy_noise()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_ratio += ratio.mean().item()
      mean_ratio_std += ratio.std(unbiased=False).item()
      mean_old_logp += old_logp.mean().item()
      mean_new_logp += new_logp.mean().item()
      mean_actor_grad_norm += 0.0 if actor_frozen else float(actor_grad_norm)
      num_updates += 1

    return (
      mean_value_loss / max(1, num_updates),
      mean_surrogate_loss / max(1, num_updates),
      mean_kl_divergence / max(1, kl_checks),
      mean_ratio / max(1, num_updates),
      mean_ratio_std / max(1, num_updates),
      mean_old_logp / max(1, num_updates),
      mean_new_logp / max(1, num_updates),
      mean_actor_grad_norm / max(1, num_updates),
    )

  def _update_discriminator(
    self,
  ) -> tuple[float, float, float, float, float, float, float, float, float]:
    assert self.storage is not None
    if self.freeze_discriminator:
      return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean_amp_loss = 0.0
    mean_grad_pen_loss = 0.0
    mean_policy_pred = 0.0
    mean_expert_pred = 0.0
    mean_accuracy_policy = 0.0
    mean_accuracy_expert = 0.0
    mean_policy_input_norm = 0.0
    mean_expert_input_norm = 0.0
    mean_pair_distance = 0.0
    mean_accuracy_policy_elem = 0.0
    mean_accuracy_expert_elem = 0.0

    batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
    disc_batch_size = max(1, int(batch_size * self.disc_batch_size_scale))
    num_samples = self.storage.num_envs * self.storage.num_transitions_per_env
    num_batches = max(1, math.ceil(num_samples / disc_batch_size))
    num_updates = max(1, num_batches * self.disc_epochs)
    for _ in range(num_updates):
      policy_state, policy_next_state = next(
        self.amp_storage.feed_forward_generator(
          num_mini_batch=1,
          mini_batch_size=disc_batch_size,
          allow_replacement=True,
        )
      )
      expert_state, expert_next_state = next(
        self.amp_data.feed_forward_generator(1, disc_batch_size)
      )

      policy_state = policy_state.to(self.device)
      policy_next_state = policy_next_state.to(self.device)
      expert_state = expert_state.to(self.device)
      expert_next_state = expert_next_state.to(self.device)

      policy_state_raw = policy_state.detach()
      policy_next_state_raw = policy_next_state.detach()
      expert_state_raw = expert_state.detach()
      expert_next_state_raw = expert_next_state.detach()

      b_policy = policy_state.size(0)
      policy_pair = torch.cat([policy_state, policy_next_state], dim=-1)
      expert_pair = torch.cat([expert_state, expert_next_state], dim=-1)
      mean_policy_input_norm += policy_pair.norm(dim=-1).mean().item()
      mean_expert_input_norm += expert_pair.norm(dim=-1).mean().item()
      mean_pair_distance += (policy_pair - expert_pair).norm(dim=-1).mean().item()
      if self.disc_input_noise_std > 0.0:
        policy_pair = policy_pair + self.disc_input_noise_std * torch.randn_like(policy_pair)
        expert_pair = expert_pair + self.disc_input_noise_std * torch.randn_like(expert_pair)
      disc_input = torch.cat((policy_pair, expert_pair), dim=0)
      disc_output = self.discriminator(disc_input)
      policy_d = disc_output[:b_policy]
      expert_d = disc_output[b_policy:]

      amp_loss, grad_pen_loss = self.discriminator.compute_loss(
        policy_d=policy_d,
        expert_d=expert_d,
        sample_amp_expert=(expert_state, expert_next_state),
        sample_amp_policy=(policy_state, policy_next_state),
        lambda_=self.disc_grad_penalty,
      )
      logit_reg_loss = torch.zeros((), device=self.device)
      if self.disc_logit_reg != 0.0:
        logit_reg_loss = self.discriminator.linear.weight.pow(2).sum()

      self.disc_optimizer.zero_grad()
      (amp_loss + grad_pen_loss + self.disc_logit_reg * logit_reg_loss).backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
      self.disc_optimizer.step()

      self.discriminator.update_normalization(
        expert_state_raw,
        expert_next_state_raw,
        policy_state_raw,
        policy_next_state_raw,
      )

      policy_d_prob = torch.sigmoid(policy_d)
      expert_d_prob = torch.sigmoid(expert_d)
      mean_amp_loss += amp_loss.item()
      mean_grad_pen_loss += grad_pen_loss.item()
      mean_policy_pred += policy_d_prob.mean().item()
      mean_expert_pred += expert_d_prob.mean().item()
      mean_accuracy_policy += torch.sum(
        torch.round(policy_d_prob) == torch.zeros_like(policy_d_prob)
      ).item()
      mean_accuracy_expert += torch.sum(
        torch.round(expert_d_prob) == torch.ones_like(expert_d_prob)
      ).item()
      mean_accuracy_policy_elem += policy_d_prob.numel()
      mean_accuracy_expert_elem += expert_d_prob.numel()

    return (
      mean_amp_loss / num_updates,
      mean_grad_pen_loss / num_updates,
      mean_policy_pred / num_updates,
      mean_expert_pred / num_updates,
      mean_accuracy_policy / max(1, mean_accuracy_policy_elem),
      mean_accuracy_expert / max(1, mean_accuracy_expert_elem),
      mean_policy_input_norm / max(1, num_updates),
      mean_expert_input_norm / max(1, num_updates),
      mean_pair_distance / max(1, num_updates),
    )

  def update(
    self,
  ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    assert self.storage is not None
    (
      mean_value_loss,
      mean_surrogate_loss,
      mean_kl_divergence,
      mean_ratio,
      mean_ratio_std,
      mean_old_logp,
      mean_new_logp,
      mean_actor_grad_norm,
    ) = self._update_policy()
    (
      mean_amp_loss,
      mean_grad_pen_loss,
      mean_policy_pred,
      mean_expert_pred,
      mean_accuracy_policy,
      mean_accuracy_expert,
      mean_policy_input_norm,
      mean_expert_input_norm,
      mean_pair_distance,
    ) = self._update_discriminator()
    self._policy_update_count += 1
    self.storage.clear()
    return (
      mean_value_loss,
      mean_surrogate_loss,
      mean_amp_loss,
      mean_grad_pen_loss,
      mean_policy_pred,
      mean_expert_pred,
      mean_accuracy_policy,
      mean_accuracy_expert,
      mean_kl_divergence,
      mean_ratio,
      mean_ratio_std,
      mean_old_logp,
      mean_new_logp,
      mean_actor_grad_norm,
      mean_policy_input_norm,
      mean_expert_input_norm,
      mean_pair_distance,
    )
