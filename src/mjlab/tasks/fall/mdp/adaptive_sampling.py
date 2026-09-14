from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class _PriorityReplayBuffer:
  """Ring-buffer metadata for outcome-weighted replay sampling."""

  def __init__(self, capacity: int, device: str) -> None:
    self.capacity = max(int(capacity), 1)
    self.device = device
    self.size = 0
    self.cursor = 0
    self.priorities = torch.zeros(self.capacity, device=device)
    self.generations = torch.zeros(self.capacity, dtype=torch.long, device=device)

  def _allocate(self, count: int) -> torch.Tensor:
    indexes = (
      torch.arange(count, device=self.device, dtype=torch.long) + self.cursor
    ) % self.capacity
    self.priorities[indexes] = 1.0
    self.generations[indexes] += 1
    self.cursor = (self.cursor + count) % self.capacity
    self.size = min(self.size + count, self.capacity)
    return indexes

  def _sampling_probabilities(self, uniform_ratio: float) -> torch.Tensor:
    priorities = self.priorities[: self.size].clamp_min(0.0)
    uniform = torch.full_like(priorities, 1.0 / float(self.size))
    priority_prob = priorities / priorities.sum().clamp_min(1e-12)
    priority_prob = torch.where(priorities.sum() > 1e-12, priority_prob, uniform)
    uniform_ratio = min(max(float(uniform_ratio), 0.0), 1.0)
    return (1.0 - uniform_ratio) * priority_prob + uniform_ratio * uniform

  def _sample_indexes(
    self, count: int, uniform_ratio: float
  ) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = self._sampling_probabilities(uniform_ratio)
    indexes = torch.multinomial(probabilities, count, replacement=True)
    return indexes, self.generations[indexes].clone()

  def update_priorities(
    self,
    indexes: torch.Tensor,
    generations: torch.Tensor,
    failed: torch.Tensor,
    alpha: float,
  ) -> None:
    """EMA-update sampled slots, ignoring slots overwritten in the meantime."""
    if indexes.numel() == 0:
      return
    safe_indexes = indexes.clamp(0, self.capacity - 1)
    valid = (
      (indexes >= 0)
      & (indexes < self.size)
      & (self.generations[safe_indexes] == generations)
    )
    indexes = indexes[valid]
    failed = failed[valid].float()
    if indexes.numel() == 0:
      return
    unique_indexes, inverse = torch.unique(indexes, return_inverse=True)
    failure_sum = torch.zeros(
      len(unique_indexes), device=self.device, dtype=torch.float32
    )
    sample_count = torch.zeros_like(failure_sum)
    failure_sum.scatter_add_(0, inverse, failed)
    sample_count.scatter_add_(0, inverse, torch.ones_like(failed))
    observed_failure_rate = failure_sum / sample_count.clamp_min(1.0)
    alpha = min(max(float(alpha), 0.0), 1.0)
    self.priorities[unique_indexes] = (1.0 - alpha) * self.priorities[
      unique_indexes
    ] + alpha * observed_failure_rate

  def sampling_stats(
    self, uniform_ratio: float
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.size == 0:
      zero = torch.zeros((), device=self.device)
      return zero, zero, zero
    probabilities = self._sampling_probabilities(uniform_ratio)
    top1 = probabilities.max()
    mean_priority = self.priorities[: self.size].mean()
    if self.size == 1:
      return torch.ones((), device=self.device), top1, mean_priority
    entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    normalized_entropy = entropy / torch.log(
      torch.tensor(float(self.size), device=self.device)
    )
    return normalized_entropy, top1, mean_priority


class _DataFailureReplayBuffer(_PriorityReplayBuffer):
  """Fixed-size priority buffer containing failed data-pool row IDs."""

  def __init__(self, capacity: int, device: str) -> None:
    super().__init__(capacity, device)
    self.state_ids = torch.full((self.capacity,), -1, dtype=torch.long, device=device)

  def add(self, state_ids: torch.Tensor) -> None:
    state_ids = state_ids[state_ids >= 0]
    count = int(state_ids.numel())
    if count == 0:
      return
    if count >= self.capacity:
      state_ids = state_ids[-self.capacity :]
      count = self.capacity
    indexes = self._allocate(count)
    self.state_ids[indexes] = state_ids

  def sample(
    self, count: int, uniform_ratio: float
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indexes, generations = self._sample_indexes(count, uniform_ratio)
    return self.state_ids[indexes].clone(), indexes, generations


class _RandomFailureReplayBuffer(_PriorityReplayBuffer):
  """Fixed-size priority buffer containing failed random-reset conditions."""

  def __init__(self, capacity: int, num_joints: int, device: str) -> None:
    super().__init__(capacity, device)
    self.data = {
      "root_state_rel": torch.zeros(self.capacity, 13, device=device),
      "joint_pos": torch.zeros(self.capacity, num_joints, device=device),
      "joint_vel": torch.zeros(self.capacity, num_joints, device=device),
      "reset_push": torch.zeros(self.capacity, 6, device=device),
      "reset_wrench": torch.zeros(self.capacity, 6, device=device),
      "reset_pulse_duration": torch.zeros(
        self.capacity, 1, dtype=torch.long, device=device
      ),
    }

  def add(self, records: dict[str, torch.Tensor]) -> None:
    count = int(records["root_state_rel"].shape[0])
    if count == 0:
      return
    if count >= self.capacity:
      records = {name: value[-self.capacity :] for name, value in records.items()}
      count = self.capacity
    indexes = self._allocate(count)
    for name, value in records.items():
      self.data[name][indexes] = value

  def sample(
    self, count: int, uniform_ratio: float
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    indexes, generations = self._sample_indexes(count, uniform_ratio)
    samples = {name: value[indexes].clone() for name, value in self.data.items()}
    return samples, indexes, generations


class FallAdaptiveResetSampler:
  """Replay safety-critical fall reset conditions with local perturbations.

  Data failures only retain the source data-row ID. Random failures retain the
  root state, joint state, and reset disturbance (velocity push or force-pulse
  wrench and duration). Fresh failures enter a ring buffer at maximum priority.
  Replayed samples update that priority from their next outcome using an EMA,
  while a uniform mixture keeps every stored failure reachable. Replayed
  failures update their source priority instead of being inserted again,
  preventing a self-reinforcing duplicate loop.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    capacity: int,
    replay_probability: float,
    min_failures: int,
    neighbor_scale: float,
    uniform_ratio: float = 0.2,
    priority_alpha: float = 0.05,
  ) -> None:
    self.num_envs = env.num_envs
    self.num_joints = env.scene["robot"].num_joints
    self.device = env.device
    per_source_capacity = max(int(capacity) // 2, 1)
    self.data_failures = _DataFailureReplayBuffer(per_source_capacity, self.device)
    self.random_failures = _RandomFailureReplayBuffer(
      per_source_capacity, self.num_joints, self.device
    )
    self.replay_probability = min(max(float(replay_probability), 0.0), 1.0)
    self.min_failures = max(int(min_failures), 1)
    self.neighbor_scale = max(float(neighbor_scale), 0.0)
    self.uniform_ratio = min(max(float(uniform_ratio), 0.0), 1.0)
    self.priority_alpha = min(max(float(priority_alpha), 0.0), 1.0)

    self.current_valid = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.current_data_mask = torch.zeros_like(self.current_valid)
    self.current_replayed = torch.zeros_like(self.current_valid)
    self.current_replay_index = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.current_replay_generation = torch.full_like(self.current_replay_index, -1)
    self.current_root_state_rel = torch.zeros(self.num_envs, 13, device=self.device)
    self.current_joint_pos = torch.zeros(
      self.num_envs, self.num_joints, device=self.device
    )
    self.current_joint_vel = torch.zeros_like(self.current_joint_pos)
    self.current_data_state_id = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self.current_reset_push = torch.zeros(self.num_envs, 6, device=self.device)
    self.replay_reset_push = torch.zeros_like(self.current_reset_push)
    self.current_reset_wrench = torch.zeros(self.num_envs, 6, device=self.device)
    self.replay_reset_wrench = torch.zeros_like(self.current_reset_wrench)
    self.current_reset_pulse_duration = torch.zeros(
      self.num_envs, 1, dtype=torch.long, device=self.device
    )
    self.replay_reset_pulse_duration = torch.zeros_like(
      self.current_reset_pulse_duration
    )

    self.last_failure_count = torch.zeros((), device=self.device)
    self.last_outcome_count = torch.zeros((), device=self.device)
    self.last_failure_rate = torch.zeros((), device=self.device)
    self.last_data_replay_rate = torch.zeros((), device=self.device)
    self.last_random_replay_rate = torch.zeros((), device=self.device)
    self.last_data_replay_failure_rate = torch.zeros((), device=self.device)
    self.last_random_replay_failure_rate = torch.zeros((), device=self.device)

  def capture_failures(
    self,
    env_ids: torch.Tensor,
    failure_mask: torch.Tensor,
    outcome_mask: torch.Tensor | None = None,
  ) -> None:
    """Record target failures and update replay priority on known outcomes.

    ``outcome_mask`` should include target failures and successful timeouts.
    Other terminal causes are ignored rather than incorrectly treated as replay
    successes. Passing ``None`` evaluates every supplied episode.
    """
    if outcome_mask is None:
      outcome_mask = torch.ones_like(failure_mask)
    valid_mask = self.current_valid[env_ids] & outcome_mask
    valid_ids = env_ids[valid_mask]
    valid_failed = failure_mask[valid_mask]
    failed_ids = valid_ids[valid_failed]
    self.last_outcome_count = torch.tensor(float(valid_ids.numel()), device=self.device)
    self.last_failure_count = torch.tensor(
      float(failed_ids.numel()), device=self.device
    )
    self.last_failure_rate = (
      valid_failed.float().mean()
      if valid_failed.numel() > 0
      else torch.zeros((), device=self.device)
    )
    self.last_data_replay_failure_rate.zero_()
    self.last_random_replay_failure_rate.zero_()
    if valid_ids.numel() == 0:
      return

    replayed = self.current_replayed[valid_ids]
    data_mask = self.current_data_mask[valid_ids]
    data_replayed = replayed & data_mask
    random_replayed = replayed & ~data_mask
    self.data_failures.update_priorities(
      self.current_replay_index[valid_ids[data_replayed]],
      self.current_replay_generation[valid_ids[data_replayed]],
      valid_failed[data_replayed],
      self.priority_alpha,
    )
    self.random_failures.update_priorities(
      self.current_replay_index[valid_ids[random_replayed]],
      self.current_replay_generation[valid_ids[random_replayed]],
      valid_failed[random_replayed],
      self.priority_alpha,
    )
    self.last_data_replay_failure_rate = self._masked_mean(valid_failed, data_replayed)
    self.last_random_replay_failure_rate = self._masked_mean(
      valid_failed, random_replayed
    )

    fresh_failed = valid_failed & ~replayed
    fresh_data_ids = valid_ids[fresh_failed & data_mask]
    self.data_failures.add(self.current_data_state_id[fresh_data_ids])

    random_ids = valid_ids[fresh_failed & ~data_mask]
    self.random_failures.add(
      {
        "root_state_rel": self.current_root_state_rel[random_ids],
        "joint_pos": self.current_joint_pos[random_ids],
        "joint_vel": self.current_joint_vel[random_ids],
        "reset_push": self.current_reset_push[random_ids],
        "reset_wrench": self.current_reset_wrench[random_ids],
        "reset_pulse_duration": self.current_reset_pulse_duration[random_ids],
      }
    )

  def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (values.float() * weights).sum() / weights.sum().clamp_min(1.0)

  def sample_for_sources(
    self,
    env_ids: torch.Tensor,
    data_mask: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    count = len(env_ids)
    use_replay = torch.zeros(count, dtype=torch.bool, device=self.device)
    samples = {
      "root_state_rel": torch.zeros(count, 13, device=self.device),
      "joint_pos": torch.zeros(count, self.num_joints, device=self.device),
      "joint_vel": torch.zeros(count, self.num_joints, device=self.device),
      "data_state_id": torch.full((count,), -1, dtype=torch.long, device=self.device),
      "reset_push": torch.zeros(count, 6, device=self.device),
      "reset_wrench": torch.zeros(count, 6, device=self.device),
      "reset_pulse_duration": torch.zeros(
        count, 1, dtype=torch.long, device=self.device
      ),
      "replay_index": torch.full((count,), -1, dtype=torch.long, device=self.device),
      "replay_generation": torch.full(
        (count,), -1, dtype=torch.long, device=self.device
      ),
    }

    data_local_ids = torch.nonzero(data_mask, as_tuple=False).squeeze(-1)
    if data_local_ids.numel() > 0 and self.data_failures.size >= self.min_failures:
      replay_local = data_local_ids[
        torch.rand(len(data_local_ids), device=self.device) < self.replay_probability
      ]
      if replay_local.numel() > 0:
        use_replay[replay_local] = True
        state_ids, indexes, generations = self.data_failures.sample(
          len(replay_local), self.uniform_ratio
        )
        samples["data_state_id"][replay_local] = state_ids
        samples["replay_index"][replay_local] = indexes
        samples["replay_generation"][replay_local] = generations

    random_local_ids = torch.nonzero(~data_mask, as_tuple=False).squeeze(-1)
    if random_local_ids.numel() > 0 and self.random_failures.size >= self.min_failures:
      replay_local = random_local_ids[
        torch.rand(len(random_local_ids), device=self.device) < self.replay_probability
      ]
      if replay_local.numel() > 0:
        use_replay[replay_local] = True
        replay_samples, indexes, generations = self.random_failures.sample(
          len(replay_local), self.uniform_ratio
        )
        for name, value in replay_samples.items():
          samples[name][replay_local] = value
        samples["replay_index"][replay_local] = indexes
        samples["replay_generation"][replay_local] = generations

    data_count = data_mask.sum().clamp_min(1)
    random_count = (~data_mask).sum().clamp_min(1)
    self.last_data_replay_rate = (
      use_replay & data_mask
    ).sum().float() / data_count.float()
    self.last_random_replay_rate = (
      use_replay & ~data_mask
    ).sum().float() / random_count.float()
    samples["use_replay"] = use_replay
    return samples

  def begin_episodes(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    data_mask: torch.Tensor,
    replay_samples: dict[str, torch.Tensor],
    root_state: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    data_state_ids: torch.Tensor,
  ) -> None:
    root_state_rel = root_state.clone()
    root_state_rel[:, 0:3] -= env.scene.env_origins[env_ids]
    self.current_valid[env_ids] = True
    self.current_data_mask[env_ids] = data_mask
    self.current_replayed[env_ids] = replay_samples["use_replay"]
    self.current_replay_index[env_ids] = replay_samples["replay_index"]
    self.current_replay_generation[env_ids] = replay_samples["replay_generation"]
    self.current_root_state_rel[env_ids] = root_state_rel
    self.current_joint_pos[env_ids] = joint_pos
    self.current_joint_vel[env_ids] = joint_vel
    self.current_data_state_id[env_ids] = data_state_ids
    self.current_reset_push[env_ids] = 0.0
    self.replay_reset_push[env_ids] = replay_samples["reset_push"]
    self.current_reset_wrench[env_ids] = 0.0
    self.replay_reset_wrench[env_ids] = replay_samples["reset_wrench"]
    self.current_reset_pulse_duration[env_ids] = 0
    self.replay_reset_pulse_duration[env_ids] = replay_samples["reset_pulse_duration"]

  def record_reset_push(
    self, env_ids: torch.Tensor, velocity_delta: torch.Tensor
  ) -> None:
    self.current_reset_push[env_ids] = velocity_delta

  def record_reset_force_pulse(
    self,
    env_ids: torch.Tensor,
    wrench: torch.Tensor,
    duration_steps: torch.Tensor,
  ) -> None:
    """Record the sampled reset wrench and its per-environment duration."""
    self.current_reset_wrench[env_ids] = wrench
    self.current_reset_pulse_duration[env_ids, 0] = duration_steps.long()

  def metrics(self) -> dict[str, torch.Tensor]:
    data_entropy, data_top1, data_mean_priority = self.data_failures.sampling_stats(
      self.uniform_ratio
    )
    random_entropy, random_top1, random_mean_priority = (
      self.random_failures.sampling_stats(self.uniform_ratio)
    )
    metrics = {
      "Metrics/fall_adaptive/failures_captured": self.last_failure_count,
      "Metrics/fall_adaptive/outcomes_evaluated": self.last_outcome_count,
      "Metrics/fall_adaptive/failure_rate": self.last_failure_rate,
      "Metrics/fall_adaptive/data_buffer_size": torch.tensor(
        float(self.data_failures.size), device=self.device
      ),
      "Metrics/fall_adaptive/random_buffer_size": torch.tensor(
        float(self.random_failures.size), device=self.device
      ),
      "Metrics/fall_adaptive/data_replay_rate": self.last_data_replay_rate,
      "Metrics/fall_adaptive/random_replay_rate": self.last_random_replay_rate,
      "Metrics/fall_adaptive/data_replay_failure_rate": (
        self.last_data_replay_failure_rate
      ),
      "Metrics/fall_adaptive/random_replay_failure_rate": (
        self.last_random_replay_failure_rate
      ),
      "Metrics/fall_adaptive/data_sampling_entropy": data_entropy,
      "Metrics/fall_adaptive/random_sampling_entropy": random_entropy,
      "Metrics/fall_adaptive/data_top1_probability": data_top1,
      "Metrics/fall_adaptive/random_top1_probability": random_top1,
      "Metrics/fall_adaptive/data_mean_priority": data_mean_priority,
      "Metrics/fall_adaptive/random_mean_priority": random_mean_priority,
    }
    # Event logs can outlive the reset call that produced them. Snapshot mutable
    # sampler tensors so a later priority update cannot rewrite older log rows.
    return {name: value.detach().clone() for name, value in metrics.items()}
