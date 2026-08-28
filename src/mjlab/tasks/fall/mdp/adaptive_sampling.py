from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class _DataFailureReplayBuffer:
  """Fixed-size ring buffer containing failed data-pool row IDs."""

  def __init__(self, capacity: int, device: str) -> None:
    self.capacity = max(int(capacity), 1)
    self.device = device
    self.size = 0
    self.cursor = 0
    self.state_ids = torch.full((self.capacity,), -1, dtype=torch.long, device=device)

  def add(self, state_ids: torch.Tensor) -> None:
    state_ids = state_ids[state_ids >= 0]
    count = int(state_ids.numel())
    if count == 0:
      return
    if count >= self.capacity:
      state_ids = state_ids[-self.capacity :]
      count = self.capacity
    indexes = (
      torch.arange(count, device=self.device, dtype=torch.long) + self.cursor
    ) % self.capacity
    self.state_ids[indexes] = state_ids
    self.cursor = (self.cursor + count) % self.capacity
    self.size = min(self.size + count, self.capacity)

  def sample(self, count: int) -> torch.Tensor:
    indexes = torch.randint(0, self.size, (count,), device=self.device)
    return self.state_ids[indexes].clone()


class _RandomFailureReplayBuffer:
  """Fixed-size ring buffer containing failed random-reset conditions."""

  def __init__(self, capacity: int, num_joints: int, device: str) -> None:
    self.capacity = max(int(capacity), 1)
    self.device = device
    self.size = 0
    self.cursor = 0
    self.data = {
      "root_state_rel": torch.zeros(self.capacity, 13, device=device),
      "joint_pos": torch.zeros(self.capacity, num_joints, device=device),
      "joint_vel": torch.zeros(self.capacity, num_joints, device=device),
      "reset_push": torch.zeros(self.capacity, 6, device=device),
    }

  def add(self, records: dict[str, torch.Tensor]) -> None:
    count = int(records["root_state_rel"].shape[0])
    if count == 0:
      return
    if count >= self.capacity:
      records = {name: value[-self.capacity :] for name, value in records.items()}
      count = self.capacity
    indexes = (
      torch.arange(count, device=self.device, dtype=torch.long) + self.cursor
    ) % self.capacity
    for name, value in records.items():
      self.data[name][indexes] = value
    self.cursor = (self.cursor + count) % self.capacity
    self.size = min(self.size + count, self.capacity)

  def sample(self, count: int) -> dict[str, torch.Tensor]:
    indexes = torch.randint(0, self.size, (count,), device=self.device)
    return {name: value[indexes].clone() for name, value in self.data.items()}


class FallAdaptiveResetSampler:
  """Replay safety-critical fall reset conditions with local perturbations.

  Data failures only retain the source data-row ID. Random failures retain the
  root state, joint state, and reset-time velocity push. Repeated failures are
  stored repeatedly, so their empirical failure frequency becomes their replay
  weight while ``replay_probability`` preserves uniform exploration.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    capacity: int,
    replay_probability: float,
    min_failures: int,
    neighbor_scale: float,
  ) -> None:
    self.num_envs = env.num_envs
    self.num_joints = env.scene["robot"].num_joints
    self.device = env.device
    per_source_capacity = max(int(capacity) // 2, 1)
    self.data_failures = _DataFailureReplayBuffer(per_source_capacity, self.device)
    self.random_failures = _RandomFailureReplayBuffer(
      per_source_capacity, self.num_joints, self.device
    )
    self.replay_probability = float(replay_probability)
    self.min_failures = max(int(min_failures), 1)
    self.neighbor_scale = max(float(neighbor_scale), 0.0)

    self.current_valid = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.current_data_mask = torch.zeros_like(self.current_valid)
    self.current_replayed = torch.zeros_like(self.current_valid)
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

    self.last_failure_count = torch.zeros((), device=self.device)
    self.last_data_replay_rate = torch.zeros((), device=self.device)
    self.last_random_replay_rate = torch.zeros((), device=self.device)

  def capture_failures(
    self,
    env_ids: torch.Tensor,
    failure_mask: torch.Tensor,
  ) -> None:
    selected = env_ids[failure_mask & self.current_valid[env_ids]]
    self.last_failure_count = torch.tensor(float(selected.numel()), device=self.device)
    if selected.numel() == 0:
      return

    data_selected = self.current_data_mask[selected]
    self.data_failures.add(self.current_data_state_id[selected[data_selected]])

    random_ids = selected[~data_selected]
    self.random_failures.add(
      {
        "root_state_rel": self.current_root_state_rel[random_ids],
        "joint_pos": self.current_joint_pos[random_ids],
        "joint_vel": self.current_joint_vel[random_ids],
        "reset_push": self.current_reset_push[random_ids],
      }
    )

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
    }

    data_local_ids = torch.nonzero(data_mask, as_tuple=False).squeeze(-1)
    if data_local_ids.numel() > 0 and self.data_failures.size >= self.min_failures:
      replay_local = data_local_ids[
        torch.rand(len(data_local_ids), device=self.device) < self.replay_probability
      ]
      if replay_local.numel() > 0:
        use_replay[replay_local] = True
        samples["data_state_id"][replay_local] = self.data_failures.sample(
          len(replay_local)
        )

    random_local_ids = torch.nonzero(~data_mask, as_tuple=False).squeeze(-1)
    if random_local_ids.numel() > 0 and self.random_failures.size >= self.min_failures:
      replay_local = random_local_ids[
        torch.rand(len(random_local_ids), device=self.device) < self.replay_probability
      ]
      if replay_local.numel() > 0:
        use_replay[replay_local] = True
        replay_samples = self.random_failures.sample(len(replay_local))
        for name, value in replay_samples.items():
          samples[name][replay_local] = value

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
    self.current_root_state_rel[env_ids] = root_state_rel
    self.current_joint_pos[env_ids] = joint_pos
    self.current_joint_vel[env_ids] = joint_vel
    self.current_data_state_id[env_ids] = data_state_ids
    self.current_reset_push[env_ids] = 0.0
    self.replay_reset_push[env_ids] = replay_samples["reset_push"]

  def record_reset_push(
    self, env_ids: torch.Tensor, velocity_delta: torch.Tensor
  ) -> None:
    self.current_reset_push[env_ids] = velocity_delta

  def metrics(self) -> dict[str, torch.Tensor]:
    return {
      "Metrics/fall_adaptive/failures_captured": self.last_failure_count,
      "Metrics/fall_adaptive/data_buffer_size": torch.tensor(
        float(self.data_failures.size), device=self.device
      ),
      "Metrics/fall_adaptive/random_buffer_size": torch.tensor(
        float(self.random_failures.size), device=self.device
      ),
      "Metrics/fall_adaptive/data_replay_rate": self.last_data_replay_rate,
      "Metrics/fall_adaptive/random_replay_rate": self.last_random_replay_rate,
    }
