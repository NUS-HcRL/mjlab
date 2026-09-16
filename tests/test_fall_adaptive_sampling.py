"""Tests for fall-task failure replay sampling."""

from types import SimpleNamespace

import torch

from mjlab.tasks.fall.mdp.adaptive_sampling import FallAdaptiveResetSampler


class _Scene(dict):
  def __init__(self, num_envs: int, num_joints: int) -> None:
    super().__init__(robot=SimpleNamespace(num_joints=num_joints))
    self.env_origins = torch.zeros(num_envs, 3)


def _fake_env(num_envs: int = 2, num_joints: int = 3):
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=_Scene(num_envs, num_joints),
  )


def _empty_replay_samples(num_envs: int, num_joints: int):
  return {
    "use_replay": torch.zeros(num_envs, dtype=torch.bool),
    "root_state_rel": torch.zeros(num_envs, 13),
    "joint_pos": torch.zeros(num_envs, num_joints),
    "joint_vel": torch.zeros(num_envs, num_joints),
    "data_state_id": torch.full((num_envs,), -1, dtype=torch.long),
    "reset_push": torch.zeros(num_envs, 6),
    "reset_wrench": torch.zeros(num_envs, 6),
    "reset_pulse_duration": torch.zeros(num_envs, 1, dtype=torch.long),
    "replay_index": torch.full((num_envs,), -1, dtype=torch.long),
    "replay_generation": torch.full((num_envs,), -1, dtype=torch.long),
  }


def test_data_and_random_failures_are_replayed_independently():
  env = _fake_env()
  sampler = FallAdaptiveResetSampler(
    env,
    capacity=8,
    replay_probability=1.0,
    min_failures=1,
    neighbor_scale=0.1,
  )
  env_ids = torch.tensor([0, 1])
  data_mask = torch.tensor([True, False])
  root_state = torch.zeros(2, 13)
  root_state[:, 3] = 1.0
  joint_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  joint_vel = -joint_pos
  data_state_ids = torch.tensor([17, -1])

  sampler.begin_episodes(
    env,
    env_ids,
    data_mask,
    _empty_replay_samples(2, 3),
    root_state,
    joint_pos,
    joint_vel,
    data_state_ids,
  )
  sampler.record_reset_push(
    env_ids,
    torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
  )
  sampler.record_reset_force_pulse(
    torch.tensor([1]),
    torch.tensor([[30.0, -20.0, -10.0, 1.0, 2.0, 3.0]]),
    torch.tensor([7]),
  )
  sampler.capture_failures(env_ids, torch.tensor([True, True]))

  replay = sampler.sample_for_sources(env_ids, data_mask)

  assert replay["use_replay"].tolist() == [True, True]
  assert replay["data_state_id"].tolist() == [17, -1]
  assert torch.equal(replay["root_state_rel"][1], root_state[1])
  assert torch.equal(replay["joint_pos"][1], joint_pos[1])
  assert torch.equal(
    replay["reset_push"][1], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  )
  assert torch.equal(replay["reset_push"][0], torch.zeros(6))
  assert torch.equal(
    replay["reset_wrench"][1],
    torch.tensor([30.0, -20.0, -10.0, 1.0, 2.0, 3.0]),
  )
  assert replay["reset_pulse_duration"][1, 0].item() == 7
  assert torch.equal(replay["reset_wrench"][0], torch.zeros(6))
  assert sampler.data_failures.size == 1
  assert sampler.random_failures.size == 1


def test_non_failure_does_not_enter_replay_buffer():
  env = _fake_env(num_envs=1)
  sampler = FallAdaptiveResetSampler(
    env,
    capacity=4,
    replay_probability=1.0,
    min_failures=1,
    neighbor_scale=0.1,
  )
  env_ids = torch.tensor([0])
  sampler.begin_episodes(
    env,
    env_ids,
    torch.tensor([False]),
    _empty_replay_samples(1, 3),
    torch.zeros(1, 13),
    torch.zeros(1, 3),
    torch.zeros(1, 3),
    torch.tensor([-1]),
  )

  sampler.capture_failures(env_ids, torch.tensor([False]))

  replay = sampler.sample_for_sources(env_ids, torch.tensor([False]))
  assert replay["use_replay"].tolist() == [False]
  assert sampler.random_failures.size == 0


def test_replay_outcomes_update_priority_without_duplicate_insertion():
  env = _fake_env(num_envs=1)
  sampler = FallAdaptiveResetSampler(
    env,
    capacity=4,
    replay_probability=1.0,
    min_failures=1,
    neighbor_scale=0.1,
    uniform_ratio=0.2,
    priority_alpha=0.5,
  )
  env_ids = torch.tensor([0])
  data_mask = torch.tensor([False])
  root_state = torch.zeros(1, 13)
  joint_pos = torch.zeros(1, 3)
  joint_vel = torch.zeros(1, 3)

  sampler.begin_episodes(
    env,
    env_ids,
    data_mask,
    _empty_replay_samples(1, 3),
    root_state,
    joint_pos,
    joint_vel,
    torch.tensor([-1]),
  )
  sampler.capture_failures(env_ids, torch.tensor([True]))
  assert sampler.random_failures.size == 1
  assert sampler.random_failures.priorities[0].item() == 1.0

  replay = sampler.sample_for_sources(env_ids, data_mask)
  sampler.begin_episodes(
    env,
    env_ids,
    data_mask,
    replay,
    root_state,
    joint_pos,
    joint_vel,
    torch.tensor([-1]),
  )
  sampler.capture_failures(env_ids, torch.tensor([False]))

  assert sampler.random_failures.size == 1
  assert sampler.random_failures.priorities[0].item() == 0.5
  assert sampler.last_random_replay_failure_rate.item() == 0.0

  replay = sampler.sample_for_sources(env_ids, data_mask)
  sampler.begin_episodes(
    env,
    env_ids,
    data_mask,
    replay,
    root_state,
    joint_pos,
    joint_vel,
    torch.tensor([-1]),
  )
  sampler.capture_failures(env_ids, torch.tensor([True]))

  assert sampler.random_failures.size == 1
  assert sampler.random_failures.priorities[0].item() == 0.75
  assert sampler.last_random_replay_failure_rate.item() == 1.0


def test_priority_sampling_retains_uniform_floor():
  env = _fake_env(num_envs=1)
  sampler = FallAdaptiveResetSampler(
    env,
    capacity=4,
    replay_probability=1.0,
    min_failures=1,
    neighbor_scale=0.1,
    uniform_ratio=0.2,
    priority_alpha=0.5,
  )
  buffer = sampler.data_failures
  buffer.add(torch.tensor([10, 20]))
  buffer.priorities[:2] = torch.tensor([1.0, 0.0])

  probabilities = buffer._sampling_probabilities(sampler.uniform_ratio)

  assert torch.allclose(probabilities, torch.tensor([0.9, 0.1]))
  metrics = sampler.metrics()
  assert "Metrics/fall_adaptive/data_sampling_entropy" in metrics
  assert "Metrics/fall_adaptive/data_top1_probability" in metrics


def test_non_target_termination_does_not_count_as_replay_success():
  env = _fake_env(num_envs=1)
  sampler = FallAdaptiveResetSampler(
    env,
    capacity=4,
    replay_probability=1.0,
    min_failures=1,
    neighbor_scale=0.1,
    priority_alpha=0.5,
  )
  env_ids = torch.tensor([0])
  data_mask = torch.tensor([False])
  root_state = torch.zeros(1, 13)
  joint_pos = torch.zeros(1, 3)
  joint_vel = torch.zeros(1, 3)
  data_state_ids = torch.tensor([-1])

  sampler.begin_episodes(
    env,
    env_ids,
    data_mask,
    _empty_replay_samples(1, 3),
    root_state,
    joint_pos,
    joint_vel,
    data_state_ids,
  )
  sampler.capture_failures(env_ids, torch.tensor([True]))
  replay = sampler.sample_for_sources(env_ids, data_mask)
  sampler.begin_episodes(
    env,
    env_ids,
    data_mask,
    replay,
    root_state,
    joint_pos,
    joint_vel,
    data_state_ids,
  )

  sampler.capture_failures(
    env_ids,
    failure_mask=torch.tensor([False]),
    outcome_mask=torch.tensor([False]),
  )

  assert sampler.random_failures.priorities[0].item() == 1.0
  assert sampler.last_outcome_count.item() == 0.0
