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
