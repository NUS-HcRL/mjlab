"""Tests for fall-task reset metadata and directional force pulses."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mjlab.tasks.fall.mdp import events


def test_stable_state_loader_accepts_legacy_qpos_and_filters_rows(tmp_path):
  path = tmp_path / "stable.npy"
  np.save(
    path,
    np.array(
      [
        [0.0, 3.0, 4.0, 0.8, 2.0, 0.0, 0.0, 0.0, 0.25],
        [0.1, 3.0, 4.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.50],
        [0.2, 3.0, 4.0, 0.8, np.nan, 0.0, 0.0, 0.0, 0.75],
      ],
      dtype=np.float64,
    ),
  )

  state = events._load_stable_state_file(
    str(path),
    expected_num_joints=1,
    device="cpu",
    min_root_height=0.6,
  )

  assert state["root_state"].shape == (1, 13)
  assert torch.allclose(state["root_state"][0, :3], torch.tensor([3.0, 4.0, 0.8]))
  assert torch.allclose(state["root_state"][0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0]))
  assert torch.equal(state["root_state"][0, 7:13], torch.zeros(6))
  assert state["joint_pos"][0, 0].item() == 0.25
  assert state["joint_vel"][0, 0].item() == 0.0


def test_stable_state_loader_restores_complete_state_and_filters_limits(tmp_path):
  path = tmp_path / "complete.npy"
  # time + root state (13) + one joint position + one joint velocity.
  np.save(
    path,
    np.array(
      [
        [
          0.0,
          3.0,
          4.0,
          0.8,
          2.0,
          0.0,
          0.0,
          0.0,
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
          0.25,
          -0.75,
        ],
        [
          0.1,
          3.0,
          4.0,
          0.8,
          1.0,
          0.0,
          0.0,
          0.0,
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
          1.25,
          -0.75,
        ],
      ],
      dtype=np.float64,
    ),
  )

  state = events._load_stable_state_file(
    str(path),
    expected_num_joints=1,
    device="cpu",
    min_root_height=0.6,
    max_root_linear_speed=4.0,
    max_root_angular_speed=10.0,
    max_abs_joint_velocity=1.0,
    joint_pos_limits=torch.tensor([[-1.0, 1.0]]),
  )

  assert state["root_state"].shape == (1, 13)
  assert torch.allclose(
    state["root_state"][0],
    torch.tensor([3.0, 4.0, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
  )
  assert torch.allclose(state["joint_pos"], torch.tensor([[0.25]]))
  assert torch.allclose(state["joint_vel"], torch.tensor([[-0.75]]))


@pytest.mark.parametrize("standing_probability", [0.0, 1.0])
def test_stable_state_sampling_discards_recorded_xy_translation(
  tmp_path, standing_probability
):
  path = tmp_path / "stable.npy"
  # time + root state (13) + two joint positions + two joint velocities.
  np.save(
    path,
    np.array(
      [
        [
          0.0,
          8.0,
          -6.0,
          0.8,
          1.0,
          0.0,
          0.0,
          0.0,
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
          0.2,
          -0.3,
          0.4,
          -0.5,
        ]
      ]
    ),
  )

  class _Scene(dict):
    pass

  env = SimpleNamespace(device="cpu")
  env.scene = _Scene()
  env.scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
  asset = Mock()
  asset.num_joints = 2
  asset.data.default_root_state = torch.tensor(
    [[0.0, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
  )
  asset.data.default_joint_pos = torch.zeros(1, 2)
  asset.data.default_joint_vel = torch.zeros(1, 2)
  asset.data.default_joint_pos_limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
  asset.data.joint_pos_limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
  asset.data.soft_joint_pos_limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])

  root_state, joint_pos, joint_vel = events._sample_stable_states(
    env=env,
    asset=asset,
    env_ids=torch.tensor([0]),
    stable_state_files=(str(path),),
    # Standing must bypass even the configured nonzero pose jitter.
    stable_pose_range={"z": (0.02, 0.02), "yaw": (0.1, 0.1)}
    if standing_probability else {},
    stable_joint_position_std=0.0,
    stable_min_root_height=0.6,
    stable_standing_probability=standing_probability,
  )

  if standing_probability:
    assert torch.allclose(root_state[0, :3], torch.tensor([10.0, 20.0, 0.82]))
    assert torch.equal(root_state[:, 3:7], asset.data.default_root_state[:, 3:7])
    assert torch.equal(root_state[:, 7:13], torch.zeros(1, 6))
    assert torch.equal(joint_pos, asset.data.default_joint_pos)
    assert torch.equal(joint_vel, torch.zeros(1, 2))
    return

  assert torch.allclose(root_state[0, :3], torch.tensor([10.0, 20.0, 0.8]))
  assert torch.allclose(root_state[0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0]))
  assert torch.allclose(
    root_state[0, 7:13], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  )
  assert torch.allclose(joint_pos[0], torch.tensor([0.2, -0.3]))
  assert torch.allclose(joint_vel[0], torch.tensor([0.4, -0.5]))


def test_world_pose_offset_rotates_recorded_world_velocities():
  root_state = torch.tensor(
    [[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
  )
  pose_offset = torch.tensor([[0.1, -0.2, 0.0, 0.0, 0.0, np.pi / 2]])

  events._apply_world_pose_offset(root_state, pose_offset)

  assert torch.allclose(root_state[0, :3], torch.tensor([0.1, -0.2, 0.8]))
  assert torch.allclose(root_state[0, 7:10], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
  assert torch.allclose(root_state[0, 10:13], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


def test_centered_range_jitter_does_not_bias_fixed_axes():
  ranges = torch.tensor(
    [
      [-10.0, 10.0],
      [-20.0, 40.0],
      [-80.0, -80.0],
    ]
  )

  jitter = events._sample_centered_range_jitter(ranges, size=(128, 3), device="cpu")

  assert torch.all(jitter[:, 0].abs() <= 10.0)
  assert torch.all(jitter[:, 1].abs() <= 30.0)
  assert torch.equal(jitter[:, 2], torch.zeros(128))


def test_motion_reset_csv_filters_low_and_fast_rows(tmp_path):
  path = tmp_path / "reset.csv"
  fieldnames = (
    "joint_pos_0",
    "joint_vel_0",
    "body_pos_w_1_x",
    "body_pos_w_1_y",
    "body_pos_w_1_z",
    "body_quat_w_1_w",
    "body_quat_w_1_x",
    "body_quat_w_1_y",
    "body_quat_w_1_z",
  )
  rows = (
    ("0", "1", "0", "0", "0.50", "1", "0", "0", "0"),
    ("0", "1", "0", "0", "0.20", "1", "0", "0", "0"),
    ("0", "41", "0", "0", "0.50", "1", "0", "0", "0"),
  )
  with path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(fieldnames)
    writer.writerows(rows)

  dataset = events._load_motion_reset_csv(
    str(path),
    root_body_idx=0,
    device="cpu",
    expected_num_joints=1,
    min_root_height=0.25,
    max_abs_joint_velocity=40.0,
  )

  assert dataset["root_state"].shape[0] == 1
  assert dataset["root_state"][0, 2].item() == 0.5
  assert dataset["joint_vel"][0, 0].item() == 1.0


def test_low_clearance_pose_tilt_is_suppressed():
  pose_samples = torch.ones((2, 6))

  events._zero_low_clearance_pose_tilt(
    pose_samples,
    root_height=torch.tensor([0.30, 0.50]),
    low_clearance_height=0.35,
  )

  assert torch.equal(pose_samples[0, 3:5], torch.zeros(2))
  assert torch.equal(pose_samples[1, 3:5], torch.ones(2))
  assert pose_samples[:, 2].tolist() == [1.0, 1.0]


def test_load_direction_vectors_w_maps_csv_directions():
  rows = [
    {"direction": "forward"},
    {"direction": "left"},
    {"direction": "backward_right"},
  ]

  vectors = events._load_direction_vectors_w(rows, ("direction",), "cpu")

  expected = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [-(2.0**-0.5), -(2.0**-0.5), 0.0],
    ]
  )
  assert torch.allclose(vectors, expected)


def test_directional_data_pulse_is_separate_from_random_pulse(monkeypatch):
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.episode_length_buf = torch.zeros(4, dtype=torch.long)
  env._fall_last_reset_data_mask = torch.tensor([False, True, True, False])
  env._fall_last_reset_data_direction_w = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )

  axiswise_calls = []
  directional_calls = []
  monkeypatch.setattr(
    events,
    "apply_external_force_torque_axiswise",
    lambda **kwargs: axiswise_calls.append(kwargs),
  )
  monkeypatch.setattr(
    events,
    "apply_external_force_directional",
    lambda **kwargs: directional_calls.append(kwargs),
  )

  events.apply_external_force_torque_axiswise_pulse(
    env,
    duration_steps=3,
    pulse_probability=1.0,
    data_direction_force_magnitude_range=(30.0, 80.0),
    data_direction_force_probability=1.0,
    data_direction_duration_steps_range=(2, 2),
    preserve_data_reset_states=True,
  )

  assert len(axiswise_calls) == 1
  assert axiswise_calls[0]["env_ids"].tolist() == [0, 3]
  assert len(directional_calls) == 1
  assert directional_calls[0]["env_ids"].tolist() == [1, 2]
  assert torch.equal(
    directional_calls[0]["direction_w"],
    torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
  )
  assert torch.equal(
    env._fall_force_pulse_steps_left,
    torch.tensor([3, 2, 2, 3]),
  )


def test_force_pulse_consumes_explicit_reset_pending_marker(monkeypatch):
  env = Mock()
  env.num_envs = 2
  env.device = "cpu"
  # An explicit env.reset() is followed by a step increment before interval
  # events run, so episode length alone cannot identify this reset.
  env.episode_length_buf = torch.ones(2, dtype=torch.long)
  env._fall_force_pulse_pending = torch.tensor([True, False])
  env._fall_last_reset_data_mask = torch.tensor([False, False])

  axiswise_calls = []
  monkeypatch.setattr(
    events,
    "apply_external_force_torque_axiswise",
    lambda **kwargs: axiswise_calls.append(kwargs),
  )

  events.apply_external_force_torque_axiswise_pulse(
    env,
    duration_steps=2,
    pulse_probability=1.0,
  )

  assert len(axiswise_calls) == 1
  assert axiswise_calls[0]["env_ids"].tolist() == [0]
  assert env._fall_force_pulse_pending.tolist() == [False, False]
  assert env._fall_force_pulse_steps_left.tolist() == [2, 0]


def test_adaptive_force_replay_bypasses_previous_episode_cooldown(monkeypatch):
  env = Mock()
  env.num_envs = 1
  env.device = "cpu"
  env.episode_length_buf = torch.zeros(1, dtype=torch.long)
  env.scene = {"robot": SimpleNamespace(num_joints=2)}
  env._fall_last_reset_data_mask = torch.tensor([False])
  env._fall_force_pulse_steps_left = torch.zeros(1, dtype=torch.long)
  env._fall_force_pulse_cooldown_steps_left = torch.tensor([100])

  sampler = events.FallAdaptiveResetSampler(
    env,
    capacity=4,
    replay_probability=1.0,
    min_failures=1,
    neighbor_scale=0.1,
  )
  sampler.current_replayed[0] = True
  sampler.replay_reset_wrench[0] = torch.tensor([30.0, -20.0, -10.0, 1.0, 2.0, 3.0])
  sampler.replay_reset_pulse_duration[0, 0] = 4
  env._fall_adaptive_reset_sampler = sampler

  replay_calls = []
  monkeypatch.setattr(
    events,
    "_apply_external_wrench_vectors",
    lambda **kwargs: replay_calls.append(kwargs),
  )

  events.apply_external_force_torque_axiswise_pulse(
    env,
    duration_steps=2,
    force_axis_range={},
    torque_axis_range={},
    cooldown_steps=200,
  )

  assert len(replay_calls) == 1
  assert replay_calls[0]["env_ids"].tolist() == [0]
  assert torch.equal(replay_calls[0]["force_w"], torch.tensor([[30.0, -20.0, -10.0]]))
  assert torch.equal(replay_calls[0]["torque_w"], torch.tensor([[1.0, 2.0, 3.0]]))
  assert env._fall_force_pulse_steps_left[0].item() == 4
