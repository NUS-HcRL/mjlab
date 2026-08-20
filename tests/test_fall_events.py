"""Tests for fall-task reset metadata and directional force pulses."""

import csv
from unittest.mock import Mock

import torch

from mjlab.tasks.fall.mdp import events


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
  setattr(
    env,
    "_fall_last_reset_data_mask",
    torch.tensor([False, True, True, False]),
  )
  setattr(
    env,
    "_fall_last_reset_data_direction_w",
    torch.tensor(
      [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
      ]
    ),
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
    getattr(env, "_fall_force_pulse_steps_left"),
    torch.tensor([3, 2, 2, 3]),
  )
