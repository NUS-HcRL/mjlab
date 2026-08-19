"""Tests for fall-task reset metadata and directional force pulses."""

from unittest.mock import Mock

import torch

from mjlab.tasks.fall.mdp import events


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
