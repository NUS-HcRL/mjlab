"""Tests for categorical AMP fall-direction observations."""

import math
from types import SimpleNamespace

import torch

from mjlab.envs.amp import (
  FALL_DIRECTION_NAMES,
  AMPCfg,
  AMPHelper,
  _fall_direction_one_hot,
  _quantize_fall_direction,
  calc_disc_obs_dim,
  compute_disc_obs,
)
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_from_euler_xyz


def _fake_amp_helper(num_envs: int):
  quat = torch.zeros(num_envs, 1, 4)
  quat[..., 0] = 1.0
  data = SimpleNamespace(
    joint_pos=torch.zeros(num_envs, 3),
    joint_vel=torch.zeros(num_envs, 3),
    default_joint_pos=torch.zeros(num_envs, 3),
    body_link_pos_w=torch.zeros(num_envs, 1, 3),
    body_link_quat_w=quat,
    body_link_lin_vel_w=torch.zeros(num_envs, 1, 3),
    body_link_ang_vel_w=torch.zeros(num_envs, 1, 3),
  )
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={"robot": SimpleNamespace(body_names=["root"], data=data)},
  )
  helper = AMPHelper(
    env,
    AMPCfg(root_body_name="root", include_fall_direction_obs=True),
  )
  return helper, data


def test_quantize_fall_direction_maps_all_eight_classes():
  direction_xy = torch.tensor(
    [
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
      [-1.0, 1.0],
      [-1.0, 0.0],
      [-1.0, -1.0],
      [0.0, -1.0],
      [1.0, -1.0],
    ]
  )

  encoded = _quantize_fall_direction(direction_xy)

  assert FALL_DIRECTION_NAMES == (
    "forward",
    "forward_left",
    "left",
    "backward_left",
    "backward",
    "backward_right",
    "right",
    "forward_right",
  )
  assert torch.equal(encoded, torch.eye(8))


def test_expert_direction_names_generate_categorical_labels():
  encoded = _fall_direction_one_hot(("backward", "forward_left", "right"), "cpu")

  assert encoded.argmax(dim=-1).tolist() == [4, 1, 6]


def test_direction_is_appended_once_per_discriminator_sample():
  num_envs = 2
  num_steps = 2
  num_joints = 3
  root_pos = torch.zeros(num_envs, num_steps, 3)
  root_quat = torch.zeros(num_envs, num_steps, 4)
  root_quat[..., 0] = 1.0
  direction_obs = torch.eye(8)[:num_envs]

  disc_obs = compute_disc_obs(
    ref_root_pos=root_pos[:, -1],
    ref_root_quat=root_quat[:, -1],
    root_pos=root_pos,
    root_quat=root_quat,
    root_lin_vel=torch.zeros(num_envs, num_steps, 3),
    root_ang_vel=torch.zeros(num_envs, num_steps, 3),
    joint_pos=torch.zeros(num_envs, num_steps, num_joints),
    joint_vel=torch.zeros(num_envs, num_steps, num_joints),
    include_root_xy=False,
    include_root_rot=False,
    fall_direction_obs=direction_obs,
  )
  expected_dim = calc_disc_obs_dim(
    num_disc_obs_steps=num_steps,
    num_joints=num_joints,
    include_root_xy=False,
    include_root_rot=False,
    include_fall_direction_obs=True,
  )

  assert disc_obs.shape == (num_envs, expected_dim)
  assert torch.equal(disc_obs[:, -8:], direction_obs)


def test_policy_direction_preserves_heading_and_fallback_rules():
  helper, data = _fake_amp_helper(4)
  zero = torch.zeros(4)
  yaw = torch.full((4,), torch.pi / 2)
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, zero, yaw)
  helper.update()

  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(
    zero, torch.tensor([0.3, 0.0, 0.0, 0.0]), yaw
  )
  # With initial heading +Y: forward lean, left velocity (-X), backward
  # displacement (-Y), then the stationary/unknown forward fallback.
  data.body_link_lin_vel_w[1, 0, 0] = -1.0
  data.body_link_pos_w[2, 0, 1] = -1.0
  for _ in range(helper._cfg.fall_direction_confirm_steps):
    helper.update()

  assert helper.get_disc_obs()[:, -8:].argmax(-1).tolist() == [0, 2, 4, 0]
  assert helper._fall_direction_locked.tolist() == [True, True, True, False]


def test_provisional_direction_can_change_before_confirmation():
  helper, data = _fake_amp_helper(1)
  zero = torch.zeros(1)
  helper.update()

  for pitch, expected_direction in ((0.3, 0), (-0.3, 4)):
    data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([pitch]), zero)
    helper.update()
    assert helper.get_disc_obs()[:, -8:].argmax(-1).item() == expected_direction
    assert not helper._fall_direction_locked.item()


def test_confirmed_direction_survives_landing_roll_and_later_push():
  helper, data = _fake_amp_helper(1)
  zero = torch.zeros(1)
  helper.update()
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([0.3]), zero)

  steps = helper._cfg.fall_direction_confirm_steps
  for step in range(steps):
    helper.update()
    assert helper._fall_direction_locked.item() == (step == steps - 1)
  saved_label = helper.get_disc_obs()[:, -8:].clone()

  data.body_link_pos_w[0, 0, 2] = -0.5
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([-1.3]), zero)
  data.body_link_lin_vel_w[0, 0, 0] = -2.0
  for _ in range(2 * steps):
    helper.update()
    assert torch.equal(helper.get_disc_obs()[:, -8:], saved_label)


def test_weak_signal_and_default_label_never_lock():
  helper, data = _fake_amp_helper(1)
  zero = torch.zeros(1)
  helper.update()
  # Tiny reset tilt and slow motion must not permanently select forward.
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([0.02]), zero)
  data.body_link_lin_vel_w[0, 0, 0] = 0.01
  data.body_link_pos_w[0, 0, 0] = 0.001
  for _ in range(3 * helper._cfg.fall_direction_confirm_steps):
    helper.update()
  assert not helper._fall_direction_locked.item()
  assert helper._fall_direction_count.item() == 0

  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([-0.3]), zero)
  for _ in range(helper._cfg.fall_direction_confirm_steps):
    helper.update()
  assert helper._fall_direction_locked.item()
  assert helper.get_disc_obs()[:, -8:].argmax(-1).item() == 4


def test_boundary_oscillation_is_averaged_then_locked():
  helper, data = _fake_amp_helper(1)
  helper.update()
  # Instantaneous labels alternate 0/1, but the mean is just above 22.5 degrees.
  for degrees in (22.0, 23.0, 22.0, 23.0, 23.0):
    angle = math.radians(degrees)
    axis = torch.tensor([[-math.sin(angle), math.cos(angle), 0.0]])
    data.body_link_quat_w[:, 0] = quat_from_angle_axis(torch.tensor([0.3]), axis)
    helper.update()
  assert helper._fall_direction_locked.item()
  assert helper.get_disc_obs()[:, -8:].argmax(-1).item() == 1


def test_contradictory_window_restarts_confirmation():
  helper, data = _fake_amp_helper(1)
  zero = torch.zeros(1)
  helper.update()
  for pitch in (0.3, -0.3, 0.3, -0.3, 0.3):
    data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([pitch]), zero)
    helper.update()
  assert not helper._fall_direction_locked.item()
  assert helper._fall_direction_count.item() == 0
  assert torch.equal(helper._fall_direction_sum_xy, torch.zeros(1, 2))

  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, torch.tensor([-0.3]), zero)
  for _ in range(helper._cfg.fall_direction_confirm_steps):
    helper.update()
  assert helper._fall_direction_locked.item()
  assert helper.get_disc_obs()[:, -8:].argmax(-1).item() == 4


def test_uninformative_gap_restarts_consecutive_confirmation():
  helper, data = _fake_amp_helper(1)
  zero = torch.zeros(1)
  helper.update()
  lean = quat_from_euler_xyz(zero, torch.tensor([0.3]), zero)
  data.body_link_quat_w[:, 0] = lean
  steps = helper._cfg.fall_direction_confirm_steps
  for _ in range(steps - 1):
    helper.update()
  assert not helper._fall_direction_locked.item()

  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(zero, zero, zero)
  helper.update()
  assert helper._fall_direction_count.item() == 0
  data.body_link_quat_w[:, 0] = lean
  for step in range(steps):
    helper.update()
    assert helper._fall_direction_locked.item() == (step == steps - 1)


def test_partial_reset_only_refreshes_selected_heading_references():
  helper, data = _fake_amp_helper(2)
  helper.update()
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(
    torch.zeros(2), torch.full((2,), 0.3), torch.zeros(2)
  )
  for _ in range(helper._cfg.fall_direction_confirm_steps):
    helper.update()
  assert helper._fall_direction_locked.all()

  data.body_link_pos_w[:, 0, 0] = torch.tensor([0.5, 2.0])
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(
    torch.zeros(2), torch.zeros(2), torch.full((2,), torch.pi / 2)
  )
  helper.reset(torch.tensor([1]))
  helper.update()

  assert helper._episode_root_pos_w[:, 0].tolist() == [0.0, 2.0]
  assert torch.allclose(
    helper._episode_heading_cos, torch.tensor([1.0, 0.0]), atol=1e-6
  )
  assert torch.allclose(helper._episode_heading_sin, torch.tensor([0.0, 1.0]))
  assert not helper._direction_reference_dirty
  assert not helper._direction_reference_pending.any()
  assert helper._fall_direction_locked.tolist() == [True, False]
  assert helper._fall_direction_count[1].item() == 0
  assert torch.equal(helper._fall_direction_sum_xy[1], torch.zeros(2))

  # The reset environment can lock a new direction without relabeling env 0.
  data.body_link_quat_w[:, 0] = quat_from_euler_xyz(
    torch.tensor([0.0, -0.3]), torch.zeros(2), torch.full((2,), torch.pi / 2)
  )
  for _ in range(helper._cfg.fall_direction_confirm_steps):
    helper.update()
  assert helper._fall_direction_locked.all()
  assert helper.get_disc_obs()[:, -8:].argmax(-1).tolist() == [0, 2]

  helper.reset()
  assert not helper._fall_direction_locked.any()
  assert not helper._fall_direction_count.any()
  assert torch.equal(helper._fall_direction_sum_xy, torch.zeros(2, 2))
  assert helper._fall_direction_obs.argmax(-1).tolist() == [0, 0]
