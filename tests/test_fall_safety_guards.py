"""Tests for fall-task reward and finite-physics safety guards."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mjlab.tasks.fall.mdp.curriculums import (
  body_contact_force_threshold_curriculum,
)
from mjlab.tasks.fall.mdp.rewards import (
  LowerBodyThenUpperBodyContactReward,
  control_descent_speed,
)
from mjlab.tasks.fall.mdp.terminations import invalid_physics_state


def test_control_descent_speed_is_bounded():
  asset = Mock()
  asset.find_bodies.return_value = ([0], ["LINK_TORSO_YAW"])
  asset.data.body_link_lin_vel_w = torch.tensor(
    [[[0.0, 0.0, -0.4]], [[0.0, 0.0, -2.5]], [[0.0, 0.0, -100.0]]]
  )
  env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset})

  reward = control_descent_speed(
    env,
    threshold=0.5,
    max_downward_speed=5.0,
  )

  torch.testing.assert_close(reward, torch.tensor([0.0, -4.0, -25.0]))


def test_early_upper_contact_force_is_bounded():
  slots = [
    SimpleNamespace(primary_name="LINK_BASE"),
    SimpleNamespace(primary_name="LINK_ELBOW_END_L"),
  ]
  sensor = SimpleNamespace(
    _slots=slots,
    data=SimpleNamespace(
      found=torch.tensor([[0, 1]]),
      force=torch.tensor([[[0.0, 0.0, 0.0], [1_000_000.0, 0.0, 0.0]]]),
    ),
    compute_first_contact=lambda dt: torch.tensor([[False, True]]),
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    scene={"body_contact_force": sensor},
    sim=SimpleNamespace(data=SimpleNamespace(time=torch.tensor([0.0]))),
  )
  term = LowerBodyThenUpperBodyContactReward(
    sensor_name="body_contact_force",
    lower_body_names=("LINK_BASE",),
    upper_body_names=("LINK_ELBOW_END_L",),
    early_upper_penalty=2.0,
    early_upper_force_scale=0.002,
    max_upper_force=1000.0,
  )

  reward = term(env)

  torch.testing.assert_close(reward, torch.tensor([-4.0]))


def test_invalid_physics_state_detects_each_finite_limit():
  body_vel = torch.zeros(4, 2, 3)
  joint_vel = torch.zeros(4, 3)
  contact_force = torch.zeros(4, 2, 3)
  body_vel[1, 0, 0] = 20.01
  joint_vel[2, 1] = -100.01
  contact_force[3, 1, 2] = 20_001.0

  asset = SimpleNamespace(
    data=SimpleNamespace(
      body_link_lin_vel_w=body_vel,
      joint_vel=joint_vel,
    )
  )
  sensor = SimpleNamespace(data=SimpleNamespace(force=contact_force))
  env = SimpleNamespace(
    scene={"robot": asset, "body_contact_force": sensor},
  )

  invalid = invalid_physics_state(env, sensor_name="body_contact_force")

  assert invalid.tolist() == [False, True, True, True]


def test_body_contact_force_threshold_curriculum_only_updates_staged_bodies():
  thresholds = {
    "LINK_HEAD_YAW": 200.0,
    "LINK_TORSO_YAW": 500.0,
    "LINK_ELBOW_END_L": 200.0,
    "LINK_ELBOW_END_R": 200.0,
  }
  term_cfg = SimpleNamespace(params={"body_force_thresholds": thresholds})
  env = SimpleNamespace(
    common_step_counter=8_000 * 32,
    termination_manager=SimpleNamespace(
      cfg={"forbidden_body_contact_force": term_cfg}
    ),
  )
  stages = [
    {
      "step": 0,
      "body_force_thresholds": {
        "LINK_TORSO_YAW": 1000.0,
        "LINK_ELBOW_END_L": 800.0,
        "LINK_ELBOW_END_R": 800.0,
      },
    },
    {
      "step": 8_000 * 32,
      "body_force_thresholds": {
        "LINK_TORSO_YAW": 800.0,
        "LINK_ELBOW_END_L": 600.0,
        "LINK_ELBOW_END_R": 600.0,
      },
    },
  ]

  body_contact_force_threshold_curriculum(
    env,
    env_ids=torch.tensor([0]),
    termination_term_name="forbidden_body_contact_force",
    threshold_stages=stages,
  )

  assert thresholds == {
    "LINK_HEAD_YAW": 200.0,
    "LINK_TORSO_YAW": 800.0,
    "LINK_ELBOW_END_L": 600.0,
    "LINK_ELBOW_END_R": 600.0,
  }
