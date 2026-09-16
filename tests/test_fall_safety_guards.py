"""Tests for fall-task reward and finite-physics safety guards."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mjlab.managers.reward_manager import RewardManager
from mjlab.tasks.fall.mdp.rewards import (
  ForbiddenContactForcePenalty,
  LowerBodyThenUpperBodyContactReward,
  ReduceContactForceWeighted,
  control_descent_speed,
  termination_event,
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

  # The categorical early-contact cost is a one-shot event (-2 / dt), while
  # the bounded force cost remains a rate (-0.002 * 1000).
  torch.testing.assert_close(reward, torch.tensor([-102.0]))
  torch.testing.assert_close(term(env), torch.tensor([-2.0]))


def test_lower_contact_bonus_requires_meaningful_force_and_is_one_shot():
  sensor = SimpleNamespace(
    _slots=[
      SimpleNamespace(primary_name="LINK_BASE"),
      SimpleNamespace(primary_name="LINK_ELBOW_END_L"),
    ],
    data=SimpleNamespace(
      found=torch.tensor([[1, 0]]),
      force=torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    ),
    compute_first_contact=lambda dt: torch.tensor([[False, False]]),
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
    lower_first_bonus=0.5,
    min_lower_contact_force=20.0,
  )

  torch.testing.assert_close(term(env), torch.zeros(1))
  sensor.data.force[:, 0, 0] = 20.0
  env.sim.data.time[:] = 0.02
  torch.testing.assert_close(term(env) * 0.1 * env.step_dt, torch.tensor([0.05]))
  torch.testing.assert_close(term(env), torch.zeros(1))


def test_contact_force_penalty_is_linear_until_bounded():
  sensor = SimpleNamespace(
    _slots=[SimpleNamespace(primary_name="LINK_ELBOW_END_L")],
    data=SimpleNamespace(
      found=torch.ones(3, 1, dtype=torch.int64),
      force=torch.tensor(
        [
          [[100.0, 0.0, 0.0]],
          [[200.0, 0.0, 0.0]],
          [[600.0, 0.0, 0.0]],
        ]
      ),
    ),
  )
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    scene={"body_contact_force": sensor},
  )
  term = ReduceContactForceWeighted(
    sensor_name="body_contact_force",
    sum_weight=0.25,
    squash_scale=0.0,
    max_penalty=2.0,
    body_force_scales={"LINK_ELBOW_END_L": 200.0},
  )

  reward = term(env)

  torch.testing.assert_close(reward, torch.tensor([-0.625, -1.25, -2.0]))


def test_contact_force_penalty_cannot_be_diluted_by_tiny_contacts():
  sensor = SimpleNamespace(
    _slots=[
      SimpleNamespace(primary_name="LINK_ELBOW_END_L"),
      SimpleNamespace(primary_name="LINK_BASE"),
    ],
    data=SimpleNamespace(
      found=torch.tensor([[1, 0], [1, 1]]),
      force=torch.tensor(
        [
          [[200.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[200.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
      ),
    ),
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene={"body_contact_force": sensor},
  )
  term = ReduceContactForceWeighted(
    sensor_name="body_contact_force",
    sum_weight=0.25,
    squash_scale=0.0,
    body_force_scales={"LINK_ELBOW_END_L": 200.0},
    default_force_scale=1000.0,
  )

  reward = term(env)

  assert reward[1] <= reward[0]


def test_forbidden_contact_barrier_is_zero_without_force():
  sensor = SimpleNamespace(
    _slots=[SimpleNamespace(primary_name="LINK_ELBOW_END_L")],
    data=SimpleNamespace(
      found=torch.zeros(1, 1, dtype=torch.int64),
      force=torch.zeros(1, 1, 3),
    ),
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={"body_contact_force": sensor},
  )
  term = ForbiddenContactForcePenalty(
    sensor_name="body_contact_force",
    body_force_thresholds={"LINK_ELBOW_END_L": 200.0},
    start_ratio=0.75,
  )

  torch.testing.assert_close(term(env), torch.zeros(1), atol=1e-8, rtol=0.0)


def test_termination_event_survives_reward_manager_dt_scaling():
  termination_manager = SimpleNamespace(
    get_term=lambda name: torch.tensor([False, True])
  )
  env = SimpleNamespace(step_dt=0.02, termination_manager=termination_manager)

  reward_rate = termination_event(env, "forbidden_body_contact_force")
  integrated_reward = reward_rate * -2.0 * env.step_dt

  torch.testing.assert_close(integrated_reward, torch.tensor([0.0, -2.0]))


def test_episode_reward_rate_uses_actual_episode_duration():
  manager = object.__new__(RewardManager)
  manager._env = SimpleNamespace(
    episode_length_buf=torch.tensor([50, 100]),
    step_dt=0.02,
  )
  manager._episode_sums = {"cost": torch.tensor([2.0, 4.0])}
  manager._class_term_cfgs = []

  extras = manager.reset(torch.tensor([0, 1]))

  # Both episodes accumulated 2 reward units per actual second.
  torch.testing.assert_close(extras["Episode_Reward/cost"], torch.tensor(2.0))


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
