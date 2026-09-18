"""Dodge geometry, episode lifecycle, and isolation from the original fall task."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from types import FunctionType, SimpleNamespace

import pytest
import torch

from mjlab.tasks.fall.config.pm1.dodge_env_cfg import (
  pm1_falling_amp_dodge_runner_cfg,
  pm1_flat_falling_dodge_env_cfg,
)
from mjlab.tasks.fall.config.pm1.env_cfgs import pm1_flat_falling_env_cfg
from mjlab.tasks.fall.config.pm1.rl_cfg import pm1_falling_amp_runner_cfg
from mjlab.tasks.fall.mdp.dodge import (
  DodgeFirstContactCost,
  DodgeLandingClearanceReward,
  DodgePredictedLandingRisk,
  DodgeRegionCommand,
  DodgeRegionCommandCfg,
  dodge_region_contact_cost,
  region_contact_mask,
  predict_landing,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def make_region(num_envs: int = 4, **kwargs):
  """CPU tensors stand in for simulation state; command/reward code is unchanged."""
  qpos = torch.zeros(num_envs, 7)
  qpos[:, 2] = 0.82
  qpos[:, 3] = 1
  qvel = torch.zeros(num_envs, 6)
  qvel[:, 0] = 1
  indexing = SimpleNamespace(
    free_joint_q_adr=torch.arange(7), free_joint_v_adr=torch.arange(6)
  )
  data = SimpleNamespace(
    is_fixed_base=False,
    data=SimpleNamespace(qpos=qpos, qvel=qvel),
    root_link_pos_w=qpos[:, :3],
    root_link_quat_w=qpos[:, 3:7],
    root_link_ang_vel_w=torch.tensor([[0.0, 0.5, 0.0]]).repeat(num_envs, 1),
    body_link_pos_w=torch.tensor([[[0.4, 0.0, 0.5]]]).repeat(num_envs, 1, 1),
    body_link_lin_vel_w=torch.tensor([[[1.0, 0.0, -0.5]]]).repeat(num_envs, 1, 1),
  )
  class FakeRobot(SimpleNamespace):
    body_names = ("test_body",)

    def find_bodies(self, names, preserve_order=False):
      del preserve_order
      ids = [self.body_names.index(name) for name in names if name in self.body_names]
      return ids, [self.body_names[i] for i in ids]

  robot = FakeRobot(data=data, indexing=indexing)

  class Scene(dict):
    env_origins = torch.zeros(num_envs, 3)

  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=Scene(robot=robot),
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    step_dt=0.02,
    termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
  )
  region = DodgeRegionCommand(
    DodgeRegionCommandCfg(landing_body_names=("test_body",), **kwargs), env
  )
  env.command_manager = SimpleNamespace(get_term=lambda name: region)
  return env, region


def test_contact_mask_uses_contact_points_and_masks_empty_slots():
  # The second point lies on the boundary; the first is outside. A zero-filled
  # empty slot at the origin must not be mistaken for a contact in the region.
  points = torch.tensor([[[0.4, 0, 0], [0.125, 0, 0], [0, 0, 0]]]).repeat(2, 1, 1)
  found = torch.tensor([[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
  mask = region_contact_mask(
    torch.tensor([True, False]),
    torch.zeros(2, 3),
    torch.full((2,), 0.125),
    found,
    points,
  )
  assert mask.tolist() == [[False, True, False], [False, False, False]]


def test_observation_rotates_into_body_frame_and_absent_region_is_zero():
  env, region = make_region(2)
  region.active[0] = True
  region.center_w[0] = torch.tensor([1, 0, 0])
  region.radius[0] = 0.1
  env.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
    [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
  )
  env.scene["robot"].data.root_link_pos_w[1] = torch.tensor([10, 20, 1])
  torch.testing.assert_close(
    region.command,
    torch.tensor([[1, 0, -1, -0.82, 0.1], [0, 0, 0, 0, 0]]),
    atol=1e-6,
    rtol=1e-6,
  )


def test_yaw_frame_does_not_rotate_ground_position_with_root_roll():
  env, region = make_region(1)
  region.active[:] = True
  region.center_w[0] = torch.tensor([0.0, 1.0, 0.82])
  region.radius[:] = 0.1
  env.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
    [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
  )
  torch.testing.assert_close(
    region.command, torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.1]])
  )
  region.cfg.reference_frame = "full"
  torch.testing.assert_close(
    region.command, torch.tensor([[1.0, 0.0, 0.0, -1.0, 0.1]]),
    atol=1e-6,
    rtol=1e-6,
  )


def test_projection_waits_for_pulse_then_targets_limb_not_base():
  env, region = make_region(
    1,
    probability=1,
    current_velocity_mix=1.0,
    placement_jitter=0.0,
    radius_range=(0.1, 0.1),
  )
  region.reset(None)
  env.episode_length_buf[:] = 1
  env._fall_force_pulse_steps_left = torch.tensor([5])
  for _ in range(8):
    region.compute(env.step_dt)
    assert not region.active.any()
  env._fall_force_pulse_steps_left[:] = 0
  for _ in range(3):
    region.compute(env.step_dt)
  expected_time = (-0.5 + math.sqrt(0.25 + 2 * 0.4 * 9.81)) / 9.81
  torch.testing.assert_close(
    region.center_w[0], torch.tensor([expected_time + 0.4, 0.0, 0.0]),
    atol=1e-6,
    rtol=1e-6,
  )


def test_reset_samples_world_fixed_regions_and_preserves_unselected_envs():
  torch.manual_seed(17)
  env, region = make_region(
    probability=1.0,
    placement_jitter=0.0,
    radius_range=(0.1, 0.1),
  )
  qpos = env.scene["robot"].data.data.qpos.clone()
  qvel = env.scene["robot"].data.data.qvel.clone()
  region.reset(None)
  assert not region.active.any()
  assert region.pending.all()
  # The region appears only after three completed post-reset control steps.
  for step in range(1, 4):
    env.scene["robot"].data.data.qpos[:, 0] = 0.05 * step
    env.episode_length_buf[:] = step
    region.compute(env.step_dt)
    assert region.active.all().item() == (step == 3)
  center = region.center_w.clone()
  radius = region.radius.clone()
  torch.testing.assert_close(
    env.scene["robot"].data.data.qpos[:, 1:], qpos[:, 1:]
  )
  torch.testing.assert_close(env.scene["robot"].data.data.qvel, qvel)
  # Even long play episodes and robot movement cannot resample/move the region.
  env.scene["robot"].data.root_link_pos_w[:, 0] += 5
  region.compute(2e9)
  torch.testing.assert_close(region.center_w, center)
  region.cfg.probability = 0
  region.reset(torch.tensor([1, 3]))
  assert region.active.tolist() == [True, False, True, False]
  torch.testing.assert_close(region.center_w[[0, 2]], center[[0, 2]])
  torch.testing.assert_close(region.radius[[0, 2]], radius[[0, 2]])
  assert torch.count_nonzero(region.command[[1, 3]]) == 0


def test_sampler_rejects_late_falls_and_initial_body_overlap():
  env, region = make_region(
    3,
    probability=1,
    placement_jitter=0,
    radius_range=(0.1, 0.1),
  )
  env.scene["robot"].data.data.qpos[0, 2] = 0.3  # Too late to dodge.
  env.scene["robot"].data.body_link_pos_w[1, 0] = torch.tensor([0.4, 0, 0])
  region.reset(None)
  assert region.requested.all()
  env.episode_length_buf[:] = 1
  for _ in range(3):
    region.compute(env.step_dt)
  assert region.active.tolist() == [False, False, True]
  assert region.center_w[2, 0] > 0.4


def test_sampling_respects_translated_environments_and_limb_velocity():
  env, region = make_region(
    1,
    probability=1,
    placement_jitter=0,
  )
  env.scene.env_origins[0] = torch.tensor([10, 20, 2])
  data = env.scene["robot"].data
  data.data.qpos[0, :3] += env.scene.env_origins[0]
  data.body_link_pos_w[0] += env.scene.env_origins[0]
  data.body_link_pos_w[0, 0, :2] = torch.tensor([10.0, 19.6])
  data.body_link_lin_vel_w[0, 0] = torch.tensor([0.0, -1.0, -0.5])
  data.root_link_ang_vel_w[:] = torch.tensor([0.5, 0.0, 0.0])
  data.data.qvel[0, :3] = torch.tensor([0, -1, 0])
  # Simulate derived velocity still reflecting the state before the reset push.
  data.root_link_lin_vel_w = torch.tensor([[1, 0, 0]])
  region.reset(None)
  env.episode_length_buf[:] = 1
  for _ in range(3):
    region.compute(env.step_dt)
  assert region.active.item()
  expected_time = (-0.5 + math.sqrt(0.25 + 2 * 0.4 * 9.81)) / 9.81
  torch.testing.assert_close(region.center_w[0], torch.tensor([10, 19.6 - expected_time, 2]))


def test_contact_cost_is_bounded_and_logs_before_partial_reset():
  env, region = make_region(3, probability=0)
  region.active[:2] = True
  region.requested[:2] = True
  region.radius[:] = 0.1
  env.scene["dodge_ground_contact"] = SimpleNamespace(
    cfg=SimpleNamespace(num_slots=2),
    data=SimpleNamespace(
      found=torch.tensor([[3, 3], [1, 0], [1, 0]]),
      pos=torch.tensor(
        [
          [[0, 0, 0], [0.05, 0, 0]],
          [[0.5, 0, 0], [0, 0, 0]],
          [[0, 0, 0], [0, 0, 0]],
        ]
      ),
    ),
  )
  # Two simultaneous contacts still incur one unit of cost, and inactive envs
  # incur none. Overflow counts use the original match count, not occupied slots.
  torch.testing.assert_close(
    dodge_region_contact_cost(env), torch.tensor([1.0, 0.0, 0.0])
  )
  metrics = region.reset(torch.tensor([0, 1]))
  assert metrics["contact_rate"] == 0.5
  assert metrics["contact_time_fraction"] == 0.5
  assert metrics["slot_overflow_rate"] == 0.5
  assert region._steps.tolist() == [0, 0, 1]
  assert region.reset(torch.tensor([0, 1])) == {}  # No synthetic empty episodes.


def test_first_contact_cost_is_fixed_once_per_episode():
  env, region = make_region(2, probability=0)
  region.active[:] = True
  region.radius[:] = 0.1
  env.scene["dodge_ground_contact"] = SimpleNamespace(
    cfg=SimpleNamespace(num_slots=1),
    data=SimpleNamespace(
      found=torch.ones(2, 1),
      pos=torch.tensor([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]),
    ),
  )
  cost = DodgeFirstContactCost()
  # RewardManager later multiplies this by step_dt, yielding one fixed unit.
  torch.testing.assert_close(cost(env), torch.tensor([50.0, 0.0]))
  torch.testing.assert_close(cost(env), torch.zeros(2))
  cost.reset(torch.tensor([0]))
  torch.testing.assert_close(cost(env), torch.tensor([50.0, 0.0]))


def test_probability_preserves_a_majority_of_unmodified_episodes():
  torch.manual_seed(123)
  _, region = make_region(4000)
  region.reset(None)
  assert 0.17 < region.requested.float().mean().item() < 0.23
  assert not region.active.any()
  assert torch.count_nonzero(region.command[~region.active]) == 0


def test_predicted_landing_risk_is_local_bounded_and_uses_selected_bodies():
  env, region = make_region(3, probability=0)
  region.active[:2] = True
  region.radius[:] = 0.1
  robot = env.scene["robot"]
  robot.body_names = ("torso", "knee", "head")
  data = env.scene["robot"].data
  data.body_link_pos_w = torch.tensor(
    [
      [[-0.30, 0.0, 0.30], [0.8, 0.0, 0.3], [0.0, 0.0, 0.1]],
      [[-0.60, 0.0, 0.30], [0.8, 0.0, 0.3], [0.0, 0.0, 0.1]],
      [[-0.30, 0.0, 0.30], [0.8, 0.0, 0.3], [0.0, 0.0, 0.1]],
    ]
  )
  data.body_link_lin_vel_w = torch.zeros(3, 3, 3)
  data.body_link_lin_vel_w[:, :, 2] = -1.0
  data.body_link_lin_vel_w[:, 0, 0] = 1.0
  risk = DodgePredictedLandingRisk(body_names=("torso", "knee"))(env)
  assert torch.all((0 <= risk) & (risk <= 1))
  assert risk[0] > risk[1] > risk[2]
  assert risk[2] == 0


def test_clearance_shaping_rewards_improvement_and_handles_terminal_and_reset():
  env, region = make_region(3, probability=0)
  region.active[:2] = True
  region.radius[:] = 0.1
  data = env.scene["robot"].data
  data.body_link_pos_w[:] = torch.tensor([0.0, 0.0, 0.1])
  data.body_link_lin_vel_w.zero_()
  reward = DodgeLandingClearanceReward(body_names=("test_body",))
  assert torch.equal(reward(env), torch.zeros(3))  # Activation is not an action.
  initial = reward._previous.clone()
  data.body_link_pos_w[:, :, 0] = 0.3
  improvement = reward(env) * env.step_dt
  assert improvement[0] > 0.5 and improvement[2] == 0
  data.body_link_pos_w[:, :, 0] = 0.0
  retreat = reward(env) * env.step_dt
  assert retreat[0] < -0.5
  # A discounted out-and-back path has exactly the potential telescoping sum,
  # not two positive "progress" bonuses.
  torch.testing.assert_close(
    improvement[:2] + reward.gamma * retreat[:2],
    (reward.gamma**2 - 1) * initial[:2],
  )
  previous = reward._previous.clone()
  env.termination_manager.terminated[0] = True
  terminal = reward(env) * env.step_dt
  torch.testing.assert_close(terminal[0], -previous[0])
  # Time limits are not true terminals: preserve potential for PPO bootstrap.
  torch.testing.assert_close(terminal[1], (reward.gamma - 1) * previous[1])
  reward.reset(torch.tensor([0]))
  env.termination_manager.terminated.zero_()
  assert reward(env)[0] == 0
  assert reward._initialized[1]


def test_grounded_projection_and_nonfinite_shaping_remain_safe():
  env, region = make_region(1, probability=0)
  region.active[:] = True
  data = env.scene["robot"].data
  data.body_link_pos_w[:] = torch.tensor([0.0, 0.0, 0.05])
  data.body_link_lin_vel_w[:] = torch.tensor([10.0, 0.0, 0.0])
  xy, time = predict_landing(
    data.body_link_pos_w, data.body_link_lin_vel_w, env.scene.env_origins[:, 2]
  )
  assert torch.count_nonzero(xy) == 0 and torch.count_nonzero(time) == 0
  reward = DodgeLandingClearanceReward(body_names=("test_body",))
  reward(env)
  data.body_link_pos_w[:] = float("nan")
  env.termination_manager.terminated[:] = True
  assert torch.isfinite(reward(env)).all()
  assert torch.isfinite(reward._previous).all()


def test_clearance_uses_activation_snapshot_for_first_action():
  env, region = make_region(1, probability=1, placement_jitter=0)
  region.reset(None)
  env.episode_length_buf[:] = 1
  for _ in range(3):
    region.compute(env.step_dt)
  assert region.activation_valid.item()
  # First action moves away after seeing the region. Its improvement must not
  # be discarded or used as a policy-controlled initial potential baseline.
  env.scene["robot"].data.body_link_pos_w[:, :, 1] += 0.4
  reward = DodgeLandingClearanceReward(body_names=("test_body",))
  assert (reward(env) * env.step_dt).item() > 0.5
  region.reset(None)
  reward.reset(None)
  assert not region.activation_valid.any()
  assert reward(env).item() == 0


def test_upright_translation_does_not_confirm_fall_and_late_limb_is_rejected():
  env, region = make_region(2, probability=1, placement_jitter=0)
  data = env.scene["robot"].data
  data.root_link_ang_vel_w[0] = 0  # Moving forward alone is not a fall.
  data.body_link_pos_w[1, 0, 2] = 0.16
  data.body_link_lin_vel_w[1, 0, 2] = -3  # Not enough reaction time.
  region.reset(None)
  env.episode_length_buf[:] = 1
  for _ in range(45):
    region.compute(env.step_dt)
  assert not region.active.any()
  assert not region.pending.any()


def snapshot(value):
  """Compare config state, including stateful reward instances, without identity."""
  if is_dataclass(value) and not isinstance(value, type):
    return (
      type(value),
      {f.name: snapshot(getattr(value, f.name)) for f in fields(value)},
    )
  if isinstance(value, dict):
    return {k: snapshot(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return (type(value), [snapshot(v) for v in value])
  if hasattr(value, "__dict__") and not isinstance(value, (FunctionType, type)):
    return (type(value), snapshot(vars(value)))
  return value


@pytest.mark.parametrize("play", [False, True])
def test_dodge_only_adds_expected_extensions_to_fall(play):
  base = pm1_flat_falling_env_cfg(play=play)
  dodge = pm1_flat_falling_dodge_env_cfg(play=play)
  assert base.amp is not None and isinstance(base.amp.motion_file, list)
  assert dodge.amp is not None and isinstance(dodge.amp.motion_file, list)
  assert dodge.amp.motion_file == [
    *base.amp.motion_file,
    "motion_file/pm_fall4:v0/tofront_dodgeleft_v2.3_50fps.npz",
    "motion_file/pm_fall4:v0/tofront_dodgeright_v2.3_50fps.npz",
  ]
  dodge.amp.motion_file = base.amp.motion_file
  assert dodge.commands["dodge"].probability == 0.5
  assert dodge.commands["dodge"].min_reaction_time == 0.12
  assert "forbidden_body_contact_force" not in dodge.terminations
  assert "invalid_physics_state" in dodge.terminations
  assert "nonfinite_state" in dodge.terminations
  dodge.terminations = base.terminations
  assert dodge.rewards["reduce_contact_force"].weight == 2.0
  assert dodge.rewards["action_rate_l2"].weight == -0.15
  for name in ("reduce_contact_force", "action_rate_l2"):
    dodge.rewards[name].weight = base.rewards[name].weight
  if "forbidden_contact_termination" in base.rewards:
    dodge.rewards["forbidden_contact_termination"] = base.rewards["forbidden_contact_termination"]
  for group in ("policy", "critic"):
    term = dodge.observations[group].terms.pop("dodge_region")
    assert term.history_length == 3 and term.noise is None
  dodge.commands = None
  cost = dodge.rewards.pop("dodge_region_contact")
  assert cost.weight == -1.0
  first_contact = dodge.rewards.pop("dodge_region_first_contact")
  assert first_contact.weight == -0.75
  assert isinstance(first_contact.func, DodgeFirstContactCost)
  landing_risk = dodge.rewards.pop("dodge_landing_clearance")
  assert "dodge_predicted_landing_risk" not in dodge.rewards
  assert landing_risk.weight == 1.0
  assert isinstance(landing_risk.func, DodgeLandingClearanceReward)
  assert landing_risk.func.body_names == (
    "LINK_TORSO_YAW",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_END_L",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_END_R",
    "LINK_KNEE_PITCH_L",
    "LINK_KNEE_PITCH_R",
  )
  assert landing_risk.func.gamma == pm1_falling_amp_dodge_runner_cfg().algorithm.gamma
  assert landing_risk.func.safety_margin == 0.05
  assert landing_risk.func.body_margin == 0.04
  assert landing_risk.func.temperature == 0.04
  sensor = dodge.scene.sensors[-1]
  assert sensor.primary.exclude == ()
  assert sensor.num_slots == 4
  assert sensor.fields == ("found", "pos")
  dodge.scene.sensors = dodge.scene.sensors[:-1]
  assert snapshot(dodge) == snapshot(base)


def test_registry_and_runner_preserve_original_amp_settings():
  task = "Mjlab-Falling-Flat-PM1-AMP-Dodge"
  assert "dodge_region" in load_env_cfg(task).observations["policy"].terms
  assert "dodge_region" in load_env_cfg(task, play=True).observations["policy"].terms
  original = load_env_cfg("Mjlab-Falling-Flat-PM1-AMP")
  assert "dodge_region" not in original.observations["policy"].terms
  runner = pm1_falling_amp_dodge_runner_cfg()
  assert snapshot(runner) == snapshot(load_rl_cfg(task))
  base = pm1_falling_amp_runner_cfg()
  assert runner.experiment_name != base.experiment_name
  assert runner.run_name == "dodge"
  runner.experiment_name = base.experiment_name
  runner.run_name = base.run_name
  assert runner.algorithm.kl_early_stop is False
  runner.algorithm.kl_early_stop = base.algorithm.kl_early_stop
  assert runner.algorithm.reward_mix_scale_clip == (
    0.02, base.algorithm.reward_mix_scale_clip[1]
  )
  runner.algorithm.reward_mix_scale_clip = base.algorithm.reward_mix_scale_clip
  assert snapshot(runner) == snapshot(base)


@pytest.mark.parametrize(
  "kwargs",
  [
    {"probability": 1.1},
    {"radius_range": (0, 0.1)},
    {"min_reaction_time": 0.7},
    {"placement_attempts": 0},
    {"placement_jitter": -0.1},
    {"max_observation_time": -0.1},
    {"motion_observation_steps": 0},
    {"reference_frame": "invalid"},
  ],
)
def test_invalid_region_configuration(kwargs):
  with pytest.raises(ValueError):
    DodgeRegionCommandCfg(**kwargs)
