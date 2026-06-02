from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _mean_selected(value: torch.Tensor, env_ids: torch.Tensor | slice) -> torch.Tensor:
  selected = value[env_ids]
  if selected.numel() == 0:
    return torch.zeros((), device=value.device, dtype=value.dtype)
  return selected.float().mean()


def _body_log_name(body_name: str) -> str:
  return body_name.removeprefix("LINK_").lower()


def _get_sensor_body_names(sensor: ContactSensor) -> list[str]:
  """Extract unique body names from sensor slots, preserving order."""
  body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      body_names.append(slot.primary_name)
      seen.add(slot.primary_name)
  return body_names


def _build_body_weight_tensor(
  body_names: list[str],
  device: torch.device,
  dtype: torch.dtype,
  high_weight_bodies: tuple[str, ...] = (),
  shoulder_weight_bodies: tuple[str, ...] = (),
  medium_weight_bodies: tuple[str, ...] = (),
  high_weight: float = 10.0,
  shoulder_weight: float = 5.0,
  medium_weight: float = 1.0,
  low_weight: float = 0.1,
) -> torch.Tensor:
  """Build per-body vulnerability weights matching contact-force reward groups."""
  weights = torch.full((len(body_names),), low_weight, device=device, dtype=dtype)
  for i, body_name in enumerate(body_names):
    if body_name in high_weight_bodies:
      weights[i] = high_weight
    elif body_name in shoulder_weight_bodies:
      weights[i] = shoulder_weight
    elif body_name in medium_weight_bodies:
      weights[i] = medium_weight
  return weights


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def control_descent_speed(
  env: ManagerBasedRlEnv,
  torso_body_name: str = "LINK_TORSO_YAW",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  threshold: float = 0.5,
) -> torch.Tensor:
  """Penalize torso downward speed beyond a safe threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids, _ = asset.find_bodies((torso_body_name,), preserve_order=True)
  if not body_ids:
    return torch.zeros(env.num_envs, device=env.device)
  torso_lin_vel_z = asset.data.body_link_lin_vel_w[:, body_ids[0], 2]
  downward_speed = torch.clamp(-torso_lin_vel_z - threshold, min=0.0)
  return -(downward_speed**2)


class ImpactVelocityReward:
  """Penalize each body's first ground contact once per episode."""

  def __init__(
    self,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    high_weight_bodies: tuple[str, ...] = (),
    shoulder_weight_bodies: tuple[str, ...] = (),
    medium_weight_bodies: tuple[str, ...] = (),
    high_weight: float = 10.0,
    shoulder_weight: float = 5.0,
    medium_weight: float = 1.0,
    low_weight: float = 0.1,
    squash_scale: float = 0.02,
  ) -> None:
    self.sensor_name = sensor_name
    self.asset_cfg = asset_cfg
    self.high_weight_bodies = high_weight_bodies
    self.shoulder_weight_bodies = shoulder_weight_bodies
    self.medium_weight_bodies = medium_weight_bodies
    self.high_weight = high_weight
    self.shoulder_weight = shoulder_weight
    self.medium_weight = medium_weight
    self.low_weight = low_weight
    self.squash_scale = squash_scale
    self._body_ids: list[int] | None = None
    self._body_names: list[str] | None = None
    self._contacted_once: torch.Tensor | None = None

  def _maybe_initialize(self, env: ManagerBasedRlEnv) -> bool:
    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.found is not None
    if self._body_names is None:
      self._body_names = _get_sensor_body_names(sensor)
    if not self._body_names:
      return False
    if self._body_ids is None:
      asset: Entity = env.scene[self.asset_cfg.name]
      self._body_ids, _ = asset.find_bodies(
        tuple(self._body_names), preserve_order=True
      )
    if not self._body_ids:
      return False
    if self._contacted_once is None or self._contacted_once.shape != sensor.data.found.shape:
      self._contacted_once = torch.zeros_like(sensor.data.found, dtype=torch.bool)
    return True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._contacted_once is None:
      return
    if env_ids is None:
      self._contacted_once[:] = False
    else:
      self._contacted_once[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if not self._maybe_initialize(env):
      return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.found is not None
    assert self._body_names is not None
    assert self._body_ids is not None
    assert self._contacted_once is not None

    contact_now = sensor.data.found > 0
    first_contact_once = contact_now & ~self._contacted_once

    asset: Entity = env.scene[self.asset_cfg.name]
    body_lin_vel = asset.data.body_link_lin_vel_w[:, self._body_ids, :]
    downward_speed = torch.clamp(-body_lin_vel[..., 2], min=0.0)
    weights = _build_body_weight_tensor(
      body_names=self._body_names,
      device=body_lin_vel.device,
      dtype=body_lin_vel.dtype,
      high_weight_bodies=self.high_weight_bodies,
      shoulder_weight_bodies=self.shoulder_weight_bodies,
      medium_weight_bodies=self.medium_weight_bodies,
      high_weight=self.high_weight,
      shoulder_weight=self.shoulder_weight,
      medium_weight=self.medium_weight,
      low_weight=self.low_weight,
    )
    weighted_impact = (
      first_contact_once.float() * downward_speed.square() * weights.unsqueeze(0)
    )
    num_impacts = torch.clamp(first_contact_once.float().sum(dim=-1), min=1.0)
    penalty = weighted_impact.sum(dim=-1) / num_impacts
    if self.squash_scale > 0.0:
      # Keep small impacts near-linear while compressing outliers.
      penalty = torch.log1p(self.squash_scale * penalty) / self.squash_scale
    self._contacted_once |= contact_now
    return -penalty


class LowerBodyThenUpperBodyContactReward:
  """Encourage lower-body ground contact before delayed upper-body contact."""

  def __init__(
    self,
    sensor_name: str,
    lower_body_names: tuple[str, ...],
    upper_body_names: tuple[str, ...],
    min_delay_s: float = 0.12,
    max_delay_s: float = 0.45,
    lower_first_bonus: float = 0.5,
    timely_upper_bonus: float = 1.0,
    early_upper_penalty: float = 2.0,
    late_upper_penalty: float = 0.2,
    early_upper_force_scale: float = 0.0,
  ) -> None:
    self.sensor_name = sensor_name
    self.lower_body_names = lower_body_names
    self.upper_body_names = upper_body_names
    self.min_delay_s = min_delay_s
    self.max_delay_s = max_delay_s
    self.lower_first_bonus = lower_first_bonus
    self.timely_upper_bonus = timely_upper_bonus
    self.early_upper_penalty = early_upper_penalty
    self.late_upper_penalty = late_upper_penalty
    self.early_upper_force_scale = early_upper_force_scale
    self._body_names: list[str] | None = None
    self._lower_body_ids: list[int] | None = None
    self._upper_body_ids: list[int] | None = None
    self._lower_contact_time: torch.Tensor | None = None
    self._upper_contacted_once: torch.Tensor | None = None
    self._lower_contacted_once: torch.Tensor | None = None
    self._first_upper_before_lower_once: torch.Tensor | None = None
    self._timely_upper_once: torch.Tensor | None = None
    self._early_upper_once: torch.Tensor | None = None
    self._late_upper_once: torch.Tensor | None = None
    self._upper_delay_sum: torch.Tensor | None = None
    self._upper_delay_count: torch.Tensor | None = None

  def _maybe_initialize(self, env: ManagerBasedRlEnv) -> bool:
    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.found is not None
    if self._body_names is None:
      self._body_names = _get_sensor_body_names(sensor)
    if self._lower_body_ids is None:
      self._lower_body_ids = [
        i for i, name in enumerate(self._body_names) if name in self.lower_body_names
      ]
    if self._upper_body_ids is None:
      self._upper_body_ids = [
        i for i, name in enumerate(self._body_names) if name in self.upper_body_names
      ]
    if not self._lower_body_ids or not self._upper_body_ids:
      return False
    if (
      self._lower_contact_time is None
      or self._lower_contact_time.shape[0] != env.num_envs
    ):
      self._lower_contact_time = torch.full(
        (env.num_envs,), -1.0, device=env.device
      )
    if (
      self._upper_contacted_once is None
      or self._upper_contacted_once.shape[0] != env.num_envs
    ):
      self._upper_contacted_once = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      )
    for attr, dtype in (
      ("_lower_contacted_once", torch.bool),
      ("_first_upper_before_lower_once", torch.bool),
      ("_timely_upper_once", torch.bool),
      ("_early_upper_once", torch.bool),
      ("_late_upper_once", torch.bool),
    ):
      value = getattr(self, attr)
      if value is None or value.shape[0] != env.num_envs:
        setattr(
          self,
          attr,
          torch.zeros(env.num_envs, dtype=dtype, device=env.device),
        )
    for attr in ("_upper_delay_sum", "_upper_delay_count"):
      value = getattr(self, attr)
      if value is None or value.shape[0] != env.num_envs:
        setattr(self, attr, torch.zeros(env.num_envs, device=env.device))
    return True

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> dict[str, torch.Tensor]:
    if self._lower_contact_time is None or self._upper_contacted_once is None:
      return {}
    if env_ids is None:
      env_ids = slice(None)
    assert self._lower_contacted_once is not None
    assert self._first_upper_before_lower_once is not None
    assert self._timely_upper_once is not None
    assert self._early_upper_once is not None
    assert self._late_upper_once is not None
    assert self._upper_delay_sum is not None
    assert self._upper_delay_count is not None

    delay_count = self._upper_delay_count[env_ids].clamp_min(1.0)
    extras = {
      "Metrics/lower_then_upper/lower_contact_rate": _mean_selected(
        self._lower_contacted_once.float(), env_ids
      ),
      "Metrics/lower_then_upper/upper_contact_rate": _mean_selected(
        self._upper_contacted_once.float(), env_ids
      ),
      "Metrics/lower_then_upper/upper_before_lower_rate": _mean_selected(
        self._first_upper_before_lower_once.float(), env_ids
      ),
      "Metrics/lower_then_upper/early_upper_rate": _mean_selected(
        self._early_upper_once.float(), env_ids
      ),
      "Metrics/lower_then_upper/timely_upper_rate": _mean_selected(
        self._timely_upper_once.float(), env_ids
      ),
      "Metrics/lower_then_upper/late_upper_rate": _mean_selected(
        self._late_upper_once.float(), env_ids
      ),
      "Metrics/lower_then_upper/upper_delay_mean": (
        self._upper_delay_sum[env_ids] / delay_count
      ).mean(),
    }
    self._lower_contact_time[env_ids] = -1.0
    self._upper_contacted_once[env_ids] = False
    self._lower_contacted_once[env_ids] = False
    self._first_upper_before_lower_once[env_ids] = False
    self._timely_upper_once[env_ids] = False
    self._early_upper_once[env_ids] = False
    self._late_upper_once[env_ids] = False
    self._upper_delay_sum[env_ids] = 0.0
    self._upper_delay_count[env_ids] = 0.0
    return extras

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if not self._maybe_initialize(env):
      return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.found is not None
    assert self._lower_body_ids is not None
    assert self._upper_body_ids is not None
    assert self._lower_contact_time is not None
    assert self._upper_contacted_once is not None
    assert self._lower_contacted_once is not None
    assert self._first_upper_before_lower_once is not None
    assert self._timely_upper_once is not None
    assert self._early_upper_once is not None
    assert self._late_upper_once is not None
    assert self._upper_delay_sum is not None
    assert self._upper_delay_count is not None

    contact_now = sensor.data.found > 0
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    lower_first = torch.any(first_contact[:, self._lower_body_ids], dim=-1)
    upper_first = torch.any(first_contact[:, self._upper_body_ids], dim=-1)
    upper_contact_now = torch.any(contact_now[:, self._upper_body_ids], dim=-1)

    current_time = env.sim.data.time
    has_lower_contact = self._lower_contact_time >= 0.0
    delay_since_lower = current_time - self._lower_contact_time

    reward = torch.zeros(env.num_envs, device=env.device)
    first_lower_contact = lower_first & ~has_lower_contact
    reward += self.lower_first_bonus * first_lower_contact.float()
    self._lower_contacted_once |= first_lower_contact

    early_upper = upper_contact_now & (
      ~has_lower_contact | (delay_since_lower < self.min_delay_s)
    )
    reward -= self.early_upper_penalty * early_upper.float()
    first_upper_before_lower = upper_first & ~has_lower_contact
    self._first_upper_before_lower_once |= first_upper_before_lower
    self._early_upper_once |= early_upper
    if self.early_upper_force_scale > 0.0:
      assert sensor.data.force is not None
      upper_force = torch.norm(
        sensor.data.force[:, self._upper_body_ids], dim=-1
      ).max(dim=-1).values
      reward -= self.early_upper_force_scale * upper_force * early_upper.float()

    timely_upper = (
      upper_first
      & has_lower_contact
      & ~self._upper_contacted_once
      & (delay_since_lower >= self.min_delay_s)
      & (delay_since_lower <= self.max_delay_s)
    )
    reward += self.timely_upper_bonus * timely_upper.float()
    self._timely_upper_once |= timely_upper
    valid_upper_delay = upper_first & has_lower_contact & ~self._upper_contacted_once
    self._upper_delay_sum += torch.where(
      valid_upper_delay,
      delay_since_lower.clamp_min(0.0),
      torch.zeros_like(delay_since_lower),
    )
    self._upper_delay_count += valid_upper_delay.float()

    late_upper = (
      has_lower_contact
      & ~self._upper_contacted_once
      & (delay_since_lower > self.max_delay_s)
      & ~upper_contact_now
    )
    reward -= self.late_upper_penalty * late_upper.float()
    self._late_upper_once |= late_upper

    self._lower_contact_time = torch.where(
      first_lower_contact, current_time, self._lower_contact_time
    )
    self._upper_contacted_once |= upper_contact_now
    return reward


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class ReduceContactForceWeighted:
  """Reward for reducing contact force based on paper formula.
  
  Implements the weighted average-plus-peak contact force penalty.
  
  Args:
    env: The environment.
    sensor_name: Name of the contact sensor (e.g., "body_contact_force").
    high_weight_bodies: List of body names with high vulnerability (e.g., head, hands).
    shoulder_weight_bodies: List of shoulder body names with dedicated vulnerability weight.
    medium_weight_bodies: List of body names with medium vulnerability.
    high_weight: Sensitivity weight for high vulnerability bodies (default: 1000.0).
    shoulder_weight: Sensitivity weight for shoulder bodies (default: 5.0).
    medium_weight: Sensitivity weight for medium vulnerability bodies (default: 1.0).
    low_weight: Sensitivity weight for low vulnerability bodies (default: 0.5).
    alpha: Weight balancing average and peak forces (default: 0.3).
    
  Returns:
    Reward tensor of shape [B] where higher values indicate lower contact forces.
    This is a penalty (negative reward), so higher values mean less penalty.
  """

  def __init__(
    self,
    sensor_name: str,
    high_weight_bodies: tuple[str, ...] = (),
    shoulder_weight_bodies: tuple[str, ...] = (),
    medium_weight_bodies: tuple[str, ...] = (),
    high_weight: float = 10.0,
    shoulder_weight: float = 5.0,
    medium_weight: float = 1.0,
    low_weight: float = 0.1,
    alpha: float = 0.3,
    squash_scale: float = 0.02,
    tracked_body_names: tuple[str, ...] = (),
  ) -> None:
    self.sensor_name = sensor_name
    self.high_weight_bodies = high_weight_bodies
    self.shoulder_weight_bodies = shoulder_weight_bodies
    self.medium_weight_bodies = medium_weight_bodies
    self.high_weight = high_weight
    self.shoulder_weight = shoulder_weight
    self.medium_weight = medium_weight
    self.low_weight = low_weight
    self.alpha = alpha
    self.squash_scale = squash_scale
    self.tracked_body_names = tracked_body_names
    self._body_names: list[str] | None = None
    self._metric_sums: dict[str, torch.Tensor] = {}
    self._metric_steps: torch.Tensor | None = None

  def _maybe_initialize(self, env: ManagerBasedRlEnv) -> bool:
    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.force is not None
    if self._body_names is None:
      self._body_names = _get_sensor_body_names(sensor)
    if not self._body_names:
      return False
    if self._metric_steps is None or self._metric_steps.shape[0] != env.num_envs:
      self._metric_steps = torch.zeros(env.num_envs, device=env.device)
      self._metric_sums = {}
    return True

  def _accumulate(self, name: str, value: torch.Tensor) -> None:
    if name not in self._metric_sums:
      self._metric_sums[name] = torch.zeros_like(value)
    self._metric_sums[name] += value

  def _group_max(
    self,
    force_norm: torch.Tensor,
    body_names: tuple[str, ...],
  ) -> torch.Tensor:
    assert self._body_names is not None
    indexes = [i for i, name in enumerate(self._body_names) if name in body_names]
    if not indexes:
      return torch.zeros(force_norm.shape[0], device=force_norm.device)
    return force_norm[:, indexes].max(dim=-1).values

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> dict[str, torch.Tensor]:
    if self._metric_steps is None:
      return {}
    if env_ids is None:
      env_ids = slice(None)
    denom = self._metric_steps[env_ids].clamp_min(1.0)
    extras = {
      f"Metrics/contact_force/{name}": (value[env_ids] / denom).mean()
      for name, value in self._metric_sums.items()
    }
    self._metric_steps[env_ids] = 0.0
    for value in self._metric_sums.values():
      value[env_ids] = 0.0
    return extras

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    if not self._maybe_initialize(env):
      return torch.zeros(env.num_envs, device=env.device)
    assert self._body_names is not None
    assert self._metric_steps is not None
    sensor: ContactSensor = env.scene[self.sensor_name]
    assert sensor.data.force is not None
    assert sensor.data.found is not None

    forces = sensor.data.force
    contact_indicators = (sensor.data.found > 0).float()
    force_norm = torch.norm(contact_indicators.unsqueeze(-1) * forces, dim=-1)
    weights = _build_body_weight_tensor(
      body_names=self._body_names,
      device=forces.device,
      dtype=forces.dtype,
      high_weight_bodies=self.high_weight_bodies,
      shoulder_weight_bodies=self.shoulder_weight_bodies,
      medium_weight_bodies=self.medium_weight_bodies,
      high_weight=self.high_weight,
      shoulder_weight=self.shoulder_weight,
      medium_weight=self.medium_weight,
      low_weight=self.low_weight,
    )
    weighted_force_norm = force_norm * weights.unsqueeze(0)

    num_active_contacts = contact_indicators.sum(dim=-1, keepdim=True).clamp(min=1.0)
    average_term = weighted_force_norm.sum(dim=-1) / num_active_contacts.squeeze(-1)
    peak_term = weighted_force_norm.max(dim=-1)[0]
    penalty = average_term + self.alpha * peak_term

    self._metric_steps += 1.0
    high_max = self._group_max(force_norm, self.high_weight_bodies)
    shoulder_max = self._group_max(force_norm, self.shoulder_weight_bodies)
    medium_max = self._group_max(force_norm, self.medium_weight_bodies)
    protected = (
      set(self.high_weight_bodies)
      | set(self.shoulder_weight_bodies)
      | set(self.medium_weight_bodies)
    )
    low_bodies = tuple(name for name in self._body_names if name not in protected)
    self._accumulate("high_max", high_max)
    self._accumulate("shoulder_max", shoulder_max)
    self._accumulate("medium_max", medium_max)
    self._accumulate("low_max", self._group_max(force_norm, low_bodies))
    self._accumulate("weighted_average", average_term)
    self._accumulate("weighted_peak", peak_term)
    for body_name in self.tracked_body_names:
      self._accumulate(
        f"{_body_log_name(body_name)}_max",
        self._group_max(force_norm, (body_name,)),
      )

    if self.squash_scale > 0.0:
      penalty = torch.log1p(self.squash_scale * penalty) / self.squash_scale
    return -penalty


def reduce_contact_force_weighted(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  high_weight_bodies: tuple[str, ...] = (),
  shoulder_weight_bodies: tuple[str, ...] = (),
  medium_weight_bodies: tuple[str, ...] = (),
  high_weight: float = 10.0,
  shoulder_weight: float = 5.0,
  medium_weight: float = 1.0,
  low_weight: float = 0.1,
  alpha: float = 0.3,
  squash_scale: float = 0.02,
) -> torch.Tensor:
  reward = ReduceContactForceWeighted(
    sensor_name=sensor_name,
    high_weight_bodies=high_weight_bodies,
    shoulder_weight_bodies=shoulder_weight_bodies,
    medium_weight_bodies=medium_weight_bodies,
    high_weight=high_weight,
    shoulder_weight=shoulder_weight,
    medium_weight=medium_weight,
    low_weight=low_weight,
    alpha=alpha,
    squash_scale=squash_scale,
  )
  return reward(env)

def joint_wrench_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 1.0,
  threshold: float = 500.0,
  wrench_type: str = "total",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """关节广义力（joint wrench）惩罚：仅当关节总广义力范数超过阈值时才惩罚。

  使用 EntityData 的 joint_qfrc_*。wrench_type 可选 "total"（默认）、"constraint"、"actuator"。
  对每个 env 取关节广义力向量的 L2 范数，超过 threshold（单位与 qfrc 一致，如 Nm）时对超出部分做平方惩罚。

  Returns:
    负值惩罚，仅在超过阈值时有非零惩罚。
  """
  asset: Entity = env.scene[asset_cfg.name]
  if not asset.data.is_articulated:
    return torch.zeros(env.num_envs, device=asset.data.joint_pos.device)
  data_any = cast(Any, asset.data)
  if wrench_type == "total":
    qfrc = data_any.joint_qfrc_total  # (num_envs, num_joint_dofs)
  elif wrench_type == "constraint":
    qfrc = data_any.joint_qfrc_constraint
  elif wrench_type == "actuator":
    qfrc = data_any.joint_qfrc_actuator
  else:
    raise ValueError(f"joint_wrench_penalty: unknown wrench_type={wrench_type!r}")
  wrench_norm = torch.norm(qfrc, dim=-1)  # (num_envs,)
  excess = torch.clamp(wrench_norm - threshold, min=0.0)
  return -scale * (excess**2)


def motor_overcurrent_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 1.0,
  threshold: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """电机过流惩罚：用 |tau|/tau_max 近似 i/Imax，仅当超过 threshold 时惩罚。

  从 env.sim.mj_model.actuator_forcerange 取每轴 tau_max，对每个 env 取所有电机
  中 max(0, ratio - threshold) 的最大值再做平方惩罚。
  """
  asset: Entity = env.scene[asset_cfg.name]
  if not asset.data.is_actuated:
    return torch.zeros(env.num_envs, device=asset.data.actuator_force.device)
  device = asset.data.actuator_force.device
  tau = asset.data.actuator_force  # (num_envs, nu)
  ctrl_ids = asset.indexing.ctrl_ids
  if isinstance(ctrl_ids, slice):
    ctrl_ids = torch.arange(env.sim.mj_model.nu, device=device, dtype=torch.long)
  else:
    ctrl_ids = ctrl_ids.to(device)
  forcerange = env.sim.mj_model.actuator_forcerange  # (nu, 2)
  tau_max = torch.from_numpy(forcerange[ctrl_ids.cpu().numpy(), 1].copy()).to(
    device=device, dtype=tau.dtype
  )
  tau_max = tau_max.unsqueeze(0).clamp(min=1e-6)  # (1, nu)
  ratio = torch.abs(tau) / tau_max  # (num_envs, nu)
  excess = torch.clamp(ratio - threshold, min=0.0)
  max_excess = excess.max(dim=-1).values  # (num_envs,)
  return -scale * (max_excess**2)


def motor_back_emf_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 1.0,
  threshold: float = 0.1,
  p_max: float = 100.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """电机反电动势惩罚：当 torque 与 velocity 反向时 -tau*w 为再生功率，-tau*w/Pmax 超过阈值则惩罚。

  仅当 -tau*w > 0（反向）时计算 -tau*w/Pmax，超过 threshold 的部分做平方惩罚。
  对每个 env 先对全部电机求和再与阈值比较，或按轴超过阈值再求和（这里按轴 excess 求和）。
  """
  asset: Entity = env.scene[asset_cfg.name]
  if not asset.data.is_actuated:
    return torch.zeros(env.num_envs, device=asset.data.actuator_force.device)
  joint_ids, _ = asset.find_joints_by_actuator_names((".*",))
  if not joint_ids:
    return torch.zeros(env.num_envs, device=asset.data.actuator_force.device)
  device = asset.data.actuator_force.device
  tau = asset.data.actuator_force  # (num_envs, nu)
  joint_ids_t = torch.tensor(joint_ids, device=device, dtype=torch.long)
  w = asset.data.joint_vel[:, joint_ids_t]  # (num_envs, nu)
  # -tau*w > 0 表示反向（再生），归一化 -tau*w / Pmax
  neg_power = -tau * w  # (num_envs, nu)
  neg_power = torch.clamp(neg_power, min=0.0)  # 只考虑再生
  norm_regen = neg_power / (p_max + 1e-9)  # (num_envs, nu)
  excess = torch.clamp(norm_regen - threshold, min=0.0)
  # 对每个 env 取所有电机 excess 之和（或 max），这里用 sum
  total_excess = excess.sum(dim=-1)  # (num_envs,)
  return -scale * (total_excess**2)
