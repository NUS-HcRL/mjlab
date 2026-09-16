from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_log_name(body_name: str) -> str:
  return body_name.removeprefix("LINK_").lower()


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)

def bad_body_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Terminate when any named body is in contact according to a contact sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None

  if not body_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.found.device)

  slot_body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      slot_body_names.append(slot.primary_name)
      seen.add(slot.primary_name)

  body_to_index = {name: i for i, name in enumerate(slot_body_names)}
  selected_indexes = [body_to_index[name] for name in body_names if name in body_to_index]
  if not selected_indexes:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.found.device)

  found = sensor.data.found[:, selected_indexes]
  return torch.any(found > 0, dim=-1)


def bad_body_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
  body_force_thresholds: dict[str, float],
) -> torch.Tensor:
  return BadBodyContactForce()(env, sensor_name, body_names, body_force_thresholds)


class BadBodyContactForce:
  """Terminate when any named body contact-force norm exceeds its threshold.

  Every name in ``body_names`` must appear in ``body_force_thresholds``.
  """

  def __init__(self) -> None:
    self._body_trigger_sums: dict[str, torch.Tensor] = {}
    self._body_force_sums: dict[str, torch.Tensor] = {}
    self._steps: torch.Tensor | None = None

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> dict[str, torch.Tensor]:
    if self._steps is None:
      return {}
    if env_ids is None:
      env_ids = slice(None)
    denom = self._steps[env_ids].clamp_min(1.0)
    extras: dict[str, torch.Tensor] = {}
    for body_name, value in self._body_trigger_sums.items():
      log_name = _body_log_name(body_name)
      extras[f"Metrics/termination_force/{log_name}_trigger_rate"] = (
        value[env_ids] / denom
      ).mean()
    for body_name, value in self._body_force_sums.items():
      log_name = _body_log_name(body_name)
      extras[f"Metrics/termination_force/{log_name}_force_mean"] = (
        value[env_ids] / denom
      ).mean()
    self._steps[env_ids] = 0.0
    for value in self._body_trigger_sums.values():
      value[env_ids] = 0.0
    for value in self._body_force_sums.values():
      value[env_ids] = 0.0
    return extras

  def _accumulate(
    self,
    store: dict[str, torch.Tensor],
    name: str,
    value: torch.Tensor,
  ) -> None:
    if name not in store:
      store[name] = torch.zeros_like(value)
    store[name] += value

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    body_names: tuple[str, ...],
    body_force_thresholds: dict[str, float],
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None

    if not body_names:
      return torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=sensor.data.force.device,
      )

    if self._steps is None or self._steps.shape[0] != env.num_envs:
      self._steps = torch.zeros(env.num_envs, device=sensor.data.force.device)
      self._body_trigger_sums = {}
      self._body_force_sums = {}

    missing_thresholds = set(body_names) - set(body_force_thresholds)
    if missing_thresholds:
      raise ValueError(
        "bad_body_contact_force: body_force_thresholds missing entries for "
        f"{sorted(missing_thresholds)}"
      )

    slot_body_names = []
    seen = set()
    for slot in sensor._slots:
      if slot.primary_name not in seen:
        slot_body_names.append(slot.primary_name)
        seen.add(slot.primary_name)

    body_to_index = {name: i for i, name in enumerate(slot_body_names)}
    missing_bodies = set(body_names) - set(body_to_index)
    if missing_bodies:
      raise ValueError(
        "bad_body_contact_force: body_names not found on contact sensor "
        f"{sensor_name!r}: {sorted(missing_bodies)}"
      )

    terminate = torch.zeros(
      env.num_envs,
      dtype=torch.bool,
      device=sensor.data.force.device,
    )
    self._steps += 1.0
    for name in body_names:
      idx = body_to_index[name]
      threshold = body_force_thresholds[name]
      force_norm = torch.norm(sensor.data.force[:, idx], dim=-1)
      body_trigger = force_norm > threshold
      terminate |= body_trigger
      self._accumulate(self._body_trigger_sums, name, body_trigger.float())
      self._accumulate(self._body_force_sums, name, force_norm)
    return terminate


def nonfinite_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate envs whose robot state contains NaN/Inf."""
  asset: Entity = env.scene[asset_cfg.name]
  state_tensors = (
    asset.data.root_link_pos_w,
    asset.data.root_link_quat_w,
    asset.data.root_link_vel_w,
    asset.data.joint_pos,
    asset.data.joint_vel,
  )
  bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for tensor in state_tensors:
    if tensor is None:
      continue
    bad |= ~torch.isfinite(tensor).all(dim=-1).reshape(env.num_envs, -1).all(dim=-1)
  return bad


def invalid_physics_state(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  max_body_linear_speed: float = 20.0,
  max_joint_speed: float = 100.0,
  max_contact_force: float = 20_000.0,
) -> torch.Tensor:
  """Terminate finite but physically pathological simulator states."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  body_speed = torch.linalg.vector_norm(asset.data.body_link_lin_vel_w, dim=-1)
  joint_speed = asset.data.joint_vel.abs()
  contact_force = torch.linalg.vector_norm(sensor.data.force, dim=-1)

  return (
    (body_speed > max_body_linear_speed).any(dim=-1)
    | (joint_speed > max_joint_speed).any(dim=-1)
    | (contact_force > max_contact_force).any(dim=-1)
  )
