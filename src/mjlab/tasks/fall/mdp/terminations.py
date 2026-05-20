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
  """Terminate when any named body contact-force norm exceeds its threshold.

  Every name in ``body_names`` must appear in ``body_force_thresholds``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None

  if not body_names:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.force.device)

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

  terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=sensor.data.force.device)
  for name in body_names:
    idx = body_to_index[name]
    threshold = body_force_thresholds[name]
    force_norm = torch.norm(sensor.data.force[:, idx], dim=-1)
    terminate |= force_norm > threshold
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