from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_MOTION_RESET_CACHE: dict[tuple[str, int, str, int], dict[str, torch.Tensor]] = {}
_MOTION_RESET_POOL_CACHE: dict[
  tuple[tuple[str, ...], int, str, int], dict[str, torch.Tensor]
] = {}
_LAST_RESET_DATA_MASK_ATTR = "_fall_last_reset_data_mask"
_FORCE_PULSE_STEPS_LEFT_ATTR = "_fall_force_pulse_steps_left"
_FORCE_PULSE_COOLDOWN_STEPS_LEFT_ATTR = "_fall_force_pulse_cooldown_steps_left"
_DATA_RESET_OBS_PENDING_ATTR = "_fall_data_reset_obs_pending"
_DATA_RESET_POOL_ROW_ATTR = "_fall_data_reset_pool_row"
_DATA_RESET_MOTION_POOL_ATTR = "_fall_data_reset_motion_pool"
_PRE_SWITCH_HISTORY_SUFFIXES = ("m4", "m3", "m2", "m1")


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
  norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-6)
  return quat / norm


def _csv_body_idx(asset_body_idx: int) -> int:
  """CSV body ids follow MuJoCo nbody and include world at index 0."""
  return asset_body_idx + 1


def _has_pre_switch_history_columns(fieldnames: Sequence[str]) -> bool:
  return "pre_m4_joint_pos_0" in fieldnames


def _pre_switch_joint_cols(prefix: str, num_joints: int, kind: str) -> list[str]:
  return [f"pre_{prefix}_joint_{kind}_{joint_idx}" for joint_idx in range(num_joints)]


def _pre_switch_base_quat_cols(prefix: str) -> list[str]:
  return [
    f"pre_{prefix}_base_quat_{axis}" for axis in ("w", "x", "y", "z")
  ]


def _pre_switch_base_ang_vel_cols(prefix: str) -> list[str]:
  return [
    f"pre_{prefix}_base_ang_vel_w_{axis}" for axis in ("x", "y", "z")
  ]


def _load_pre_switch_history_tensors(
  rows: list[dict[str, str]],
  fieldnames: Sequence[str],
  num_joints: int,
  device: str,
) -> dict[str, torch.Tensor] | None:
  """Load pre-switch observation history (4 frames: m4..m1) when CSV provides it."""
  if not _has_pre_switch_history_columns(fieldnames):
    return None

  num_rows = len(rows)
  num_frames = len(_PRE_SWITCH_HISTORY_SUFFIXES)
  joint_pos = torch.zeros(
    (num_rows, num_frames, num_joints), dtype=torch.float32, device=device
  )
  joint_vel = torch.zeros_like(joint_pos)
  base_quat = torch.zeros((num_rows, num_frames, 4), dtype=torch.float32, device=device)
  base_ang_vel_w = torch.zeros(
    (num_rows, num_frames, 3), dtype=torch.float32, device=device
  )

  for frame_idx, prefix in enumerate(_PRE_SWITCH_HISTORY_SUFFIXES):
    joint_pos_cols = _pre_switch_joint_cols(prefix, num_joints, "pos")
    joint_vel_cols = _pre_switch_joint_cols(prefix, num_joints, "vel")
    quat_cols = _pre_switch_base_quat_cols(prefix)
    ang_cols = _pre_switch_base_ang_vel_cols(prefix)
    missing = [
      col
      for col in joint_pos_cols + joint_vel_cols + quat_cols + ang_cols
      if col not in fieldnames
    ]
    if missing:
      raise ValueError(
        "Reset motion CSV has pre-switch history marker but is missing columns: "
        f"{', '.join(missing[:8])}"
        + ("..." if len(missing) > 8 else "")
      )
    joint_pos[:, frame_idx] = torch.tensor(
      [[float(row[col]) for col in joint_pos_cols] for row in rows],
      dtype=torch.float32,
      device=device,
    )
    joint_vel[:, frame_idx] = torch.tensor(
      [[float(row[col]) for col in joint_vel_cols] for row in rows],
      dtype=torch.float32,
      device=device,
    )
    base_quat[:, frame_idx] = torch.tensor(
      [[float(row[col]) for col in quat_cols] for row in rows],
      dtype=torch.float32,
      device=device,
    )
    base_ang_vel_w[:, frame_idx] = torch.tensor(
      [[float(row[col]) for col in ang_cols] for row in rows],
      dtype=torch.float32,
      device=device,
    )

  base_quat = _normalize_quat(base_quat.reshape(-1, 4)).reshape(num_rows, num_frames, 4)
  return {
    "pre_joint_pos": joint_pos,
    "pre_joint_vel": joint_vel,
    "pre_base_quat": base_quat,
    "pre_base_ang_vel_w": base_ang_vel_w,
    "has_pre_history": torch.ones(num_rows, dtype=torch.bool, device=device),
  }


def _root_state_is_placeholder(root_state: torch.Tensor) -> bool:
  if root_state.numel() == 0:
    return True
  identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root_state.device)
  pos_is_zero = torch.max(torch.abs(root_state[:, 0:3])).item() < 1e-6
  quat_is_identity = (
    torch.max(torch.abs(root_state[:, 3:7] - identity_quat.unsqueeze(0))).item() < 1e-6
  )
  vel_is_zero = torch.max(torch.abs(root_state[:, 7:13])).item() < 1e-6
  return pos_is_zero and quat_is_identity and vel_is_zero


def _load_motion_reset_csv(
  path: str,
  root_body_idx: int,
  device: str,
  expected_num_joints: int,
) -> dict[str, torch.Tensor]:
  with open(path, newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

  if not rows:
    raise ValueError(f"Reset motion '{path}' is empty.")

  csv_root_body_idx = _csv_body_idx(root_body_idx)
  joint_pos_cols = [f"joint_pos_{joint_idx}" for joint_idx in range(expected_num_joints)]
  joint_vel_cols = [f"joint_vel_{joint_idx}" for joint_idx in range(expected_num_joints)]
  root_pos_cols = [f"body_pos_w_{csv_root_body_idx}_{axis}" for axis in ("x", "y", "z")]
  root_quat_cols = [
    f"body_quat_w_{csv_root_body_idx}_{axis}" for axis in ("w", "x", "y", "z")
  ]
  root_lin_vel_cols = [
    f"body_lin_vel_w_{csv_root_body_idx}_{axis}" for axis in ("x", "y", "z")
  ]
  root_ang_vel_cols = [
    f"body_ang_vel_w_{csv_root_body_idx}_{axis}" for axis in ("x", "y", "z")
  ]

  required_cols = joint_pos_cols + joint_vel_cols + root_pos_cols + root_quat_cols
  missing_required = [col for col in required_cols if col not in fieldnames]
  if missing_required:
    raise ValueError(
      f"Reset motion '{path}' is missing required csv columns: "
      f"{', '.join(missing_required[:8])}"
      + ("..." if len(missing_required) > 8 else "")
    )

  has_root_lin_vel = all(col in fieldnames for col in root_lin_vel_cols)
  has_root_ang_vel = all(col in fieldnames for col in root_ang_vel_cols)
  if has_root_lin_vel != has_root_ang_vel:
    raise ValueError(
      f"Reset motion '{path}' must contain both root linear/angular velocity "
      "columns or neither of them."
    )

  joint_pos = torch.tensor(
    [[float(row[col]) for col in joint_pos_cols] for row in rows],
    dtype=torch.float32,
    device=device,
  )
  joint_vel = torch.tensor(
    [[float(row[col]) for col in joint_vel_cols] for row in rows],
    dtype=torch.float32,
    device=device,
  )
  root_pos = torch.tensor(
    [[float(row[col]) for col in root_pos_cols] for row in rows],
    dtype=torch.float32,
    device=device,
  )
  root_quat = torch.tensor(
    [[float(row[col]) for col in root_quat_cols] for row in rows],
    dtype=torch.float32,
    device=device,
  )
  if has_root_lin_vel:
    root_lin_vel = torch.tensor(
      [[float(row[col]) for col in root_lin_vel_cols] for row in rows],
      dtype=torch.float32,
      device=device,
    )
    root_ang_vel = torch.tensor(
      [[float(row[col]) for col in root_ang_vel_cols] for row in rows],
      dtype=torch.float32,
      device=device,
    )
  else:
    root_lin_vel = torch.zeros_like(root_pos)
    root_ang_vel = torch.zeros_like(root_pos)

  num_rows = len(rows)
  num_frames = len(_PRE_SWITCH_HISTORY_SUFFIXES)
  pre_joint_pos = torch.zeros(
    (num_rows, num_frames, expected_num_joints),
    dtype=torch.float32,
    device=device,
  )
  dataset: dict[str, torch.Tensor] = {
    "root_state": torch.cat(
      [root_pos, root_quat, root_lin_vel, root_ang_vel],
      dim=-1,
    ),
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "has_pre_history": torch.zeros(num_rows, dtype=torch.bool, device=device),
    "pre_joint_pos": pre_joint_pos,
    "pre_joint_vel": torch.zeros_like(pre_joint_pos),
    "pre_base_quat": torch.zeros(
      (num_rows, num_frames, 4), dtype=torch.float32, device=device
    ),
    "pre_base_ang_vel_w": torch.zeros(
      (num_rows, num_frames, 3), dtype=torch.float32, device=device
    ),
  }
  pre_history = _load_pre_switch_history_tensors(
    rows=rows,
    fieldnames=fieldnames,
    num_joints=expected_num_joints,
    device=device,
  )
  if pre_history is not None:
    dataset.update(pre_history)
  return dataset


def _load_motion_reset_file(
  path: str,
  root_body_idx: int,
  device: str,
  expected_num_joints: int,
) -> dict[str, torch.Tensor]:
  cache_key = (path, root_body_idx, device, expected_num_joints)
  cached = _MOTION_RESET_CACHE.get(cache_key)
  if cached is not None:
    return cached

  cached = _load_motion_reset_csv(
    path=path,
    root_body_idx=root_body_idx,
    device=device,
    expected_num_joints=expected_num_joints,
  )
  _MOTION_RESET_CACHE[cache_key] = cached
  return cached


def _get_motion_reset_pool(
  motion_files: Sequence[str],
  root_body_idx: int,
  device: str,
  expected_num_joints: int,
) -> dict[str, torch.Tensor]:
  cache_key = (tuple(motion_files), root_body_idx, device, expected_num_joints)
  cached = _MOTION_RESET_POOL_CACHE.get(cache_key)
  if cached is not None:
    return cached

  datasets = [
    _load_motion_reset_file(
      path=path,
      root_body_idx=root_body_idx,
      device=device,
      expected_num_joints=expected_num_joints,
    )
    for path in motion_files
  ]
  if not datasets:
    raise ValueError("data reset requested but no motion files were configured.")

  cached = {
    "root_state": torch.cat([motion["root_state"] for motion in datasets], dim=0),
    "joint_pos": torch.cat([motion["joint_pos"] for motion in datasets], dim=0),
    "joint_vel": torch.cat([motion["joint_vel"] for motion in datasets], dim=0),
    "has_pre_history": torch.cat(
      [motion["has_pre_history"] for motion in datasets], dim=0
    ),
    "pre_joint_pos": torch.cat([motion["pre_joint_pos"] for motion in datasets], dim=0),
    "pre_joint_vel": torch.cat([motion["pre_joint_vel"] for motion in datasets], dim=0),
    "pre_base_quat": torch.cat([motion["pre_base_quat"] for motion in datasets], dim=0),
    "pre_base_ang_vel_w": torch.cat(
      [motion["pre_base_ang_vel_w"] for motion in datasets], dim=0
    ),
  }
  _MOTION_RESET_POOL_CACHE[cache_key] = cached
  return cached


def _default_states(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  default_root_state = asset.data.default_root_state
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  assert default_root_state is not None
  assert default_joint_pos is not None
  assert default_joint_vel is not None

  root_state = default_root_state[env_ids].clone()
  root_state[:, 0:3] += env.scene.env_origins[env_ids]
  joint_pos = default_joint_pos[env_ids].clone()
  joint_vel = default_joint_vel[env_ids].clone()
  return root_state, joint_pos, joint_vel


def _sample_tilt_states(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  tilt_pose_range: dict[str, tuple[float, float]],
  tilt_velocity_range: dict[str, tuple[float, float]] | None,
  tilt_joint_position_range: tuple[float, float],
  tilt_joint_velocity_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  root_state, joint_pos, joint_vel = _default_states(env, asset, env_ids)

  pose_ranges = torch.tensor(
    [
      tilt_pose_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ],
    device=env.device,
    dtype=torch.float32,
  )
  pose_samples = sample_uniform(
    pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  root_state[:, 0:3] += pose_samples[:, 0:3]
  root_state[:, 3:7] = quat_mul(
    root_state[:, 3:7],
    quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    ),
  )

  if tilt_velocity_range is None:
    tilt_velocity_range = {}
  vel_ranges = torch.tensor(
    [
      tilt_velocity_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ],
    device=env.device,
    dtype=torch.float32,
  )
  vel_samples = sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  root_state[:, 7:13] += vel_samples

  joint_pos += sample_uniform(
    tilt_joint_position_range[0],
    tilt_joint_position_range[1],
    joint_pos.shape,
    device=env.device,
  )
  joint_vel += sample_uniform(
    tilt_joint_velocity_range[0],
    tilt_joint_velocity_range[1],
    joint_vel.shape,
    device=env.device,
  )

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  joint_limits = soft_joint_pos_limits[env_ids]
  joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

  return root_state, joint_pos, joint_vel


def _sample_motion_states(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  motion_files: Sequence[str],
  data_root_body_name: str,
  data_pose_range: dict[str, tuple[float, float]] | None,
  data_velocity_range: dict[str, tuple[float, float]] | None,
  data_joint_position_range: tuple[float, float],
  data_joint_velocity_range: tuple[float, float],
  asset_name: str,
  use_data_reset_obs_history: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  root_ids, _ = asset.find_bodies((data_root_body_name,), preserve_order=True)
  if not root_ids:
    raise ValueError(
      f"Could not find data root body '{data_root_body_name}' in asset "
      f"'{asset_name}'."
    )
  root_body_idx = root_ids[0]

  motion_pool = _get_motion_reset_pool(
    motion_files=motion_files,
    root_body_idx=root_body_idx,
    device=env.device,
    expected_num_joints=asset.num_joints,
  )
  root_state_pool = motion_pool["root_state"]
  joint_pos_pool = motion_pool["joint_pos"]
  joint_vel_pool = motion_pool["joint_vel"]
  if root_state_pool.shape[0] == 0:
    raise ValueError("data reset requested but loaded motion files contain no states.")

  state_ids = torch.randint(
    low=0,
    high=root_state_pool.shape[0],
    size=(len(env_ids),),
    device=env.device,
  )
  if _root_state_is_placeholder(root_state_pool):
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_state = default_root_state[env_ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
  else:
    root_state = root_state_pool[state_ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
    root_state[:, 3:7] = _normalize_quat(root_state[:, 3:7])
  joint_pos = joint_pos_pool[state_ids].clone()
  joint_vel = joint_vel_pool[state_ids].clone()

  if data_pose_range is None:
    data_pose_range = {}
  pose_ranges = torch.tensor(
    [
      data_pose_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ],
    device=env.device,
    dtype=torch.float32,
  )
  pose_samples = sample_uniform(
    pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  root_state[:, 0:3] += pose_samples[:, 0:3]
  root_state[:, 3:7] = quat_mul(
    root_state[:, 3:7],
    quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    ),
  )
  root_state[:, 3:7] = _normalize_quat(root_state[:, 3:7])

  if data_velocity_range is None:
    data_velocity_range = {}
  vel_ranges = torch.tensor(
    [
      data_velocity_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ],
    device=env.device,
    dtype=torch.float32,
  )
  vel_samples = sample_uniform(
    vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
  )
  root_state[:, 7:13] += vel_samples

  joint_pos += sample_uniform(
    data_joint_position_range[0],
    data_joint_position_range[1],
    joint_pos.shape,
    device=env.device,
  )
  joint_vel += sample_uniform(
    data_joint_velocity_range[0],
    data_joint_velocity_range[1],
    joint_vel.shape,
    device=env.device,
  )

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  joint_limits = soft_joint_pos_limits[env_ids]
  joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

  if use_data_reset_obs_history:
    _register_data_reset_obs_history(
      env=env,
      data_env_ids=env_ids,
      pool_row_ids=state_ids,
      motion_pool=motion_pool,
    )
  return root_state, joint_pos, joint_vel


def _init_data_reset_obs_history_state(env: ManagerBasedRlEnv) -> None:
  if getattr(env, _DATA_RESET_OBS_PENDING_ATTR, None) is None:
    setattr(
      env,
      _DATA_RESET_OBS_PENDING_ATTR,
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    setattr(
      env,
      _DATA_RESET_POOL_ROW_ATTR,
      torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
    )


def _register_data_reset_obs_history(
  env: ManagerBasedRlEnv,
  data_env_ids: torch.Tensor,
  pool_row_ids: torch.Tensor,
  motion_pool: dict[str, torch.Tensor],
) -> None:
  """Mark envs that should receive CSV pre-switch frames in observation history."""
  _init_data_reset_obs_history_state(env)
  pending = getattr(env, _DATA_RESET_OBS_PENDING_ATTR)
  pool_row = getattr(env, _DATA_RESET_POOL_ROW_ATTR)
  assert isinstance(pending, torch.Tensor)
  assert isinstance(pool_row, torch.Tensor)
  pending[data_env_ids] = motion_pool["has_pre_history"][pool_row_ids]
  pool_row[data_env_ids] = pool_row_ids
  setattr(env, _DATA_RESET_MOTION_POOL_ATTR, motion_pool)


def clear_data_reset_obs_history_state(env: ManagerBasedRlEnv) -> None:
  """Clear pending data-reset observation history after it has been applied."""
  setattr(env, _DATA_RESET_OBS_PENDING_ATTR, None)
  setattr(env, _DATA_RESET_POOL_ROW_ATTR, None)
  setattr(env, _DATA_RESET_MOTION_POOL_ATTR, None)


def try_build_data_reset_obs_history_frames(
  env: ManagerBasedRlEnv,
  term_name: str,
  current_obs: torch.Tensor,
  history_length: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
  """Build chronological obs history ``[pre_m4, ..., pre_m1, current]`` for one term.

  Returns:
    ``(env_ids, frames)`` with ``frames`` shape ``(n, history_length, feat_dim)``, or
    ``None`` when this term should use the default append/backfill path.
  """
  pending = getattr(env, _DATA_RESET_OBS_PENDING_ATTR, None)
  pool_row = getattr(env, _DATA_RESET_POOL_ROW_ATTR, None)
  motion_pool = getattr(env, _DATA_RESET_MOTION_POOL_ATTR, None)
  if (
    pending is None
    or pool_row is None
    or motion_pool is None
    or not pending.any()
  ):
    return None

  num_pre_frames = history_length - 1
  if num_pre_frames != len(_PRE_SWITCH_HISTORY_SUFFIXES):
    return None

  env_ids = pending.nonzero(as_tuple=False).squeeze(-1)
  if env_ids.ndim == 0:
    env_ids = env_ids.unsqueeze(0)
  row_ids = pool_row[env_ids]
  valid_mask = motion_pool["has_pre_history"][row_ids]
  if not valid_mask.any():
    return None
  env_ids = env_ids[valid_mask]
  row_ids = row_ids[valid_mask]

  asset: Entity = env.scene["robot"]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  gravity_w = asset.data.gravity_vec_w

  if term_name == "actions":
    frames = torch.zeros(
      (env_ids.numel(), history_length, current_obs.shape[-1]),
      device=env.device,
      dtype=current_obs.dtype,
    )
    return env_ids, frames

  if term_name == "joint_pos":
    pre = motion_pool["pre_joint_pos"][row_ids] - default_joint_pos[env_ids].unsqueeze(1)
    frames = torch.cat([pre, current_obs[env_ids].unsqueeze(1)], dim=1)
    return env_ids, frames

  if term_name == "joint_vel":
    pre = motion_pool["pre_joint_vel"][row_ids] - default_joint_vel[env_ids].unsqueeze(1)
    frames = torch.cat([pre, current_obs[env_ids].unsqueeze(1)], dim=1)
    return env_ids, frames

  if term_name == "base_ang_vel":
    pre_quat = motion_pool["pre_base_quat"][row_ids]
    pre_ang_w = motion_pool["pre_base_ang_vel_w"][row_ids]
    n, t, _ = pre_ang_w.shape
    pre_b = quat_apply_inverse(
      pre_quat.reshape(n * t, 4),
      pre_ang_w.reshape(n * t, 3),
    ).reshape(n, t, 3)
    frames = torch.cat([pre_b, current_obs[env_ids].unsqueeze(1)], dim=1)
    return env_ids, frames

  if term_name == "projected_gravity":
    pre_quat = motion_pool["pre_base_quat"][row_ids]
    gravity = gravity_w[env_ids].unsqueeze(1).expand(-1, num_pre_frames, -1)
    n, t, _ = gravity.shape
    pre_g = quat_apply_inverse(
      pre_quat.reshape(n * t, 4),
      gravity.reshape(n * t, 3),
    ).reshape(n, t, 3)
    frames = torch.cat([pre_g, current_obs[env_ids].unsqueeze(1)], dim=1)
    return env_ids, frames

  return None


def _write_state_and_forward(
  env: ManagerBasedRlEnv,
  asset: Entity,
  env_ids: torch.Tensor,
  root_state: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> None:
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
  asset.clear_state(env_ids=env_ids)
  env.sim.forward()


def reset_root_state_mixed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  tilt_pose_range: dict[str, tuple[float, float]],
  tilt_velocity_range: dict[str, tuple[float, float]] | None = None,
  tilt_joint_position_range: tuple[float, float] = (0.0, 0.0),
  tilt_joint_velocity_range: tuple[float, float] = (0.0, 0.0),
  data_probability: float = 0.0,
  motion_files: Sequence[str] = (),
  data_root_body_name: str = "LINK_BASE",
  data_pose_range: dict[str, tuple[float, float]] | None = None,
  data_velocity_range: dict[str, tuple[float, float]] | None = None,
  data_joint_position_range: tuple[float, float] = (0.0, 0.0),
  data_joint_velocity_range: tuple[float, float] = (0.0, 0.0),
  use_data_reset_obs_history: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robot using a mixture of dangerous tilt states and motion data states."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return

  reset_data_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  setattr(env, _LAST_RESET_DATA_MASK_ATTR, reset_data_mask)
  _init_data_reset_obs_history_state(env)
  pending = getattr(env, _DATA_RESET_OBS_PENDING_ATTR)
  pool_row = getattr(env, _DATA_RESET_POOL_ROW_ATTR)
  assert isinstance(pending, torch.Tensor)
  assert isinstance(pool_row, torch.Tensor)
  pending[env_ids] = False
  pool_row[env_ids] = -1

  asset: Entity = env.scene[asset_cfg.name]
  use_data = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
  if motion_files and data_probability > 0.0:
    use_data = torch.rand(len(env_ids), device=env.device) < float(data_probability)

  tilt_env_ids = env_ids[~use_data]
  if tilt_env_ids.numel() > 0:
    tilt_root, tilt_joint_pos, tilt_joint_vel = _sample_tilt_states(
      env=env,
      asset=asset,
      env_ids=tilt_env_ids,
      tilt_pose_range=tilt_pose_range,
      tilt_velocity_range=tilt_velocity_range,
      tilt_joint_position_range=tilt_joint_position_range,
      tilt_joint_velocity_range=tilt_joint_velocity_range,
    )
    _write_state_and_forward(
      env, asset, tilt_env_ids, tilt_root, tilt_joint_pos, tilt_joint_vel
    )

  data_env_ids = env_ids[use_data]
  if data_env_ids.numel() > 0:
    reset_data_mask[data_env_ids] = True
    data_root, data_joint_pos, data_joint_vel = _sample_motion_states(
      env=env,
      asset=asset,
      env_ids=data_env_ids,
      motion_files=motion_files,
      data_root_body_name=data_root_body_name,
      data_pose_range=data_pose_range,
      data_velocity_range=data_velocity_range,
      data_joint_position_range=data_joint_position_range,
      data_joint_velocity_range=data_joint_velocity_range,
      asset_name=asset_cfg.name,
      use_data_reset_obs_history=use_data_reset_obs_history,
    )
    _write_state_and_forward(
      env, asset, data_env_ids, data_root, data_joint_pos, data_joint_vel
    )


def push_by_setting_velocity_preserve_data(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  preserve_data_reset_states: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply reset push without modifying envs initialized from motion data."""
  if preserve_data_reset_states:
    data_mask = getattr(env, _LAST_RESET_DATA_MASK_ATTR, None)
    if data_mask is not None:
      data_mask = data_mask.to(env.device).bool()
      env_ids = env_ids[~data_mask[env_ids]]
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque_axiswise(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  force_axis_range: dict[str, tuple[float, float]],
  torque_axis_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply external wrench with per-axis ranges in world frame.

  force_axis_range keys: x, y, z
  torque_axis_range keys: roll, pitch, yaw
  """
  if env_ids.numel() == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  num_bodies = (
    len(asset_cfg.body_ids)
    if isinstance(asset_cfg.body_ids, list)
    else asset.num_bodies
  )
  size = (len(env_ids), num_bodies, 3)

  force_ranges = torch.tensor(
    [force_axis_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")],
    device=env.device,
    dtype=torch.float32,
  )
  torque_ranges = torch.tensor(
    [torque_axis_range.get(k, (0.0, 0.0)) for k in ("roll", "pitch", "yaw")],
    device=env.device,
    dtype=torch.float32,
  )
  forces = sample_uniform(
    force_ranges[:, 0], force_ranges[:, 1], size=size, device=env.device
  )
  torques = sample_uniform(
    torque_ranges[:, 0], torque_ranges[:, 1], size=size, device=env.device
  )
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
  )


def apply_external_force_torque_axiswise_pulse(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  force_axis_range: dict[str, tuple[float, float]] | None = None,
  torque_axis_range: dict[str, tuple[float, float]] | None = None,
  duration_steps: int | None = None,
  duration_steps_range: tuple[int, int] | None = None,
  pulse_probability: float = 1.0,
  cooldown_steps: int = 0,
  preserve_data_reset_states: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Single-event external wrench pulse manager (interval mode).

  This function is designed to be called every env step via one interval event:
  1) tick and clear expired pulses;
  2) detect envs just reset in this step and start new pulses.
  """
  steps_left = getattr(env, _FORCE_PULSE_STEPS_LEFT_ATTR, None)
  if steps_left is None or not isinstance(steps_left, torch.Tensor):
    steps_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, _FORCE_PULSE_STEPS_LEFT_ATTR, steps_left)
  cooldown_left = getattr(env, _FORCE_PULSE_COOLDOWN_STEPS_LEFT_ATTR, None)
  if cooldown_left is None or not isinstance(cooldown_left, torch.Tensor):
    cooldown_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    setattr(env, _FORCE_PULSE_COOLDOWN_STEPS_LEFT_ATTR, cooldown_left)

  # 1) Tick existing pulses for all envs.
  tick_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
  active_mask = steps_left[tick_env_ids] > 0
  if active_mask.any():
    active_env_ids = tick_env_ids[active_mask]
    steps_left[active_env_ids] -= 1

    finished_mask = steps_left[active_env_ids] <= 0
    if finished_mask.any():
      finished_env_ids = active_env_ids[finished_mask]
      asset: Entity = env.scene[asset_cfg.name]
      num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
      )
      zeros = torch.zeros((len(finished_env_ids), num_bodies, 3), device=env.device)
      asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=finished_env_ids, body_ids=asset_cfg.body_ids
      )

  cooldown_left = torch.clamp(cooldown_left - 1, min=0)
  setattr(env, _FORCE_PULSE_COOLDOWN_STEPS_LEFT_ATTR, cooldown_left)

  # 2) Start pulses for envs that were just reset in current env step.
  if duration_steps is None:
    if duration_steps_range is None:
      duration_steps = 1
    else:
      duration_low_raw, duration_high_raw = duration_steps_range
      duration_low = int(min(duration_low_raw, duration_high_raw))
      duration_high = int(max(duration_low_raw, duration_high_raw))
      duration_steps = int(
        torch.randint(
          low=duration_low,
          high=duration_high + 1,
          size=(1,),
          device=env.device,
        ).item()
      )
  if duration_steps <= 0:
    return
  just_reset_env_ids = torch.nonzero(
    env.episode_length_buf == 0, as_tuple=False
  ).squeeze(-1)
  if just_reset_env_ids.numel() == 0:
    return
  if force_axis_range is None:
    force_axis_range = {}
  if torque_axis_range is None:
    torque_axis_range = {}

  steps_left[just_reset_env_ids] = 0
  target_env_ids = just_reset_env_ids
  if preserve_data_reset_states:
    data_mask = getattr(env, _LAST_RESET_DATA_MASK_ATTR, None)
    if data_mask is not None:
      data_mask = data_mask.to(env.device).bool()
      target_env_ids = just_reset_env_ids[~data_mask[just_reset_env_ids]]
  if target_env_ids.numel() == 0:
    return
  if pulse_probability <= 0.0:
    return
  if pulse_probability < 1.0:
    trigger_mask = torch.rand(len(target_env_ids), device=env.device) < float(
      pulse_probability
    )
    target_env_ids = target_env_ids[trigger_mask]
    if target_env_ids.numel() == 0:
      return
  if cooldown_steps > 0:
    target_env_ids = target_env_ids[cooldown_left[target_env_ids] <= 0]
    if target_env_ids.numel() == 0:
      return

  apply_external_force_torque_axiswise(
    env=env,
    env_ids=target_env_ids,
    force_axis_range=force_axis_range,
    torque_axis_range=torque_axis_range,
    asset_cfg=asset_cfg,
  )
  steps_left[target_env_ids] = int(duration_steps)
  if cooldown_steps > 0:
    cooldown_left[target_env_ids] = int(cooldown_steps)


def randomize_gravity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  nominal_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
  roll_range: tuple[float, float] = (-0.08, 0.08),
  pitch_range: tuple[float, float] = (-0.08, 0.08),
  magnitude_range: tuple[float, float] = (0.92, 1.08),
) -> None:
  """Randomize world gravity and sync ``EntityData.gravity_vec_w`` for projected-gravity obs.

  Samples a small roll/pitch tilt and magnitude scale around ``nominal_gravity``. When the
  batched sim exposes independent rows in ``model.opt.gravity``, each env gets its own
  sample; otherwise one sample is applied to every env (shared MuJoCo option buffer).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)

  asset: Entity = env.scene[asset_cfg.name]
  n = env_ids.numel()
  roll = sample_uniform(roll_range[0], roll_range[1], (n,), device=env.device)
  pitch = sample_uniform(pitch_range[0], pitch_range[1], (n,), device=env.device)
  magnitude = sample_uniform(
    magnitude_range[0], magnitude_range[1], (n,), device=env.device
  )

  g_nom = torch.tensor(nominal_gravity, device=env.device, dtype=torch.float32)
  g_scale = torch.linalg.vector_norm(g_nom).clamp_min(1e-6)
  g_unit = g_nom / g_scale

  tilt_quat = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
  g_tilted = quat_apply(tilt_quat, g_unit.unsqueeze(0).expand(n, -1))
  g_world = g_tilted * (magnitude * g_scale).unsqueeze(-1)
  g_dir = g_world / torch.linalg.vector_norm(g_world, dim=-1, keepdim=True).clamp_min(1e-6)

  gravity_field = env.sim.model.opt.gravity
  if gravity_field.ndim == 1 or gravity_field.shape[0] == 1:
    g_shared = g_world[0]
    gravity_field[:] = g_shared
    env.sim.mj_model.opt.gravity[:] = g_shared.detach().cpu().numpy()
    asset.data.gravity_vec_w[env_ids] = g_dir[0].unsqueeze(0).expand(n, -1)
  else:
    gravity_field[env_ids] = g_world
    asset.data.gravity_vec_w[env_ids] = g_dir
