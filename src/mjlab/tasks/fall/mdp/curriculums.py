from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict
from typing import Any, cast

from typing_extensions import NotRequired

import numpy as np
import torch

from mjlab.asset_zoo.robots.engineai_pm01.pm01_8 import (
  EFFORT_LIMIT_Q25,
  PM_ACTION_SCALE,
  SOLIMP_CONTACT_DEFAULT,
  SOLIMP_CONTACT_SOFT_6mm,
  SOLREF_CONTACT_DEFAULT,
  SOLREF_CONTACT_SOFT_6mm,
)
from mjlab.entity import Entity
from mjlab.envs.mdp.actions.joint_actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PushStage(TypedDict):
  step: int
  x: NotRequired[tuple[float, float]]
  y: NotRequired[tuple[float, float]]
  z: NotRequired[tuple[float, float]]
  roll: NotRequired[tuple[float, float]]
  pitch: NotRequired[tuple[float, float]]
  yaw: NotRequired[tuple[float, float]]


class ResetInitStage(TypedDict):
  step: int
  data_probability: float
  tilt_pose_range: NotRequired[dict[str, tuple[float, float]]]
  tilt_velocity_range: NotRequired[dict[str, tuple[float, float]]]
  tilt_joint_position_range: NotRequired[tuple[float, float]]
  tilt_joint_velocity_range: NotRequired[tuple[float, float]]


class ForcePulseStage(TypedDict):
  step: int
  duration_steps_range: NotRequired[tuple[int, int]]
  force_axis_range: NotRequired[dict[str, tuple[float, float]]]
  torque_axis_range: NotRequired[dict[str, tuple[float, float]]]


class WeightStage(TypedDict):
  step: int
  scale: float


class Q25EffortLimitStage(TypedDict):
  """PM1 Q25 actuator torque ceiling node (see ``fall_env_cfg`` q25_effort_limit)."""

  step: int
  effort_limit: float


class SoftContactSolStage(TypedDict):
  """Knee/elbow contact solref/solimp node (see ``fall_env_cfg`` pm_soft_contact)."""

  step: int
  profile: Literal["default", "soft_6mm"]


_CONTACT_SOL_PROFILES: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
  "default": (SOLREF_CONTACT_DEFAULT, SOLIMP_CONTACT_DEFAULT),
  "soft_6mm": (SOLREF_CONTACT_SOFT_6mm, SOLIMP_CONTACT_SOFT_6mm),
}


def _set_geom_sol_params(
  env: ManagerBasedRlEnv,
  geom_ids: list[int],
  solref: tuple[float, ...],
  solimp: tuple[float, ...],
) -> None:
  """Write solref/solimp for geoms into CPU ``mj_model`` and batched sim model if present."""
  mj = env.sim.mj_model
  solref_np = np.asarray(solref, dtype=np.float64)
  solimp_np = np.asarray(solimp, dtype=np.float64)
  for gid in geom_ids:
    mj.geom_solref[gid] = solref_np
    imp = mj.geom_solimp[gid].copy()
    imp[: len(solimp_np)] = solimp_np
    mj.geom_solimp[gid] = imp

  model = env.sim.model
  if not hasattr(model, "geom_solref") or not hasattr(model, "geom_solimp"):
    return

  device = env.device
  env_ids_all = torch.arange(env.num_envs, device=device, dtype=torch.int)
  gid_t = torch.tensor(geom_ids, device=device, dtype=torch.long)
  solref_t = torch.tensor(solref_np, device=device, dtype=torch.float32)
  solimp_t = torch.tensor(solimp_np, device=device, dtype=torch.float32)
  try:
    model.geom_solref[env_ids_all[:, None], gid_t, :] = solref_t
    model.geom_solimp[env_ids_all[:, None], gid_t, : solimp_t.numel()] = solimp_t
  except (IndexError, RuntimeError, TypeError):
    model.geom_solref[gid_t, :] = solref_t
    model.geom_solimp[gid_t, : solimp_t.numel()] = solimp_t


def pm_soft_contact_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  geom_names: tuple[str, ...],
  sol_stages: list[SoftContactSolStage],
) -> dict[str, torch.Tensor]:
  """Stage knee/elbow collision solref/solimp (DEFAULT then SOFT_6mm).

  Active stage is the last entry with ``step <= env.common_step_counter``.
  """
  del env_ids
  active = sol_stages[0]
  for stage in sol_stages:
    if env.common_step_counter >= stage["step"]:
      active = stage
  profile = active["profile"]
  solref, solimp = _CONTACT_SOL_PROFILES[profile]

  asset: Entity = env.scene[asset_cfg.name]
  geom_ids, matched = asset.find_geoms(geom_names, preserve_order=True)
  if len(matched) != len(geom_names):
    missing = set(geom_names) - set(matched)
    raise ValueError(
      f"pm_soft_contact_curriculum: geom(s) not found on {asset_cfg.name!r}: "
      f"{sorted(missing)}"
    )
  _set_geom_sol_params(env, geom_ids, solref, solimp)

  return {
    "pm_soft_contact_profile": torch.tensor(
      0.0 if profile == "default" else 1.0, dtype=torch.float32
    ),
  }


def reset_push_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  push_stages: list[PushStage],
) -> dict[str, torch.Tensor]:
  """Update reset push strength by training stage."""
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  velocity_range = event_term_cfg.params["velocity_range"]

  active_stage = push_stages[0]
  for stage in push_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  for key in ("x", "y", "z", "roll", "pitch", "yaw"):
    stage_range = active_stage.get(key)
    if stage_range is not None:
      velocity_range[key] = stage_range

  return {
    "push_x_max": torch.tensor(abs(velocity_range["x"][1]), dtype=torch.float32),
    "push_y_max": torch.tensor(abs(velocity_range["y"][1]), dtype=torch.float32),
    "push_pitch_max": torch.tensor(abs(velocity_range["pitch"][1]), dtype=torch.float32),
    "push_roll_max": torch.tensor(abs(velocity_range["roll"][1]), dtype=torch.float32),
  }


def reset_initialization_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  init_stages: list[ResetInitStage],
) -> dict[str, torch.Tensor]:
  """Update reset initialization difficulty and data mixing by training stage."""
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  params = event_term_cfg.params

  active_stage = init_stages[0]
  for stage in init_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  params["data_probability"] = active_stage["data_probability"]
  if "tilt_pose_range" in active_stage:
    params["tilt_pose_range"] = dict(active_stage["tilt_pose_range"])
  if "tilt_velocity_range" in active_stage:
    params["tilt_velocity_range"] = dict(active_stage["tilt_velocity_range"])
  if "tilt_joint_position_range" in active_stage:
    params["tilt_joint_position_range"] = active_stage["tilt_joint_position_range"]
  if "tilt_joint_velocity_range" in active_stage:
    params["tilt_joint_velocity_range"] = active_stage["tilt_joint_velocity_range"]

  tilt_pose_range = params["tilt_pose_range"]
  return {
    "reset_data_probability": torch.tensor(
      params["data_probability"], dtype=torch.float32
    ),
    "reset_tilt_roll_max": torch.tensor(
      max(abs(v) for v in tilt_pose_range.get("roll", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "reset_tilt_pitch_max": torch.tensor(
      max(abs(v) for v in tilt_pose_range.get("pitch", (0.0, 0.0))),
      dtype=torch.float32,
    ),
  }


def reset_force_pulse_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  pulse_stages: list[ForcePulseStage],
) -> dict[str, torch.Tensor]:
  """Update reset force pulse magnitude/duration by training stage (discrete jumps)."""
  del env_ids
  event_term_cfg = env.event_manager.get_term_cfg(event_name)
  params = event_term_cfg.params

  active_stage = pulse_stages[0]
  for stage in pulse_stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage

  duration_steps_range = active_stage.get(
    "duration_steps_range", params["duration_steps_range"]
  )
  duration_low_raw, duration_high_raw = duration_steps_range
  duration_low = int(min(duration_low_raw, duration_high_raw))
  duration_high = int(max(duration_low_raw, duration_high_raw))
  sampled_duration = int(
    torch.randint(
      low=duration_low,
      high=duration_high + 1,
      size=(1,),
      device=env.device,
    ).item()
  )
  params["duration_steps"] = sampled_duration
  if "force_axis_range" in active_stage:
    params["force_axis_range"] = dict(active_stage["force_axis_range"])
  if "torque_axis_range" in active_stage:
    params["torque_axis_range"] = dict(active_stage["torque_axis_range"])

  force_axis_range = params["force_axis_range"]
  torque_axis_range = params["torque_axis_range"]
  return {
    "force_pulse_duration_steps": torch.tensor(
      params["duration_steps"], dtype=torch.float32
    ),
    "force_pulse_abs_fx_max": torch.tensor(
      max(abs(v) for v in force_axis_range.get("x", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_fy_max": torch.tensor(
      max(abs(v) for v in force_axis_range.get("y", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_fz_max": torch.tensor(
      max(abs(v) for v in force_axis_range.get("z", (0.0, 0.0))),
      dtype=torch.float32,
    ),
    "force_pulse_abs_pitch_torque_max": torch.tensor(
      max(abs(v) for v in torque_axis_range.get("pitch", (0.0, 0.0))),
      dtype=torch.float32,
    ),
  }


def task_reward_weight_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  stages: list[WeightStage],
) -> dict[str, torch.Tensor]:
  """Set task-reward mix scale by training step for AMP reward mixing."""
  del env_ids
  env_any = cast(Any, env)
  active_stage = stages[0]
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      active_stage = stage
  scale = float(active_stage["scale"])
  env_any.task_reward_weight_scale = scale
  return {
    "task_reward_weight_scale": torch.tensor(scale, dtype=torch.float32),
  }


def q25_effort_limit_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  actuator_indices: tuple[int, ...],
  effort_stages: list[Q25EffortLimitStage],
  action_term_name: str = "joint_pos",
  sync_action_scale: bool = True,
) -> dict[str, torch.Tensor]:
  """Apply staged Q25 torque limits and match joint position action scales.

  Active stage is the last entry with ``step <= env.common_step_counter`` (same as
  ``reset_force_pulse_curriculum``). Runs when curriculum is computed (typically on reset).

  For each stage, updates:

  - Warp ``env.sim.model.actuator_forcerange`` (sim clip).
  - CPU ``env.sim.mj_model.actuator_forcerange`` so rewards that read ``mj_model`` (e.g.
    ``motor_overcurrent_penalty``) see the same limits.
  - ``JointPositionAction._scale`` for Q25 joints:
    ``PM_ACTION_SCALE[j] * (effort / EFFORT_LIMIT_Q25)`` so policy command span matches the
    tighter torque bound (same form as initial ``0.25 * effort / stiffness``).
  """
  del env_ids
  active = effort_stages[0]
  for stage in effort_stages:
    if env.common_step_counter >= stage["step"]:
      active = stage
  effort = float(active["effort_limit"])
  ratio = effort / float(EFFORT_LIMIT_Q25)

  asset: Entity = env.scene[asset_cfg.name]
  env_ids_all = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  mj = env.sim.mj_model

  for ai in actuator_indices:
    actuator = asset.actuators[ai]
    ctrl_ids = actuator.ctrl_ids
    nctrl = int(ctrl_ids.numel())
    effort_mat = torch.full(
      (env.num_envs, nctrl),
      effort,
      device=env.device,
      dtype=torch.float32,
    )
    env.sim.model.actuator_forcerange[env_ids_all[:, None], ctrl_ids, 0] = -effort_mat
    env.sim.model.actuator_forcerange[env_ids_all[:, None], ctrl_ids, 1] = effort_mat

    for cid in ctrl_ids.detach().cpu().numpy().astype(np.int64).ravel():
      cid_i = int(cid)
      mj.actuator_forcerange[cid_i, 0] = -effort
      mj.actuator_forcerange[cid_i, 1] = effort

  if sync_action_scale:
    term = env.action_manager.get_term(action_term_name)
    if not isinstance(term, JointPositionAction):
      raise TypeError(
        f"q25_effort_limit_curriculum: action term {action_term_name!r} must be "
        f"JointPositionAction, got {type(term).__name__}"
      )
    if isinstance(term._scale, torch.Tensor):
      for ai in actuator_indices:
        for jname in asset.actuators[ai].joint_names:
          jidx = term._joint_names.index(jname)
          base = PM_ACTION_SCALE[jname]
          term._scale[:, jidx] = base * ratio
    else:
      raise TypeError(
        "q25_effort_limit_curriculum: expected per-joint scale tensor (dict scale in cfg); "
        f"got scalar scale on term {action_term_name!r}"
      )

  return {
    "q25_effort_limit": torch.tensor(effort, dtype=torch.float32),
    "q25_effort_ratio": torch.tensor(ratio, dtype=torch.float32),
  }
