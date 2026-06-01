"""PM1 flat fall environment configurations."""

from pathlib import Path

from mjlab.asset_zoo.robots import (
  PM_ACTION_SCALE,
  PM_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.fall.fall_env_cfg import make_fall_env_cfg


def _pm1_fall_reset_motion_csv_paths() -> tuple[str, ...]:
  """Paths to every ``*.csv`` under repo ``data/amp_pm1_fall/`` (sorted by name).

  Rows from all files are concatenated into one reset pool (see fall ``mdp.events``).
  """
  # env_cfgs.py -> pm1 -> config -> fall -> tasks -> mjlab -> src -> repo root
  repo_root = Path(__file__).resolve().parents[6]
  d = repo_root / "data" / "amp_pm1_fall"
  if not d.is_dir():
    return ()
  return tuple(str(p) for p in sorted(d.glob("*.csv")))


def pm1_flat_falling_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  use_data_reset: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create PM1 flat terrain fall (joint-state tracking) configuration.

  has_state_estimation: Kept for API compatibility with tracking; fall policy
    does not use base_lin_vel or motion anchor, so this has no effect.
  """
  del has_state_estimation  # Unused for fall; policy has no motion anchor / base_lin_vel
  cfg = make_fall_env_cfg()

  cfg.scene.entities = {"robot": PM_ROBOT_CFG}

  # Self-collision detection for PM1
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  body_contact_force_cfg = ContactSensorCfg(
    name="body_contact_force",
    primary=ContactMatch(
      mode="body",
      pattern=r"^LINK_.*$",
      entity="robot",
      exclude=(
        "LINK_ANKLE_PITCH_L",
        "LINK_ANKLE_PITCH_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force", "found"),
    reduce="maxforce",
    num_slots=1,
    track_air_time=True,
  )

  cfg.scene.sensors = (self_collision_cfg, body_contact_force_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM_ACTION_SCALE

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^collision_(left|right)_foot(_toe)?$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_TORSO_YAW",)
  cfg.events["push_force_pulse"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    body_names=("LINK_TORSO_YAW",),
  )

  cfg.terminations["forbidden_body_contact_force"].params["body_names"] = (
    "LINK_HEAD_YAW",
    "LINK_TORSO_YAW",
    "LINK_ELBOW_END_L",
    "LINK_ELBOW_END_R",
  )
  cfg.terminations["forbidden_body_contact_force"].params["body_force_thresholds"] = {
    "LINK_HEAD_YAW": 500.0,
    "LINK_TORSO_YAW": 500.0,
    "LINK_ELBOW_END_L": 500.0,
    "LINK_ELBOW_END_R": 500.0,
  }

  # PM1 LINK_BASE 在 MJCF 中 pos="0 0 0.82"，站立时 base 相对地面约 0.82 m
  # if "base_height" in cfg.rewards:
  #   cfg.rewards["base_height"].params["nominal_height"] = 0.82

  cfg.viewer.body_name = "LINK_TORSO_YAW"
  cfg.events["reset_base"].params["motion_files"] = (
    _pm1_fall_reset_motion_csv_paths() if use_data_reset else ()
  )
  # cfg.events["reset_base"].params["motion_files"] = ("data/amp_pm1_fall/policy_switch_walking_combined.csv",)
  cfg.events["reset_base"].params["data_root_body_name"] = "LINK_BASE"
  if not use_data_reset and cfg.curriculum is not None and "reset_init" in cfg.curriculum:
    init_stages = cfg.curriculum["reset_init"].params["init_stages"]
    for stage in init_stages:
      stage["data_probability"] = 0.0

  # AMP: expert ``.npz`` only (do not mix with reset CSV pool above).
  if cfg.amp is not None:
    cfg.amp.motion_file = [
      "motion_file/pm_fall4:v0/Back_3_converted.npz",
      "motion_file/pm_fall4:v0/Front_1_converted_50fps.npz",
      "motion_file/pm_fall4:v0/Left_1_converted_50fps.npz",
      "motion_file/pm_fall4:v0/Right_1_converted_50fps.npz",
      "motion_file/pm_fall4:v0/LeftFront_1_converted_50fps.npz",
      "motion_file/pm_fall4:v0/LeftBack_2_converted.npz",
      "motion_file/pm_fall4:v0/RightFront_1_converted_50fps.npz",
      "motion_file/pm_fall4:v0/RightBack_2_converted.npz",
    ]

  # PM1 IMU 传感器名与 G1 不同：imu_angular_velocity / imu_link_linear_velocity
  for group in ("policy", "critic"):
    if "base_ang_vel" in cfg.observations[group].terms:
      cfg.observations[group].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"
    if "base_lin_vel" in cfg.observations[group].terms:
      cfg.observations[group].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_force_pulse", None)
    if cfg.curriculum is not None:
      cfg.curriculum.pop("reset_init", None)
      cfg.curriculum.pop("reset_push", None)
      cfg.curriculum.pop("reset_force_pulse", None)
      cfg.curriculum.pop("q25_effort_limit", None)
      cfg.curriculum.pop("pm_soft_contact", None)
    if "push_at_reset" in cfg.events:
      # In play mode, use a deterministic forward push so resets are reproducible.
      cfg.events["push_at_reset"].params["velocity_range"] = {
        "x": (1.0, 1.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      }
    cfg.events["reset_base"].params["data_probability"] = 0.0

  return cfg
