"""Fall task configuration.

This module provides a factory function to create a base fall task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.asset_zoo.robots.engineai_pm01.pm01_8 import (
  EFFORT_LIMIT_Q25,
  PM_Q25_ACTUATOR_INDICES,
  PM_SOFT_CONTACT_CURRICULUM_GEOM_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  CurriculumTermCfg,
  ActionTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs.amp import AMPCfg
from mjlab.tasks.fall import mdp
from mjlab.tasks.fall.mdp.curriculums import (
  pm_soft_contact_curriculum,
  q25_effort_limit_curriculum,
  reset_force_pulse_curriculum,
  reset_initialization_curriculum,
  reset_push_curriculum,
  task_reward_weight_curriculum,
)
from mjlab.tasks.fall.mdp.events import (
  apply_external_force_torque_axiswise_pulse,
  push_by_setting_velocity_preserve_data,
  randomize_gravity,
)
from mjlab.tasks.fall.mdp.terminations import nonfinite_state
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_fall_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base fall task configuration."""

  ##
  # Observations
  ##

  # Actor (policy): deployable sensor measurements only.
  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      history_length=5,
      flatten_history_dim=True,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # Projected gravity g_b in pelvis frame, encoding roll / pitch.
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      history_length=5,
      flatten_history_dim=True,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Joint positions q relative to defaults.
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      history_length=5,
      flatten_history_dim=True,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Joint velocities q̇ relative to defaults.
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      history_length=5,
      flatten_history_dim=True,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # Previous action a_{t-1}.
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
      flatten_history_dim=True,
      ),
  }

  # Critic: same as actor plus privileged base (shorter history for speed).
  critic_terms = {
    **policy_terms,
    "base_pos": ObservationTermCfg(
      func=mdp.base_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=5,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      history_length=5,
      flatten_history_dim=True,
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_mixed,
      mode="reset",
      params={
        "tilt_pose_range": {
          "x": (-1.2, 1.2),
          "y": (-1.2, 1.2),
          "z": (0, 0.08),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-3.14, 3.14),
        },
        "tilt_velocity_range": {},
        "tilt_joint_position_range": (-0.25, 0.25),
        "tilt_joint_velocity_range": (-0.1, 0.1),
        "data_probability": 0.15,
        "motion_files": (),
        "data_root_body_name": "LINK_BASE",
        "data_pose_range": {
          "x": (-1, 1),
          "y": (-1, 1),
          "z": (-0.03, 0.3),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-3.14, 3.14),
        },
        "data_velocity_range": {
          "x": (-1, 1),
          "y": (-1, 1),
          "z": (-0.4, 0.4),
          "roll": (-0.6, 0.6),
          "pitch": (-0.6, 0.6),
          "yaw": (-0.8, 0.8),
        },
        "data_joint_position_range": (-0.25, 0.25),
        "data_joint_velocity_range": (-0.2, 0.2),
      },
    ),
    # Apply an extra reset push only to non-data initializations so motion-derived
    # root velocities remain unchanged.
    "push_at_reset": EventTermCfg(
      func=push_by_setting_velocity_preserve_data,
      mode="reset",
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
          "z": (-0.3, 0.3),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
        "preserve_data_reset_states": True,
      },
    ),
    # Single pulse event: detects just-reset envs, applies force pulse, and
    # decrements/clears pulses every step.
    "push_force_pulse": EventTermCfg(
      func=apply_external_force_torque_axiswise_pulse,
      mode="interval",
      interval_range_s=(0.0, 0.0),
      params={
        # World-frame external force range per axis.
        "force_axis_range": {
          # "x": (-200.0, 200.0),
          # "y": (-200.0, 200.0),
          # "z": (-30.0, -30.0),
        },
        # World-frame external torque range per axis.
        "torque_axis_range": {
          # "roll": (-12.0, 12.0),
          # "pitch": (-20.0, -5.0),
          # "yaw": (-10.0, 10.0),
        },
        "duration_steps_range": (0, 1),
        # Enforce cooldown to avoid too many consecutive high-impact episodes.
        "cooldown_steps": 200,
        "preserve_data_reset_states": True,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 2.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.2, 0.2),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.4, 0.4),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "randomize_gravity": EventTermCfg(
      mode="startup",
      func=randomize_gravity,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "nominal_gravity": (0.0, 0.0, -9.81),
        "roll_range": (-0.08, 0.08),
        "pitch_range": (-0.08, 0.08),
        "magnitude_range": (0.92, 1.08),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
    "reduce_contact_force": RewardTermCfg(
      func=mdp.ReduceContactForceWeighted(
        sensor_name="body_contact_force",
        high_weight_bodies=(
          "LINK_ELBOW_END_L",
          "LINK_ELBOW_END_R",
          "LINK_HEAD_YAW",
          "LINK_TORSO_YAW",
        ),
        medium_weight_bodies=(
          "LINK_SHOULDER_ROLL_L",
          "LINK_SHOULDER_ROLL_R",
          "LINK_SHOULDER_YAW_L",
          "LINK_SHOULDER_YAW_R",
        ),
        high_weight=100.0,
        medium_weight=60.0,
        low_weight=0.5,
        alpha=0.5,
        squash_scale=0.02,
        tracked_body_names=(
          "LINK_HEAD_YAW",
          "LINK_TORSO_YAW",
          "LINK_ELBOW_END_L",
          "LINK_ELBOW_END_R",
          "LINK_SHOULDER_ROLL_L",
          "LINK_SHOULDER_ROLL_R",
        ),
      ),
      weight=0.015, # 0.01
    ),
    "control_descent_speed": RewardTermCfg(
      func=mdp.control_descent_speed,
      weight=1,
      params={
        "torso_body_name": "LINK_TORSO_YAW",
        "threshold": 0.5,
      },
    ),
    "impact_velocity_reward": RewardTermCfg(
      func=mdp.ImpactVelocityReward(
        sensor_name="body_contact_force",
        high_weight_bodies=(
          "LINK_ELBOW_END_L",
          "LINK_ELBOW_END_R",
          "LINK_HEAD_YAW",
          "LINK_TORSO_YAW",
        ),
        medium_weight_bodies=(
          "LINK_SHOULDER_ROLL_L",
          "LINK_SHOULDER_ROLL_R",
          "LINK_SHOULDER_YAW_L",
          "LINK_SHOULDER_YAW_R",
        ),
        high_weight=20.0,
        medium_weight=10.0,
        low_weight=0.5,
        squash_scale=0.02,
      ),
      weight=1,
    ),
    "lower_then_upper_contact": RewardTermCfg(
      func=mdp.LowerBodyThenUpperBodyContactReward(
        sensor_name="body_contact_force",
        lower_body_names=(
          "LINK_BASE",
          "LINK_HIP_PITCH_L",
          "LINK_HIP_PITCH_R",
          "LINK_HIP_ROLL_L",
          "LINK_HIP_ROLL_R",
          "LINK_HIP_YAW_L",
          "LINK_HIP_YAW_R",
          "LINK_KNEE_PITCH_L",
          "LINK_KNEE_PITCH_R",
        ),
        upper_body_names=(
          "LINK_TORSO_YAW",
          "LINK_SHOULDER_ROLL_L",
          "LINK_SHOULDER_ROLL_R",
          "LINK_SHOULDER_YAW_L",
          "LINK_SHOULDER_YAW_R",
          "LINK_ELBOW_PITCH_L",
          "LINK_ELBOW_PITCH_R",
          "LINK_ELBOW_YAW_L",
          "LINK_ELBOW_YAW_R",
          "LINK_ELBOW_END_L",
          "LINK_ELBOW_END_R",
        ),
        min_delay_s=0.2,
        max_delay_s=0.6,
        lower_first_bonus=0.5,
        timely_upper_bonus=1.0,
        early_upper_penalty=4.0,
        late_upper_penalty=0.2,
        early_upper_force_scale=0.0,
      ),
      weight=0.05,
    ),
    "motor_overcurrent": RewardTermCfg(
      func=mdp.motor_overcurrent_penalty,
      weight=1e-3,
      params={
        "command_name": "motion",
        "scale": 1.0,
        "threshold": 1.0,
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nonfinite_state": TerminationTermCfg(
      func=nonfinite_state,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "forbidden_body_contact_force": TerminationTermCfg(
      func=mdp.BadBodyContactForce(),
      params={
        "sensor_name": "body_contact_force",
        "body_names": (),  # Set per-robot.
        "body_force_thresholds": {},  # Set per-robot.
      },
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    # "task_reward_weight": CurriculumTermCfg(
    #   func=task_reward_weight_curriculum,
    #   params={
    #     "stages": [
    #       {"step": 0, "scale": 1.0},
    #       {"step": 30_000 * 32, "scale": 1.5},
    #     ],
    #   },
    # ),
    "reset_init": CurriculumTermCfg(
      func=reset_initialization_curriculum,
      params={
        "event_name": "reset_base",
        "init_stages": [
          {
            "step": 0,
            "data_probability": 0.05,
            "tilt_pose_range": {
              "x": (-0.4, 0.4),
              "y": (-0.4, 0.4),
              "z": (0.00, 0.06),
              "roll": (-0.1, 0.1),
              "pitch": (-0.1, 0.1),
              "yaw": (-3.14, 3.14),
            },
            "tilt_velocity_range": {},
            "tilt_joint_position_range": (-0.1, 0.1),
            "tilt_joint_velocity_range": (-0.03, 0.03),
          },
          {
            "step": 6_000 * 32,
            "data_probability": 0.15,
            "tilt_pose_range": {
              "x": (-0.6, 0.6),
              "y": (-0.6, 0.6),
              "z": (0.00, 0.1),
              "roll": (-0.15, 0.15),
              "pitch": (-0.15, 0.15),
              "yaw": (-3.14, 3.14),
            },
            "tilt_velocity_range": {},
            "tilt_joint_position_range": (-0.15, 0.15),
            "tilt_joint_velocity_range": (-0.05, 0.05),
          },
          {
            "step": 15_000 * 32,
            "data_probability": 0.25,
            "tilt_pose_range": {
              "x": (-1, 1),
              "y": (-1, 1),
              "z": (0.00, 0.1),
              "roll": (-0.2, 0.2),
              "pitch": (-0.2, 0.2),
              "yaw": (-3.14, 3.14),
            },
            "tilt_velocity_range": {},
            "tilt_joint_position_range": (-0.25, 0.25),
            "tilt_joint_velocity_range": (-0.08, 0.08),
          },
        ],
      },
    ),
    # "reset_push": CurriculumTermCfg(
    #   func=reset_push_curriculum,
    #   params={
    #     "event_name": "push_at_reset",
    #     # env.common_step_counter counts env steps, not iterations.
    #     "push_stages": [
    #       {
    #         "step": 0,
    #         "x": (-1.0, 1.0),
    #         "y": (-1.0, 1.0),
    #         "z": (-0.1, 0.1),
    #         "roll": (-0.2, 0.2),
    #         "pitch": (-0.2, 0.2),
    #         "yaw": (-0.3, 0.3),
    #       },
    #       {
    #         "step": 8_000 * 32,
    #         "x": (-3.0, 3.0),
    #         "y": (-3.0, 3.0),
    #         "z": (-0.15, 0.15),
    #         "roll": (-0.5, 0.5),
    #         "pitch": (-0.5, 0.5),
    #         "yaw": (-0.4, 0.4),
    #       },
    #       {
    #         "step": 15000 * 32,
    #         "x": (-5.0, 5.0),
    #         "y": (-5.0, 5.0),
    #         "z": (-0.2, 0.2),
    #         "roll": (-1, 1),
    #         "pitch": (-1, 1),
    #         "yaw": (-0.5, 0.5),
    #       },
    #     ],
    #   },
    # ),
    "reset_force_pulse": CurriculumTermCfg(
      func=reset_force_pulse_curriculum,
      params={
        "event_name": "push_force_pulse",
        "pulse_stages": [
          {
            "step": 0,
            "duration_steps_range": (0, 3),
            "force_axis_range": {
              "x": (-30.0, 30.0),
              "y": (-30.0, 30.0),
              "z": (-10.0, -10.0),
            },
          },
          {
            "step": 4_000 * 32,
            "duration_steps_range": (2, 8),
            "force_axis_range": {
              "x": (-80.0, 80.0),
              "y": (-80.0, 80.0),
              "z": (-10.0, -10.0),
            },
          },
          {
            "step": 10_000 * 32,
            "duration_steps_range": (4, 15),
            "force_axis_range": {
              "x": (-180.0, 180.0),
              "y": (-180.0, 180.0),
              "z": (-10.0, -10.0),
            },
          },
          {
            "step": 15_000 * 32,
            "duration_steps_range": (5, 20),
            "force_axis_range": {
              "x": (-260.0, 260.0),
              "y": (-260.0, 260.0),
              "z": (-50.0, -50.0),
            },
          },
        ],
      },
    ),
    "q25_effort_limit": CurriculumTermCfg(
      func=q25_effort_limit_curriculum,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "actuator_indices": PM_Q25_ACTUATOR_INDICES,
        "effort_stages": [
          {"step": 0, "effort_limit": float(EFFORT_LIMIT_Q25)},
          {"step": 30_000 * 32, "effort_limit": float(EFFORT_LIMIT_Q25) * 0.7},
        ],
      },
    ),
    "pm_soft_contact": CurriculumTermCfg(
      func=pm_soft_contact_curriculum,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "geom_names": PM_SOFT_CONTACT_CURRICULUM_GEOM_NAMES,
        "sol_stages": [
          {"step": 0, "profile": "default"},
          {"step": 40_000 * 32, "profile": "soft_6mm"},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="plane",
        terrain_generator=None,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=None,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1600,
      contact_sensor_maxmatch=200,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=6,
        ls_iterations=12,
        ccd_iterations=200,
      ),
    ),
    decimation=4,
    episode_length_s=5.0,
    amp=AMPCfg(
      # Keep root z for fall-state awareness, but drop root x/y and root 6D
      # orientation so the discriminator cannot separate expert/policy too
      # easily using obvious global pose shortcuts.
      num_disc_obs_steps=2,  # 52-dim per step with current settings
      asset_name="robot",
      root_body_name="LINK_BASE",
      motion_file=None,
      global_obs=False,
      root_height_obs=True,
      include_root_xy=False,
      include_root_rot=False,
      include_root_vel=True,
      include_projected_gravity=False,
      disc_body_pos_b_link_names=(
        # "LINK_ANKLE_ROLL_L",
        # "LINK_ANKLE_ROLL_R",
        # "LINK_SHOULDER_ROLL_L",
        # "LINK_SHOULDER_ROLL_R",
      ),
    ),
  )
