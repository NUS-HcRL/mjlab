"""PM1 fall with occasional virtual ground regions to avoid on landing."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.fall.mdp.dodge import (
  DodgeFirstContactCost,
  DodgeLandingClearanceReward,
  DodgeRegionCommandCfg,
  dodge_region_contact_cost,
  dodge_region_observation,
)

from .env_cfgs import pm1_flat_falling_env_cfg
from .rl_cfg import pm1_falling_amp_runner_cfg


def pm1_flat_falling_dodge_env_cfg(
  play: bool = False,
  use_data_reset: bool = False,
  use_data_reset_obs_history: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Prioritize dodge while retaining soft impact costs and physics guards."""
  cfg = pm1_flat_falling_env_cfg(
    play=play,
    use_data_reset=use_data_reset,
    use_data_reset_obs_history=use_data_reset_obs_history,
  )
  assert cfg.amp is not None and isinstance(cfg.amp.motion_file, list)
  cfg.amp.motion_file.extend(
    (
      "motion_file/pm_fall4:v0/tofront_dodgeleft_v2.3_50fps.npz",
      "motion_file/pm_fall4:v0/tofront_dodgeright_v2.3_50fps.npz",
    )
  )
  cfg.commands = {"dodge": DodgeRegionCommandCfg(probability=0.5, at_reset=True)}
  if not play:
    cfg.amp.motion_file = [
      path for path in cfg.amp.motion_file
      if path.rsplit("/", 1)[-1] in (
        "Front_1_converted_50fps.npz",
        "tofront_dodgeleft_v2.3_50fps.npz",
        "tofront_dodgeright_v2.3_50fps.npz",
      )
    ]
    reset = cfg.events["reset_base"].params
    reset["stable_state_files"] = tuple(
      p for p in reset["stable_state_files"] if p.endswith("forward_walk.npy")
    )
    if not reset["stable_state_files"]:
      raise ValueError("Forward dodge requires forward_walk.npy reset states")
    reset["stable_standing_probability"] = 0.5
    reset["data_probability"] = 0.0
    reset["motion_files"] = ()
    reset["use_data_reset_obs_history"] = False
    reset["adaptive_replay_probability"] = 0.0
    # Do not allow the independent world-axis velocity disturbances to turn a
    # forward-only training episode into a lateral/backward push.
    cfg.events.pop("push_robot", None)
    pulse = cfg.events["push_force_pulse"].params
    pulse["forward_command_name"] = "dodge"
    pulse["data_direction_force_magnitude_range"] = None
    pulse["preserve_data_reset_states"] = False
  cfg.rewards["reduce_contact_force"].weight = 2.0
  cfg.rewards["action_rate_l2"].weight = -0.15
  cfg.terminations.pop("forbidden_body_contact_force", None)
  # This event reward would reference a termination that no longer exists.
  cfg.rewards.pop("forbidden_contact_termination", None)
  # The original fall factory and its safety settings remain unchanged.
  for group in ("policy", "critic"):
    cfg.observations[group].terms["dodge_region"] = ObservationTermCfg(
      func=dodge_region_observation,
      params={"command_name": "dodge"},
      history_length=3,
      flatten_history_dim=True,
    )

  # Keep the original ground force sensor untouched. This sensor includes feet
  # and retains multiple contact points per body to check spatial membership.
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name="dodge_ground_contact",
      primary=ContactMatch(mode="body", pattern=r"^LINK_.*$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "pos"),
      reduce="maxforce",
      num_slots=4,
    ),
  )
  cfg.rewards["dodge_region_contact"] = RewardTermCfg(
    func=dodge_region_contact_cost,
    weight=-1.0,
    params={"command_name": "dodge", "sensor_name": "dodge_ground_contact"},
  )
  cfg.rewards["dodge_region_first_contact"] = RewardTermCfg(
    func=DodgeFirstContactCost(),
    weight=-0.75,
  )
  cfg.rewards["dodge_landing_clearance"] = RewardTermCfg(
    func=DodgeLandingClearanceReward(
      body_names=(
        "LINK_TORSO_YAW",
        "LINK_ELBOW_PITCH_L",
        "LINK_ELBOW_END_L",
        "LINK_ELBOW_PITCH_R",
        "LINK_ELBOW_END_R",
        "LINK_KNEE_PITCH_L",
        "LINK_KNEE_PITCH_R",
      ),
      gamma=pm1_falling_amp_runner_cfg().algorithm.gamma,
      safety_margin=0.05,
      body_margin=0.04,
      temperature=0.04,
    ),
    weight=1.0,
  )
  return cfg


def pm1_falling_amp_dodge_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use fall AMP/PPO settings with a separate experiment directory."""
  cfg = pm1_falling_amp_runner_cfg()
  cfg.algorithm.reward_mix_scale_clip = (
    0.02, cfg.algorithm.reward_mix_scale_clip[1]
  )
  cfg.algorithm.kl_early_stop = False
  cfg.experiment_name = "pm1_falling_amp_dodge"
  # W&B sequential run names become dodge1, dodge2, ... (not mjlab1, ...).
  cfg.run_name = "dodge"
  return cfg
