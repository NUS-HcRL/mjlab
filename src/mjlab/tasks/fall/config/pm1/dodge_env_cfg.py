"""PM1 fall with occasional virtual ground regions to avoid on landing."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.fall.mdp.dodge import (
  DodgeRegionCommandCfg,
  dodge_region_contact_cost,
  dodge_region_observation,
  dodge_region_proximity_risk,
)

from .env_cfgs import pm1_flat_falling_env_cfg
from .rl_cfg import pm1_falling_amp_runner_cfg


def pm1_flat_falling_dodge_env_cfg(
  play: bool = False,
  use_data_reset: bool = True,
  use_data_reset_obs_history: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Extend the base fall configuration without changing any existing terms."""
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
  cfg.commands = {"dodge": DodgeRegionCommandCfg()}
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
  cfg.rewards["dodge_region_proximity"] = RewardTermCfg(
    func=dodge_region_proximity_risk,
    weight=-0.20,
    params={"command_name": "dodge", "asset_name": "robot"},
  )
  return cfg


def pm1_falling_amp_dodge_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use fall AMP/PPO settings with Dodge-specific update safeguards."""
  cfg = pm1_falling_amp_runner_cfg()
  cfg.algorithm.kl_early_stop = True
  cfg.algorithm.kl_early_stop_multiplier = 2.0
  cfg.experiment_name = "pm1_falling_amp_dodge"
  # W&B sequential run names become dodge1, dodge2, ... (not mjlab1, ...).
  cfg.run_name = "dodge"
  return cfg
