"""PM1 fall with occasional virtual ground regions to avoid on landing."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.fall.mdp.dodge import (
  DodgeRegionCommandCfg,
  dodge_region_contact_cost,
  dodge_region_observation,
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
  cfg.commands = {"dodge": DodgeRegionCommandCfg()}
  for group in ("policy", "critic"):
    cfg.observations[group].terms["dodge_region"] = ObservationTermCfg(
      func=dodge_region_observation,
      params={"command_name": "dodge"},
      history_length=0,  # Current frame only; old proprioception histories unchanged.
    )

  # Keep the original ground force sensor untouched. This sensor includes feet
  # and retains multiple contact points per body to check spatial membership.
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name="dodge_ground_contact",
      primary=ContactMatch(mode="body", pattern=r"^LINK_.*$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force", "pos"),
      reduce="maxforce",
      num_slots=8,
    ),
  )
  cfg.rewards["dodge_region_contact"] = RewardTermCfg(
    func=dodge_region_contact_cost,
    weight=-1.0,
    params={"command_name": "dodge", "sensor_name": "dodge_ground_contact"},
  )
  return cfg


def pm1_falling_amp_dodge_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use unchanged fall AMP/PPO settings with a separate experiment directory."""
  cfg = pm1_falling_amp_runner_cfg()
  cfg.experiment_name = "pm1_falling_amp_dodge"
  return cfg
