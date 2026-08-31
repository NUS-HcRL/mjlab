from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.fall.rl import FallOnPolicyRunner
from mjlab.rl.mj_amp_runner import MjlabAmpOnPolicyRunner

from .dodge_env_cfg import (
  pm1_falling_amp_dodge_runner_cfg,
  pm1_flat_falling_dodge_env_cfg,
)
from .env_cfgs import pm1_flat_falling_env_cfg
from .rl_cfg import (
  pm1_falling_amp_protective_finetune_runner_cfg,
  pm1_falling_amp_runner_cfg,
  pm1_falling_ppo_runner_cfg,
)

try:
  import importlib

  import rsl_rl.utils as _rsl_utils

  # Compatibility shim:
  # amp-rsl-rl expects `rsl_rl.utils.store_code_state` in some versions,
  # but newer rsl-rl may not expose it. Provide a safe no-op fallback.
  if not hasattr(_rsl_utils, "store_code_state"):
    def _store_code_state_noop(*_args, **_kwargs):
      return []

    setattr(_rsl_utils, "store_code_state", _store_code_state_noop)
except ImportError:
  pass

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1",
  env_cfg=pm1_flat_falling_env_cfg(),
  play_env_cfg=pm1_flat_falling_env_cfg(play=True),
  rl_cfg=pm1_falling_ppo_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1-No-State-Estimation",
  env_cfg=pm1_flat_falling_env_cfg(has_state_estimation=False),
  play_env_cfg=pm1_flat_falling_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pm1_falling_ppo_runner_cfg(),
  runner_cls=FallOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1-AMP",
  env_cfg=pm1_flat_falling_env_cfg(),
  play_env_cfg=pm1_flat_falling_env_cfg(play=True),
  rl_cfg=pm1_falling_amp_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1-AMP-Protective-Finetune",
  env_cfg=pm1_flat_falling_env_cfg(protective_finetune=True),
  play_env_cfg=pm1_flat_falling_env_cfg(play=True, protective_finetune=True),
  rl_cfg=pm1_falling_amp_protective_finetune_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Falling-Flat-PM1-AMP-Dodge",
  env_cfg=pm1_flat_falling_dodge_env_cfg(),
  play_env_cfg=pm1_flat_falling_dodge_env_cfg(play=True),
  rl_cfg=pm1_falling_amp_dodge_runner_cfg(),
  runner_cls=MjlabAmpOnPolicyRunner,
)
