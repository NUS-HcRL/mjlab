from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.fall.mdp.events import (
  _get_motion_reset_pool,
  _normalize_quat,
  _root_state_is_placeholder,
  reset_root_state_mixed,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import BaseViewer, NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class VisualizeDataResetConfig:
  task: str = "Mjlab-Falling-Flat-PM1-AMP"
  device: str | None = None
  seed: int = 0
  viewer: Literal["native", "viser"] = "native"
  keep_push: bool = False
  freeze_first_frame: bool = True
  apply_reset_base: bool = True


def _validate_task(task: str) -> None:
  if task not in list_tasks():
    raise ValueError(f"Unknown task '{task}'.")


def _sample_one_data_reset_state(env: ManagerBasedRlEnv, keep_push: bool) -> None:
  reset_term = env.cfg.events.get("reset_base")
  if reset_term is None:
    raise RuntimeError("Environment has no reset_base event.")
  params = dict(reset_term.params)

  params["data_probability"] = 1.0
  params["tilt_pose_range"] = {}
  params["tilt_velocity_range"] = {}
  params["tilt_joint_position_range"] = (0.0, 0.0)
  params["tilt_joint_velocity_range"] = (0.0, 0.0)

  motion_files = params.get("motion_files", ())
  if not motion_files:
    raise RuntimeError(
      "reset_base.motion_files is empty. Please set fall data csv paths first."
    )

  env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)
  reset_root_state_mixed(env=env, env_ids=env_ids, **params)

  if not keep_push and "push_at_reset" in env.cfg.events:
    push_term = env.cfg.events["push_at_reset"]
    velocity_range = push_term.params.get("velocity_range", {})
    zero_velocity_range = {k: (0.0, 0.0) for k in velocity_range.keys()}
    push_term.params["velocity_range"] = zero_velocity_range

  env.scene.write_data_to_sim()
  env.sim.forward()
  _print_link_base_omega_xy(env)


def _print_link_base_omega_xy(env: ManagerBasedRlEnv) -> None:
  asset = env.scene["robot"]
  omega_xy = torch.linalg.vector_norm(
    asset.data.root_link_ang_vel_w[0, :2], dim=0
  ).item()
  wx, wy = asset.data.root_link_ang_vel_w[0, 0].item(), asset.data.root_link_ang_vel_w[
    0, 1
  ].item()
  print(
    f"[data-reset] LINK_BASE omega_xy={omega_xy:.6f} (wx={wx:.6f}, wy={wy:.6f})"
  )


def _sample_one_raw_data_state(env: ManagerBasedRlEnv) -> None:
  reset_term = env.cfg.events.get("reset_base")
  if reset_term is None:
    raise RuntimeError("Environment has no reset_base event.")
  params = dict(reset_term.params)
  motion_files = params.get("motion_files", ())
  if not motion_files:
    raise RuntimeError(
      "reset_base.motion_files is empty. Please set fall data csv paths first."
    )

  asset = env.scene["robot"]
  root_ids, _ = asset.find_bodies((params["data_root_body_name"],), preserve_order=True)
  if not root_ids:
    raise RuntimeError(
      f"Could not find data root body '{params['data_root_body_name']}' in asset 'robot'."
    )
  root_body_idx = root_ids[0]
  motion_pool = _get_motion_reset_pool(
    motion_files=motion_files,
    root_body_idx=root_body_idx,
    device=env.device,
    expected_num_joints=asset.num_joints,
    min_root_height=params.get("data_min_root_height"),
    max_abs_joint_velocity=params.get("data_max_abs_joint_velocity"),
  )
  num_states = motion_pool["root_state"].shape[0]
  if num_states == 0:
    raise RuntimeError("Motion files contain no states.")

  state_id = torch.randint(low=0, high=num_states, size=(1,), device=env.device)
  env_ids = torch.tensor([0], dtype=torch.int64, device=env.device)

  if _root_state_is_placeholder(motion_pool["root_state"]):
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_state = default_root_state[env_ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
  else:
    root_state = motion_pool["root_state"][state_id].clone()
    root_state[:, 0:3] += env.scene.env_origins[env_ids]
    root_state[:, 3:7] = _normalize_quat(root_state[:, 3:7])

  joint_pos = motion_pool["joint_pos"][state_id].clone()
  joint_vel = motion_pool["joint_vel"][state_id].clone()
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  joint_limits = soft_joint_pos_limits[env_ids]
  joint_pos = joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
  asset.clear_state(env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  _print_link_base_omega_xy(env)


def _install_frozen_data_reset(
  viewer: BaseViewer,
  env: ManagerBasedRlEnv,
  keep_push: bool,
  apply_reset_base: bool,
) -> None:
  def _reset_environment(self: BaseViewer) -> None:
    env.reset()
    if apply_reset_base:
      _sample_one_data_reset_state(env, keep_push=keep_push)
    else:
      _sample_one_raw_data_state(env)
    self._step_count = 0
    self._timer.tick()
    self.pause()

  viewer.reset_environment = MethodType(_reset_environment, viewer)
  viewer.pause()


def run_visualize_data_reset(cfg: VisualizeDataResetConfig) -> None:
  configure_torch_backends()
  _validate_task(cfg.task)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task, play=False)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  _obs, _extras = env.reset(seed=cfg.seed)
  del _obs, _extras

  if cfg.apply_reset_base:
    _sample_one_data_reset_state(env, keep_push=cfg.keep_push)
  else:
    _sample_one_raw_data_state(env)

  agent_cfg = load_rl_cfg(cfg.task)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  action_shape = vec_env.unwrapped.action_space.shape  # type: ignore[attr-defined]

  class ZeroPolicy:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(action_shape, device=vec_env.unwrapped.device)

  policy = ZeroPolicy()
  if cfg.viewer == "native":
    viewer = NativeMujocoViewer(vec_env, policy)
  else:
    viewer = ViserPlayViewer(vec_env, policy)
  if cfg.freeze_first_frame:
    _install_frozen_data_reset(
      viewer,
      env,
      keep_push=cfg.keep_push,
      apply_reset_base=cfg.apply_reset_base,
    )
  viewer.run()
  env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(
    VisualizeDataResetConfig,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_visualize_data_reset(cfg)


if __name__ == "__main__":
  main()
