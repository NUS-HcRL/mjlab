"""Export RSL-RL checkpoint (.pt) to ONNX for deployment.

Tracking tasks embed motion reference trajectories in the ONNX graph; velocity tasks
export policy-only ONNX. Use ``onnx_to_mnn`` / ``convert_onnx_to_mnn_batch`` for MNN.

Examples:
  python -m mjlab.scripts.export_pt_to_onnx Mjlab-Tracking-Flat-PM1 \\
    --checkpoint logs/rsl_rl/pm1_tracking/2026-06-04_15-21-12/model_17000.pt

  python -m mjlab.scripts.export_pt_to_onnx Mjlab-Tracking-Flat-PM1 \\
    --checkpoint logs/rsl_rl/pm1_tracking/2026-06-04_15-21-12/model_17000.pt \\
    --motion-file motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz \\
    --output-file logs/rsl_rl/pm1_tracking/2026-06-04_15-21-12/model_17000.onnx
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl.exporter import (
  attach_onnx_metadata as attach_tracking_onnx_metadata,
  export_motion_policy_as_onnx,
)
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata as attach_velocity_onnx_metadata,
  export_velocity_policy_as_onnx,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportPtToOnnxConfig:
  checkpoint: Path
  """Path to ``model_*.pt`` checkpoint."""
  motion_file: Path | None = None
  """Motion ``.npz`` for tracking tasks. If omitted, read from ``params/env.yaml`` next to checkpoint."""
  output_file: Path | None = None
  """Output ``.onnx`` path. Default: same directory as checkpoint, ``<stem>.onnx``."""
  run_path: str | None = None
  """Identifier stored in ONNX metadata (default: ``local/<run_dir>/<onnx_name>``)."""
  device: str = "cpu"
  attach_metadata: bool = True
  """Attach joint/body metadata to the ONNX file."""


def _motion_file_from_log_dir(log_dir: Path) -> Path | None:
  env_yaml = log_dir / "params" / "env.yaml"
  if not env_yaml.is_file():
    return None
  match = re.search(r"^\s*motion_file:\s*(.+)\s*$", env_yaml.read_text(), re.MULTILINE)
  if match is None:
    return None
  return Path(match.group(1).strip())


def _is_tracking_task(env_cfg) -> bool:
  return (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )


def export_pt_to_onnx(task_id: str, cfg: ExportPtToOnnxConfig) -> Path:
  configure_torch_backends()

  checkpoint = cfg.checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  log_dir = checkpoint.parent
  output_file = (
    cfg.output_file.resolve()
    if cfg.output_file is not None
    else log_dir / f"{checkpoint.stem}.onnx"
  )
  output_file.parent.mkdir(parents=True, exist_ok=True)

  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1

  is_tracking = _is_tracking_task(env_cfg)
  if is_tracking:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    motion_path = cfg.motion_file
    if motion_path is None:
      motion_path = _motion_file_from_log_dir(log_dir)
    if motion_path is None:
      raise ValueError(
        "Tracking export requires --motion-file or params/env.yaml with motion_file "
        f"under {log_dir}"
      )
    motion_path = motion_path.resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Motion file: {motion_cmd.motion_file}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id)
  if is_tracking or runner_cls is MotionTrackingOnPolicyRunner:
    runner = MotionTrackingOnPolicyRunner(
      env, asdict(agent_cfg), log_dir=str(log_dir), device=cfg.device
    )
  else:
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=cfg.device)

  print(f"[INFO] Loading checkpoint: {checkpoint}")
  runner.load(str(checkpoint), map_location=cfg.device)

  normalizer = (
    runner.alg.policy.actor_obs_normalizer
    if runner.alg.policy.actor_obs_normalization
    else None
  )

  onnx_dir = str(output_file.parent)
  onnx_name = output_file.name
  if is_tracking:
    export_motion_policy_as_onnx(
      env.unwrapped,
      runner.alg.policy,
      normalizer=normalizer,
      path=onnx_dir,
      filename=onnx_name,
    )
    if cfg.attach_metadata:
      run_path = cfg.run_path or f"local/{log_dir.name}/{onnx_name}"
      attach_tracking_onnx_metadata(env.unwrapped, run_path, onnx_dir, onnx_name)
  else:
    export_velocity_policy_as_onnx(
      runner.alg.policy,
      path=onnx_dir,
      normalizer=normalizer,
      filename=onnx_name,
    )
    if cfg.attach_metadata:
      run_path = cfg.run_path or f"local/{log_dir.name}/{onnx_name}"
      attach_velocity_onnx_metadata(env.unwrapped, run_path, onnx_dir, onnx_name)

  env.close()
  size_mb = output_file.stat().st_size / (1024 * 1024)
  print(f"[INFO] ONNX exported: {output_file} ({size_mb:.2f} MB)")
  return output_file


def main() -> None:
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  cfg = tyro.cli(
    ExportPtToOnnxConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  del remaining_args

  export_pt_to_onnx(chosen_task, cfg)


if __name__ == "__main__":
  main()
