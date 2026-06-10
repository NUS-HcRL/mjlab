"""Script to train RL agent with RSL-RL."""

import csv
import inspect
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


def _extract_sorted_suffix_indices(fieldnames: Sequence[str], prefix: str) -> list[int]:
  indices: list[int] = []
  for name in fieldnames:
    if not name.startswith(prefix):
      continue
    suffix = name[len(prefix) :]
    if suffix.isdigit():
      indices.append(int(suffix))
  return sorted(indices)


def _default_csv_root_body_idx(fieldnames: Sequence[str]) -> int:
  """CSV body ids follow MuJoCo nbody and usually include world at index 0."""
  if f"body_pos_w_1_z" in fieldnames and f"body_quat_w_1_w" in fieldnames:
    return 1
  return 0


def _csv_to_amp_npy_dict(csv_path: str | Path, fps_default: float = 50.0) -> dict:
  """Convert flattened csv motion data to AMPLoader .npy dict.

  AMPLoader expects: joints_list, joint_positions (list of per-frame arrays),
  root_position (list of (3,)), root_quaternion (list of (4,) in xyzw), fps.
  """
  path = Path(csv_path)
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames or []
    rows = list(reader)

  if not rows:
    raise ValueError(f"AMP motion file is empty: {path}")

  joint_indices = _extract_sorted_suffix_indices(fieldnames, "joint_pos_")
  if not joint_indices or joint_indices != list(range(len(joint_indices))):
    raise ValueError(
      f"CSV motion file {path} must provide contiguous joint_pos_i columns starting at 0."
    )
  joint_vel_indices = _extract_sorted_suffix_indices(fieldnames, "joint_vel_")
  if joint_vel_indices != joint_indices:
    raise ValueError(
      f"CSV motion file {path} must provide matching joint_pos_i / joint_vel_i columns."
    )

  root_body_idx = _default_csv_root_body_idx(fieldnames)
  required_root_cols = [
    f"body_pos_w_{root_body_idx}_{axis}" for axis in ("x", "y", "z")
  ] + [f"body_quat_w_{root_body_idx}_{axis}" for axis in ("w", "x", "y", "z")]
  missing_root_cols = [col for col in required_root_cols if col not in fieldnames]
  if missing_root_cols:
    raise ValueError(
      f"CSV motion file {path} is missing root body columns for body index 0: "
      f"{', '.join(missing_root_cols[:4])}"
      + ("..." if len(missing_root_cols) > 4 else "")
    )

  joint_pos_cols = [f"joint_pos_{joint_idx}" for joint_idx in joint_indices]
  joint_positions = [
    np.array([float(row[col]) for col in joint_pos_cols], dtype=np.float64)
    for row in rows
  ]
  root_position = [
    np.array(
      [float(row[f"body_pos_w_{root_body_idx}_{axis}"]) for axis in ("x", "y", "z")],
      dtype=np.float64,
    )
    for row in rows
  ]
  # CSV stores quaternions as wxyz; AMPLoader expects xyzw.
  root_quaternion = [
    np.array(
      [
        float(row[f"body_quat_w_{root_body_idx}_x"]),
        float(row[f"body_quat_w_{root_body_idx}_y"]),
        float(row[f"body_quat_w_{root_body_idx}_z"]),
        float(row[f"body_quat_w_{root_body_idx}_w"]),
      ],
      dtype=np.float64,
    )
    for row in rows
  ]
  fps = float(rows[0].get("fps", fps_default) or fps_default)
  return {
    "joints_list": [f"joint_{i}" for i in joint_indices],
    "joint_positions": joint_positions,
    "root_position": root_position,
    "root_quaternion": root_quaternion,
    "fps": fps,
  }


def _build_amp_dataset_from_csv(
  motion_files: str | list[str], cwd: Path
) -> tuple[str, dict[str, float]]:
  """Convert mjlab csv motion files to AMPLoader .npy in a temp dir."""
  paths = [motion_files] if isinstance(motion_files, str) else list(motion_files)
  if not paths:
    raise ValueError("amp.motion_file is empty")
  out_dir = Path(tempfile.mkdtemp(prefix="mjlab_amp_csv_"))
  dataset_names = {}
  for i, p in enumerate(paths):
    resolved = (cwd / p).resolve() if not Path(p).is_absolute() else Path(p)
    if not resolved.exists():
      raise FileNotFoundError(f"AMP motion file not found: {resolved}")
    npy_dict = _csv_to_amp_npy_dict(resolved)
    name = f"motion_{i}"
    out_path = out_dir / f"{name}.npy"
    np.save(out_path, cast(Any, npy_dict), allow_pickle=True)
    dataset_names[name] = 1.0
  return str(out_dir), dataset_names


def _ensure_amp_wandb_compat() -> None:
  """Monkey-patch amp_rsl_rl bits to work with current rsl_rl version."""
  try:
    from amp_rsl_rl.utils import wandb_utils as _amp_wandb  # type: ignore[attr-defined]
  except Exception:
    return

  cls = getattr(_amp_wandb, "WandbSummaryWriter", None)
  if cls is None:
    return

  if not hasattr(cls, "log_config"):
    def log_config(self, env_cfg, train_cfg, alg_cfg, policy_cfg):  # type: ignore[no-untyped-def]
      # Older amp_rsl_rl versions do not define log_config; for compatibility
      # we make it a no-op so training can proceed.
      _ = (env_cfg, train_cfg, alg_cfg, policy_cfg)

    setattr(cls, "log_config", log_config)

  # When using our AMP cfg proxy, rsl_rl's store_config(env_cfg, ...) calls asdict(env_cfg)
  # and fails. Unwrap env_cfg to the real dataclass before any asdict().
  try:
    from rsl_rl.utils import wandb_utils as _rsl_wandb  # type: ignore[import-not-found]
  except Exception:
    pass
  else:
    _RslWriter = getattr(_rsl_wandb, "WandbSummaryWriter", None)
    if _RslWriter is not None and hasattr(_RslWriter, "store_config"):
      _orig_store = _RslWriter.store_config

      def _store_config_unwrap(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):  # type: ignore[no-untyped-def]
        env_cfg = getattr(env_cfg, "_real_cfg", env_cfg)
        return _orig_store(self, env_cfg, runner_cfg, alg_cfg, policy_cfg)

      setattr(_RslWriter, "store_config", _store_config_unwrap)

  # Storage API shim: AMP_PPO expects RolloutStorage.add_transitions, but
  # newer rsl_rl exposes add_transition only. It also expects
  # RolloutStorage.compute_returns(last_values, gamma, lam) while newer
  # rsl_rl moves this logic into PPO.compute_returns().
  try:
    from rsl_rl.storage.rollout_storage import RolloutStorage  # type: ignore[import-not-found]
  except Exception:
    return

  if not hasattr(RolloutStorage, "add_transitions"):
    def add_transitions(self, transition):  # type: ignore[no-untyped-def]
      return self.add_transition(transition)
    setattr(RolloutStorage, "add_transitions", add_transitions)

  if not hasattr(RolloutStorage, "compute_returns"):
    def compute_returns(self, last_values, gamma, lam):  # type: ignore[no-untyped-def]
      # Simplified GAE(lambda) implementation matching older amp_rsl_rl
      advantage = 0
      for step in reversed(range(self.num_transitions_per_env)):
        next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
        next_is_not_terminal = 1.0 - self.dones[step].float()
        delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        self.returns[step] = advantage + self.values[step]
      self.advantages = self.returns - self.values
    setattr(RolloutStorage, "compute_returns", compute_returns)


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  # Set wandb entity and project early, before runner initialization
  if "WANDB_ENTITY" not in os.environ:
    os.environ["WANDB_ENTITY"] = "e1519767-national-university-of-singapore"
  if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = cfg.agent.wandb_project
  
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    cfg.env.commands is not None
    and "motion" in cfg.env.commands
    and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task:
    assert cfg.env.commands is not None
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # If motion_file is provided, use it directly.
    if cfg.motion_file is not None:
      motion_file_path = Path(cfg.motion_file)
      if not motion_file_path.exists():
        raise FileNotFoundError(
          f"Motion file not found: {motion_file_path}\n"
          f"Please provide a valid path to the motion.npz file."
        )
      motion_cmd.motion_file = str(motion_file_path.resolve())
      if rank == 0:
        print(f"[INFO] Using motion file from CLI: {motion_cmd.motion_file}")
    elif cfg.registry_name is not None:
      # Download from wandb registry.
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb
      from wandb.errors import CommError

      try:
        api = wandb.Api()
        if rank == 0:
          print(f"[INFO] Downloading motion artifact from wandb registry: {registry_name}")
        artifact = api.artifact(registry_name)
        motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        if rank == 0:
          print(f"[INFO] Successfully downloaded motion file: {motion_cmd.motion_file}")
      except CommError as e:
        error_msg = (
          f"Failed to download motion artifact from wandb registry: {registry_name}\n"
          f"Error: {e}\n\n"
          f"Possible solutions:\n"
          f"  1. Request access to the wandb registry from the project owner\n"
          f"  2. Download the motion file manually and use --motion-file <path>\n"
          f"  3. Verify your wandb authentication: wandb login"
        )
        raise RuntimeError(error_msg) from e
      except Exception as e:
        error_msg = (
          f"Unexpected error while downloading motion artifact: {registry_name}\n"
          f"Error: {e}\n\n"
          f"Consider using --motion-file <path> to specify a local motion file instead."
        )
        raise RuntimeError(error_msg) from e
    else:
      raise ValueError(
        "For tracking tasks, you must provide either:\n"
        "  --registry-name <wandb-registry-path>  (to download from wandb)\n"
        "  --motion-file <path-to-motion.npz>     (to use a local file)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  # Resolve custom algorithm class (mjlab / amp_rsl_rl) and inject env for AMP.
  alg_cfg = agent_cfg.get("algorithm") or {}
  alg_class_name = alg_cfg.get("class_name", "")
  if isinstance(alg_class_name, str) and "mjlab" in alg_class_name:
    import rsl_rl.runners.on_policy_runner as _runner_mod
    import mjlab as _mjlab
    setattr(_runner_mod, "mjlab", _mjlab)
  if isinstance(alg_class_name, str) and "amp_rsl_rl" in alg_class_name:
    _ensure_amp_wandb_compat()
    use_mjlab_amp_runner = getattr(runner_cls, "__name__", "") == "MjlabAmpOnPolicyRunner"
    # Our custom AMP implementation keeps amp-rsl-rl-compatible config shape,
    # but swaps in an optimized algorithm while still relying on env AMP obs.
    agent_cfg["algorithm"] = {
      **alg_cfg,
      "env": env,
      "class_name": (
        "mjlab.rl.mj_amp_ppo.MjlabAmpPPO"
        if use_mjlab_amp_runner
        else alg_cfg.get("class_name", "")
      ),
    }

    # Newer amp-rsl-rl runners (AMPOnPolicyRunner) expect extra top-level
    # config sections: "discriminator", "dataset" and "wandb_kwargs". Older
    # mjlab configs only encode discriminator hyperparameters inside
    # algorithm.* fields and use simple wandb_* fields, so we synthesize
    # minimal sections here for backward compatibility.
    if "discriminator" not in agent_cfg:
      disc_hidden = alg_cfg.get("disc_hidden_dims", (1024, 512))
      agent_cfg["discriminator"] = {
        # Match amp_rsl_rl.networks.Discriminator.__init__ expected kwargs.
        "hidden_dims": list(disc_hidden),
        "reward_scale": alg_cfg.get("disc_reward_scale", 2.0),
        # MimicKit-style AMP uses BCE-style discriminator training plus
        # observation normalization; Wasserstein was making the discriminator
        # saturate too easily on this task.
        "loss_type": "BCEWithLogits",
        "empirical_normalization": True,
      }

    if "dataset" not in agent_cfg and not use_mjlab_amp_runner:
      # Prefer env's AMP csv motion_file; else MJLAB_AMP_DATA_ROOT (.npy dir).
      amp_cfg = getattr(cfg.env, "amp", None)
      motion_file = getattr(amp_cfg, "motion_file", None) if amp_cfg is not None else None
      if motion_file is not None and (isinstance(motion_file, (list, tuple)) or isinstance(motion_file, str)):
        cwd = Path.cwd()
        files: list[str] = list(motion_file) if isinstance(motion_file, (list, tuple)) else [motion_file]
        amp_data_path, datasets = _build_amp_dataset_from_csv(files, cwd)
        if rank == 0:
          print(f"[INFO] AMP dataset built from {len(datasets)} csv motion file(s) -> {amp_data_path}")
        agent_cfg["dataset"] = {
          "amp_data_path": amp_data_path,
          "datasets": datasets,
          "slow_down_factor": 1,
        }
      else:
        amp_data_root = os.environ.get("MJLAB_AMP_DATA_ROOT")
        if amp_data_root is None:
          raise RuntimeError(
            "Detected amp-rsl-rl AMP_PPO but no AMP dataset: set env.amp.motion_file (list of .csv) "
            "or MJLAB_AMP_DATA_ROOT (directory of .npy), or provide agent_cfg['dataset']."
          )
        agent_cfg["dataset"] = {
          "amp_data_path": amp_data_root,
          "datasets": {"default": 1.0},
          "slow_down_factor": 1,
        }
    elif "dataset" not in agent_cfg:
      agent_cfg["dataset"] = {}

    # Logging: amp-rsl-rl uses its own WandbSummaryWriter which expects
    # cfg["wandb_kwargs"]["project"], etc. Map mjlab fields into that
    # structure if missing so that wandb logging works.
    if "wandb_kwargs" not in agent_cfg:
      agent_cfg["wandb_kwargs"] = {
        "project": cfg.agent.wandb_project,
        "entity": os.environ.get("WANDB_ENTITY"),
        "group": cfg.agent.experiment_name,
        "notes": "",
      }

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  # Remove non-serializable env from agent_cfg before dumping (used by amp-rsl-rl AMP)
  _alg = agent_cfg.get("algorithm")
  if _alg is not None and "env" in _alg:
    del _alg["env"]

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    load_sig = inspect.signature(runner.load)
    if "load_optimizer" in load_sig.parameters:
      runner.load(str(resume_path), load_optimizer=cfg.agent.load_optimizer)
    else:
      runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"
  
  # Set wandb entity and project for logging to e1519767-national-university-of-singapore/mjlab
  if "WANDB_ENTITY" not in os.environ:
    os.environ["WANDB_ENTITY"] = "e1519767-national-university-of-singapore"
  if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = args.agent.wandb_project

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*", "WANDB*"),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
