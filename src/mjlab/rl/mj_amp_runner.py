from __future__ import annotations

from collections import deque
import inspect
import warnings
from typing import Any, Callable, cast

import os
import statistics
import time

import amp_rsl_rl
import rsl_rl
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils import resolve_obs_groups, store_code_state
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from amp_rsl_rl.algorithms import AMP_PPO
from amp_rsl_rl.networks import ActorCriticMoE, Discriminator

from mjlab.rl.amp_loader import MjlabAmpLoader
from mjlab.utils.lab_api.rl.exporter import export_policy_as_onnx as mjlab_export_policy_as_onnx


_BUILTIN_CLASSES = {
  "ActorCritic": ActorCritic,
  "ActorCriticRecurrent": ActorCriticRecurrent,
  "ActorCriticMoE": ActorCriticMoE,
  "AMP_PPO": AMP_PPO,
}


def resolve_class(class_name: str) -> type:
  """Resolve a class by name, mirroring amp_rsl_rl.runners.amp_on_policy_runner."""
  if class_name in _BUILTIN_CLASSES:
    return _BUILTIN_CLASSES[class_name]
  if "." in class_name:
    module_path, cls_name = class_name.rsplit(".", 1)
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)
  raise ValueError(
    f"Unknown class name '{class_name}'. Provide a fully-qualified "
    f"module path (e.g. 'my_package.module.{class_name}') or one of "
    f"the built-in names: {list(_BUILTIN_CLASSES.keys())}."
  )


class MjlabAmpOnPolicyRunner:
  """AMP runner that uses mjlab's disc_obs demos instead of AMPLoader.

  This is a near-copy of amp_rsl_rl.runners.AMPOnPolicyRunner, with the
  only functional change being that expert AMP data comes from
  MjlabAmpLoader(env, device), which in turn calls the environment's
  AMPHelper.fetch_disc_obs_demo_pairs(). This guarantees that policy
  and expert AMP observations have matching dimensions.
  """

  def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
    self.cfg = train_cfg
    self.alg_cfg = train_cfg["algorithm"]
    self.policy_cfg = train_cfg["policy"]
    # dataset_cfg is preserved for logging / compatibility but not used
    self.dataset_cfg = train_cfg.get("dataset", {})
    self.discriminator_cfg = train_cfg["discriminator"]
    self.device = device
    self.env = env

    # Optional custom exporter function (set via set_export_policy_fn)
    self._export_policy_fn: Callable | None = None

    observations = self.env.get_observations()
    default_sets = ["critic"]
    self.cfg["obs_groups"] = resolve_obs_groups(
      observations, self.cfg.get("obs_groups"), default_sets
    )

    actor_critic_class = resolve_class(self.policy_cfg.pop("class_name"))
    actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticMoE = (
      actor_critic_class(
        observations,
        self.cfg["obs_groups"],
        self.env.num_actions,
        **self.policy_cfg,
      ).to(self.device)
    )

    # Initialize all the ingredients required for AMP (discriminator, dataset loader)
    num_amp_obs = observations["amp"].shape[1]

    # Expert AMP data from mjlab env disc_obs demos (same dim as policy obs["amp"])
    amp_data = MjlabAmpLoader(cast(Any, self.env), device=self.device)

    self.discriminator = Discriminator(
      input_dim=num_amp_obs * 2,  # concat current and next AMP obs
      hidden_layer_sizes=self.discriminator_cfg["hidden_dims"],
      reward_scale=self.discriminator_cfg["reward_scale"],
      device=self.device,
      loss_type=self.discriminator_cfg["loss_type"],
      empirical_normalization=self.discriminator_cfg["empirical_normalization"],
    ).to(self.device)

    # Initialize the AMP-PPO algorithm
    alg_class = resolve_class(self.alg_cfg.pop("class_name"))
    alg_kwargs = dict(self.alg_cfg)
    valid_params = set(inspect.signature(alg_class.__init__).parameters)
    intentionally_ignored_alg_keys = {
      "class_name",
      "env",
      # These are consumed when building the top-level discriminator config.
      "disc_hidden_dims",
      "disc_reward_scale",
      # Used directly by this custom runner for reward mixing.
      "reward_mix_mode",
      "reward_mix_ema_decay",
      "reward_mix_scale_clip",
    }
    dropped_alg_keys: list[str] = []
    for key in list(alg_kwargs.keys()):
      if key not in valid_params:
        if key not in intentionally_ignored_alg_keys:
          dropped_alg_keys.append(key)
        alg_kwargs.pop(key)
    if dropped_alg_keys:
      warnings.warn(
        "Ignoring unsupported AMP algorithm config keys for "
        f"{alg_class.__name__}: {sorted(dropped_alg_keys)}. "
        "These fields are defined in the config but are not wired into "
        "the custom mjlab AMP runner.",
        stacklevel=2,
      )

    self.alg: Any = alg_class(
      actor_critic=actor_critic,
      discriminator=self.discriminator,
      amp_data=amp_data,
      device=self.device,
      **alg_kwargs,
    )
    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]
    # init storage and model
    obs_template = observations.clone().detach().to(self.device)
    self.alg.init_storage(
      self.env.num_envs,
      self.num_steps_per_env,
      obs_template,
      (self.env.num_actions,),
    )

    # Log
    self.log_dir = log_dir
    self.writer = None
    self.logger_type = None
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0
    self.git_status_repos = [rsl_rl.__file__, amp_rsl_rl.__file__]
    self._reward_mix_mode = str(self.alg_cfg.get("reward_mix_mode", "legacy"))
    self._reward_mix_ema_decay = float(self.alg_cfg.get("reward_mix_ema_decay", 0.99))
    lo, hi = self.alg_cfg.get("reward_mix_scale_clip", (0.2, 5.0))
    self._reward_mix_scale_clip = (float(lo), float(hi))
    self._task_abs_ema = 1.0
    self._style_abs_ema = 1.0

  # The rest of the methods (learn, log, save, load, etc.) are identical
  # to amp_rsl_rl.runners.AMPOnPolicyRunner and are reused via delegation.

  # Delegate train/eval/save/load/log to a wrapped AMPOnPolicyRunner instance
  # initialized with the same state, except for amp_data. For simplicity and
  # to avoid code duplication, we keep all high-level methods the same by
  # inheriting from AMP_PPO directly.

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    import wandb

    if self.log_dir is not None and self.writer is None:
      self.logger_type = self.cfg.get("logger", "tensorboard").lower()
      if self.logger_type == "neptune":
        from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

        self.writer = NeptuneSummaryWriter(
          log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
        )
        self.writer.log_config(
          self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
        )
      elif self.logger_type == "wandb":
        from amp_rsl_rl.utils.wandb_utils import WandbSummaryWriter

        def update_run_name_with_sequence(prefix: str) -> None:
          if wandb.run is None:
            return
          project = wandb.run.project
          entity = wandb.run.entity
          api = wandb.Api()
          runs = api.runs(f"{entity}/{project}")
          max_num = 0
          for run in runs:
            if run.name.startswith(prefix):
              numeric_suffix = run.name[len(prefix) :]
              try:
                run_num = int(numeric_suffix)
                if run_num > max_num:
                  max_num = run_num
              except ValueError:
                continue
          wandb.run.name = f"{prefix}{max_num + 1}"
          print("Updated run name to:", wandb.run.name)

        self.writer = WandbSummaryWriter(
          log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
        )
        update_run_name_with_sequence(prefix=self.cfg["wandb_kwargs"]["project"])
        self.writer.log_config(
          self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
        )
      elif self.logger_type == "tensorboard":
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
      else:
        raise AssertionError("logger type not found")

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    amp_obs = obs["amp"].clone()
    self.train_mode()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )
    cur_episode_length = torch.zeros(
      self.env.num_envs, dtype=torch.float, device=self.device
    )

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations
    self._run_start_iteration = start_iter
    log_dir = cast(str, self.log_dir)
    for it in range(start_iter, tot_iter):
      start = time.time()
      mean_style_reward_log = 0.0
      mean_style_reward_std_log = 0.0
      mean_task_reward_log = 0.0
      mean_style_balance_scale_log = 1.0
      mean_task_weight_scale_log = 1.0

      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(obs)
          self.alg.act_amp(amp_obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs = obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)
          # Guard training from occasional simulator non-finite spikes.
          rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
          for key, value in obs.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
              obs[key] = torch.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
              )

          next_amp_obs = obs["amp"].clone()
          pair_next_amp_obs = next_amp_obs
          done_mask = dones.view(-1).bool()
          if done_mask.any():
            # Do not create AMP pairs across episode boundaries. Using the
            # post-reset AMP state as s_{t+1} for a terminal transition makes
            # discriminator targets inconsistent and quickly saturates it.
            pair_next_amp_obs = next_amp_obs.clone()
            pair_next_amp_obs[done_mask] = amp_obs[done_mask]
          style_rewards = self.alg.predict_style_reward(amp_obs, pair_next_amp_obs)
          style_rewards = torch.nan_to_num(
            style_rewards, nan=0.0, posinf=0.0, neginf=0.0
          )
          style_rewards_for_mix = style_rewards
          if self._reward_mix_mode == "ema_balance":
            task_abs = float(rewards.detach().abs().mean().item())
            style_abs = float(style_rewards.detach().abs().mean().item())
            decay = self._reward_mix_ema_decay
            self._task_abs_ema = decay * self._task_abs_ema + (1.0 - decay) * task_abs
            self._style_abs_ema = decay * self._style_abs_ema + (1.0 - decay) * style_abs
            raw_scale = self._task_abs_ema / max(self._style_abs_ema, 1e-6)
            lo, hi = self._reward_mix_scale_clip
            style_balance_scale = max(lo, min(hi, raw_scale))
            mean_style_balance_scale_log += style_balance_scale
            style_rewards_for_mix = style_rewards * style_balance_scale

          mean_task_reward_log += rewards.mean().item()
          mean_style_reward_log += style_rewards.mean().item()
          mean_style_reward_std_log += style_rewards.std(unbiased=False).item()
          task_weight_scale = float(
            getattr(self.env, "task_reward_weight_scale", 1.0)
          )
          mean_task_weight_scale_log += task_weight_scale

          rewards = (
            self.alg.task_reward_weight * task_weight_scale * rewards
            + self.alg.disc_reward_weight * style_rewards_for_mix
          )

          self.alg.process_env_step(obs, rewards, dones, extras)
          self.alg.process_amp_step(pair_next_amp_obs)
          amp_obs = next_amp_obs

          if self.log_dir is not None:
            if "episode" in extras:
              ep_infos.append(extras["episode"])
            elif "log" in extras:
              ep_infos.append(extras["log"])
            cur_reward_sum += rewards
            cur_episode_length += 1
            new_ids = torch.nonzero(dones, as_tuple=False)
            if new_ids.numel() > 0:
              env_indices = new_ids.view(-1)
              rewbuffer.extend(cur_reward_sum[env_indices].cpu().tolist())
              lenbuffer.extend(cur_episode_length[env_indices].cpu().tolist())
              cur_reward_sum[env_indices] = 0
              cur_episode_length[env_indices] = 0

        stop = time.time()
        collection_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      mean_style_reward_log /= self.num_steps_per_env
      mean_style_reward_std_log /= self.num_steps_per_env
      mean_task_reward_log /= self.num_steps_per_env
      mean_style_balance_scale_log /= self.num_steps_per_env
      mean_task_weight_scale_log /= self.num_steps_per_env

      (
        mean_value_loss,
        mean_surrogate_loss,
        mean_amp_loss,
        mean_grad_pen_loss,
        mean_policy_pred,
        mean_expert_pred,
        mean_accuracy_policy,
        mean_accuracy_expert,
        mean_kl_divergence,
        mean_ratio,
        mean_ratio_std,
        mean_old_logp,
        mean_new_logp,
        mean_actor_grad_norm,
        mean_policy_input_norm,
        mean_expert_input_norm,
        mean_pair_distance,
      ) = self.alg.update()
      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      if self.log_dir is not None:
        self.log(locals())
      if it > 0 and it % self.save_interval == 0:
        self.save(os.path.join(log_dir, f"model_{it}.pt"), save_onnx=True)
      ep_infos.clear()
      if it == start_iter:
        git_file_paths = store_code_state(log_dir, self.git_status_repos)
        if self.logger_type in ["wandb", "neptune"] and git_file_paths:
          for path in git_file_paths:
            cast(Any, self.writer).save_file(path)

    self.save(
      os.path.join(log_dir, f"model_{self.current_learning_iteration}.pt"),
      save_onnx=True,
    )

  # The remaining helper methods (log, save, load, etc.) are also delegated.

  def log(self, locs: dict, width: int = 80, pad: int = 35):
    writer = cast(Any, self.writer)
    self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    self.tot_time += locs["collection_time"] + locs["learn_time"]
    iteration_time = locs["collection_time"] + locs["learn_time"]

    ep_string = ""
    if locs["ep_infos"]:
      for key in locs["ep_infos"][0]:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in locs["ep_infos"]:
          if key not in ep_info:
            continue
          if not isinstance(ep_info[key], torch.Tensor):
            ep_info[key] = torch.Tensor([ep_info[key]])
          if len(ep_info[key].shape) == 0:
            ep_info[key] = ep_info[key].unsqueeze(0)
          infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor)
        if "/" in key:
          writer.add_scalar(key, value, locs["it"])
          ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        else:
          writer.add_scalar("Episode/" + key, value, locs["it"])
          ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    if getattr(self.alg.actor_critic, "noise_std_type", "scalar") == "log":
      mean_std_value = torch.exp(self.alg.actor_critic.log_std).mean()
    else:
      mean_std_value = self.alg.actor_critic.std.mean()
    fps = int(
      self.num_steps_per_env
      * self.env.num_envs
      / (locs["collection_time"] + locs["learn_time"])
    )

    writer.add_scalar(
      "Loss/value_function", locs["mean_value_loss"], locs["it"]
    )
    writer.add_scalar(
      "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
    )
    writer.add_scalar("Loss/amp_loss", locs["mean_amp_loss"], locs["it"])
    writer.add_scalar(
      "Loss/grad_pen_loss", locs["mean_grad_pen_loss"], locs["it"]
    )
    writer.add_scalar("Loss/policy_pred", locs["mean_policy_pred"], locs["it"])
    writer.add_scalar("Loss/expert_pred", locs["mean_expert_pred"], locs["it"])
    writer.add_scalar(
      "Loss/accuracy_policy", locs["mean_accuracy_policy"], locs["it"]
    )
    writer.add_scalar(
      "Loss/accuracy_expert", locs["mean_accuracy_expert"], locs["it"]
    )
    writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
    writer.add_scalar(
      "Loss/mean_kl_divergence", locs["mean_kl_divergence"], locs["it"]
    )
    writer.add_scalar("Debug/ratio_mean", locs["mean_ratio"], locs["it"])
    writer.add_scalar("Debug/ratio_std", locs["mean_ratio_std"], locs["it"])
    writer.add_scalar("Debug/old_logp_mean", locs["mean_old_logp"], locs["it"])
    writer.add_scalar("Debug/new_logp_mean", locs["mean_new_logp"], locs["it"])
    writer.add_scalar(
      "Debug/actor_grad_norm", locs["mean_actor_grad_norm"], locs["it"]
    )
    writer.add_scalar(
      "Debug/policy_amp_input_norm", locs["mean_policy_input_norm"], locs["it"]
    )
    writer.add_scalar(
      "Debug/expert_amp_input_norm", locs["mean_expert_input_norm"], locs["it"]
    )
    writer.add_scalar(
      "Debug/policy_expert_pair_distance", locs["mean_pair_distance"], locs["it"]
    )
    writer.add_scalar(
      "Policy/mean_noise_std", mean_std_value.item(), locs["it"]
    )
    writer.add_scalar("Perf/total_fps", fps, locs["it"])
    writer.add_scalar(
      "Perf/collection time", locs["collection_time"], locs["it"]
    )
    if self.log_dir and self.logger_type == "wandb":
      writer.add_video_files(self.log_dir, step=locs["it"])
    writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
    if len(locs["rewbuffer"]) > 0:
      writer.add_scalar(
        "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
      )
      writer.add_scalar(
        "Train/mean_episode_length",
        statistics.mean(locs["lenbuffer"]),
        locs["it"],
      )
      writer.add_scalar(
        "Train/mean_style_reward", locs["mean_style_reward_log"], locs["it"]
      )
      writer.add_scalar(
        "Train/style_reward_std", locs["mean_style_reward_std_log"], locs["it"]
      )
      writer.add_scalar(
        "Train/style_balance_scale", locs["mean_style_balance_scale_log"], locs["it"]
      )
      writer.add_scalar(
        "Train/task_weight_scale", locs["mean_task_weight_scale_log"], locs["it"]
      )
      writer.add_scalar(
        "Train/mean_task_reward", locs["mean_task_reward_log"], locs["it"]
      )
      if self.logger_type != "wandb":
        writer.add_scalar(
          "Train/mean_reward/time",
          statistics.mean(locs["rewbuffer"]),
          self.tot_time,
        )
        writer.add_scalar(
          "Train/mean_episode_length/time",
          statistics.mean(locs["lenbuffer"]),
          self.tot_time,
        )

    title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
    if len(locs["rewbuffer"]) > 0:
      log_string = (
        f"""{'#' * width}\n"""
        f"""{title.center(width, ' ')}\n\n"""
        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
        f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
        f"""{'Ratio mean/std:':>{pad}} {locs['mean_ratio']:.4f} / {locs['mean_ratio_std']:.4f}\n"""
        f"""{'Old/New logp mean:':>{pad}} {locs['mean_old_logp']:.4f} / {locs['mean_new_logp']:.4f}\n"""
        f"""{'Actor grad norm:':>{pad}} {locs['mean_actor_grad_norm']:.4f}\n"""
        f"""{'AMP input norm p/e:':>{pad}} {locs['mean_policy_input_norm']:.4f} / {locs['mean_expert_input_norm']:.4f}\n"""
        f"""{'AMP pair distance:':>{pad}} {locs['mean_pair_distance']:.4f}\n"""
        f"""{'Mean action noise std:':>{pad}} {mean_std_value.item():.2f}\n"""
        f"""{'Mean mixed reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
        f"""{'Mean style reward:':>{pad}} {locs['mean_style_reward_log']:.4f}\n"""
        f"""{'Style balance scale:':>{pad}} {locs['mean_style_balance_scale_log']:.4f}\n"""
        f"""{'Task weight scale:':>{pad}} {locs['mean_task_weight_scale_log']:.4f}\n"""
        f"""{'Mean task reward:':>{pad}} {locs['mean_task_reward_log']:.4f}\n"""
        f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
      )
    else:
      log_string = (
        f"""{'#' * width}\n"""
        f"""{title.center(width, ' ')}\n\n"""
        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
        f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
        f"""{'Ratio mean/std:':>{pad}} {locs['mean_ratio']:.4f} / {locs['mean_ratio_std']:.4f}\n"""
        f"""{'Old/New logp mean:':>{pad}} {locs['mean_old_logp']:.4f} / {locs['mean_new_logp']:.4f}\n"""
        f"""{'Actor grad norm:':>{pad}} {locs['mean_actor_grad_norm']:.4f}\n"""
        f"""{'AMP input norm p/e:':>{pad}} {locs['mean_policy_input_norm']:.4f} / {locs['mean_expert_input_norm']:.4f}\n"""
        f"""{'AMP pair distance:':>{pad}} {locs['mean_pair_distance']:.4f}\n"""
        f"""{'Mean action noise std:':>{pad}} {mean_std_value.item():.2f}\n"""
      )

    log_string += ep_string

    run_start_iteration = getattr(self, "_run_start_iteration", 0)
    session_iters = max(1, locs["it"] - run_start_iteration + 1)
    remaining_iters = max(0, locs["tot_iter"] - locs["it"] - 1)
    eta_seconds = self.tot_time / session_iters * remaining_iters
    eta_h, rem = divmod(eta_seconds, 3600)
    eta_m, eta_s = divmod(rem, 60)
    global_timesteps = (
      (locs["it"] + 1) * self.num_steps_per_env * self.env.num_envs
    )

    log_string += (
      f"""{'-' * width}\n"""
      f"""{'Total timesteps:':>{pad}} {global_timesteps}\n"""
      f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
      f"""{'Session time:':>{pad}} {self.tot_time:.2f}s\n"""
      f"""{'ETA:':>{pad}} {int(eta_h)}h {int(eta_m)}m {int(eta_s)}s\n"""
    )
    print(log_string)

  def set_export_policy_fn(self, fn: Callable) -> None:
    self._export_policy_fn = fn

  def save(self, path, infos=None, save_onnx=False):
    saved_dict = {
      "model_state_dict": self.alg.actor_critic.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "discriminator_state_dict": self.alg.discriminator.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos,
    }
    torch.save(saved_dict, path)

    if self.logger_type in ["neptune", "wandb"]:
      cast(Any, self.writer).save_model(path, self.current_learning_iteration)

    if save_onnx:
      onnx_folder = os.path.dirname(path)
      iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
      onnx_model_name = f"policy_{iteration}.onnx"

      was_training = self.alg.actor_critic.training
      self.alg.actor_critic.eval()
      try:
        export_fn = self._export_policy_fn or mjlab_export_policy_as_onnx
        export_fn(
          self.alg.actor_critic,
          normalizer=self.alg.actor_critic.actor_obs_normalizer,
          path=onnx_folder,
          filename=onnx_model_name,
        )
      finally:
        if was_training:
          self.alg.actor_critic.train()

      if self.logger_type in ["neptune", "wandb"]:
        cast(Any, self.writer).save_model(
          os.path.join(onnx_folder, onnx_model_name),
          self.current_learning_iteration,
        )

  def load(self, path, load_optimizer=True, weights_only=False):
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    result = _Base.load(base, path, load_optimizer, weights_only)
    # _Base.load mutates runner state (e.g. current_learning_iteration, optimizers).
    # Sync all mutated fields back so resumed training continues from checkpoint iter.
    self.__dict__.update(base.__dict__)
    return result

  def get_inference_policy(self, device=None):
    from amp_rsl_rl.runners.amp_on_policy_runner import AMPOnPolicyRunner as _Base

    base = _Base.__new__(_Base)  # type: ignore
    base.__dict__.update(self.__dict__)
    return _Base.get_inference_policy(base, device)

  def train_mode(self):
    self.alg.actor_critic.train()
    self.alg.discriminator.train()

  def eval_mode(self):
    self.alg.actor_critic.eval()
    self.alg.discriminator.eval()

  def add_git_repo_to_log(self, repo_file_path):
    self.git_status_repos.append(repo_file_path)
