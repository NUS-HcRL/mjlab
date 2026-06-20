from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.scripts.resample_npz import resample_npz
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.time_step_total = self.joint_pos.shape[0]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w[:, self._body_indexes]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self._body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self._body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self._body_indexes]


def resolve_motion_paths(motion_file: str) -> list[str]:
  """Resolve motion_file to a list of npz paths (single file or directory)."""
  path = Path(motion_file)
  if not path.exists():
    raise FileNotFoundError(f"Motion path not found: {path}")
  if path.is_dir():
    files = sorted(path.glob("*.npz"))
    if not files:
      raise ValueError(f"No .npz files found in directory: {path}")
    return [str(f.resolve()) for f in files]
  if path.is_file():
    return [str(path.resolve())]
  raise ValueError(f"Motion path is not a file or directory: {path}")


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    motion_paths = [
      self._check_and_resample_fps(p, env.step_dt)
      for p in resolve_motion_paths(cfg.motion_file)
    ]
    self.motions = [
      MotionLoader(p, self.body_indexes, device=self.device) for p in motion_paths
    ]
    self.num_motions = len(self.motions)
    self.motion = self.motions[0]
    self.motion_lengths = torch.tensor(
      [m.time_step_total for m in self.motions], device=self.device, dtype=torch.long
    )
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    if self.num_motions > 1:
      print(
        f"[INFO] Loaded {self.num_motions} motion files "
        f"(frame lengths: min={self.motion_lengths.min().item()}, "
        f"max={self.motion_lengths.max().item()})"
      )

    self._qd_mask: torch.Tensor | None = None
    if cfg.qd_mask is not None:
      nj = self.motions[0].joint_pos.shape[1]
      if len(cfg.qd_mask) != nj:
        raise ValueError(
          f"qd_mask length ({len(cfg.qd_mask)}) must equal num joints ({nj}) "
          "from motion file joint_pos."
        )
      self._qd_mask = torch.tensor(
        list(cfg.qd_mask), dtype=torch.float32, device=self.device
      ).view(1, -1)

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    step_dt_inv = 1 / env.step_dt
    self.bin_counts = torch.tensor(
      [int(m.time_step_total // step_dt_inv) + 1 for m in self.motions],
      device=self.device,
      dtype=torch.long,
    )
    self.max_bin_count = int(self.bin_counts.max().item())
    self.bin_count = (
      self.max_bin_count if self.num_motions > 1 else int(self.bin_counts[0].item())
    )
    if self.num_motions == 1:
      self.bin_failed_count = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
      )
      self._current_bin_failed = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
      )
    else:
      self.bin_failed_count = torch.zeros(
        self.num_motions, self.max_bin_count, dtype=torch.float, device=self.device
      )
      self._current_bin_failed = torch.zeros(
        self.num_motions, self.max_bin_count, dtype=torch.float, device=self.device
      )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def future_frames_command(self) -> torch.Tensor:
    """返回未来9帧的关节位置和速度目标，按帧顺序堆叠（第一帧所有数据，第二帧所有数据...）
    
    Returns:
      torch.Tensor: shape (N, 9 * num_joints * 2) 的张量
    """
    # 获取未来9帧的索引（从 t+1 到 t+9）
    future_steps = torch.arange(1, 10, device=self.time_steps.device).unsqueeze(0)  # (1, 9)
    future_indices = self.time_steps.unsqueeze(1) + future_steps  # (N, 9)

    future_indices = self.clamp_frame_indices(future_indices)

    future_pos_frames = self._gather_for_envs_2d("joint_pos", future_indices)
    future_vel_frames = self._gather_for_envs_2d("joint_vel", future_indices)
    
    # 按帧顺序堆叠：对每一帧，拼接 pos 和 vel，然后堆叠所有帧
    frame_data_list = []
    for frame_idx in range(9):
      frame_pos = future_pos_frames[:, frame_idx, :]  # (N, num_joints)
      frame_vel = future_vel_frames[:, frame_idx, :]  # (N, num_joints)
      frame_data = torch.cat([frame_pos, frame_vel], dim=1)  # (N, num_joints * 2)
      frame_data_list.append(frame_data)
    
    return torch.cat(frame_data_list, dim=1)  # (N, 9 * num_joints * 2)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._gather_for_envs("joint_pos", self.time_steps)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._gather_for_envs("joint_vel", self.time_steps)

  def apply_qd_mask_to_vel(self, joint_vel: torch.Tensor) -> torch.Tensor:
    """乘参考关节速度 ``qd``（即 ``joint_vel``）。不作用于关节位置 ``q`` / ``joint_pos``。"""
    if self._qd_mask is None:
      return joint_vel
    # joint_vel: (..., num_joints); _qd_mask: (1, num_joints)
    return joint_vel * self._qd_mask

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self._gather_for_envs("body_pos_w", self.time_steps)
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._gather_for_envs("body_quat_w", self.time_steps)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._gather_for_envs("body_lin_vel_w", self.time_steps)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._gather_for_envs("body_ang_vel_w", self.time_steps)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._gather_for_envs(
        "body_pos_w", self.time_steps, body_index=self.motion_anchor_body_index
      )
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._gather_for_envs(
      "body_quat_w", self.time_steps, body_index=self.motion_anchor_body_index
    )

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._gather_for_envs(
      "body_lin_vel_w", self.time_steps, body_index=self.motion_anchor_body_index
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._gather_for_envs(
      "body_ang_vel_w", self.time_steps, body_index=self.motion_anchor_body_index
    )

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _motion_lengths_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    if self.num_motions == 1:
      return self.motion_lengths[0].expand(len(env_ids))
    return self.motion_lengths[self.motion_ids[env_ids]]

  def _bin_counts_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    if self.num_motions == 1:
      return self.bin_counts[0].expand(len(env_ids))
    return self.bin_counts[self.motion_ids[env_ids]]

  def _sample_motion_ids(self, env_ids: torch.Tensor) -> None:
    if self.num_motions > 1:
      self.motion_ids[env_ids] = torch.randint(
        0, self.num_motions, (len(env_ids),), device=self.device
      )

  def _gather_for_envs(
    self,
    attr: str,
    time_steps: torch.Tensor,
    body_index: int | None = None,
  ) -> torch.Tensor:
    time_steps = self.clamp_frame_indices(time_steps)
    if self.num_motions == 1:
      data = getattr(self.motions[0], attr)
      if body_index is not None:
        return data[time_steps, body_index]
      return data[time_steps]

    data0 = getattr(self.motions[0], attr)
    if body_index is not None:
      sample = data0[:1, body_index]
    else:
      sample = data0[:1]
    out = torch.empty(
      time_steps.shape[0], *sample.shape[1:], device=self.device, dtype=sample.dtype
    )
    for i, motion in enumerate(self.motions):
      mask = self.motion_ids == i
      if not mask.any():
        continue
      env_idx = mask.nonzero(as_tuple=True)[0]
      data = getattr(motion, attr)
      max_index = int(self.motion_lengths[i].item()) - 1
      local_steps = torch.clamp(time_steps[env_idx], 0, max_index)
      if body_index is not None:
        out[env_idx] = data[local_steps, body_index]
      else:
        out[env_idx] = data[local_steps]
    return out

  def clamp_frame_indices(self, indices: torch.Tensor) -> torch.Tensor:
    """Clamp frame indices to the valid range for each environment."""
    indices = torch.clamp(indices, min=0)
    if self.num_motions == 1:
      max_valid_index = int(self.motion_lengths[0].item()) - 1
      return torch.clamp(indices, max=max_valid_index)
    max_valid_index = self.motion_lengths[self.motion_ids] - 1
    if indices.dim() == 1:
      return torch.minimum(indices, max_valid_index)
    return torch.minimum(indices, max_valid_index.unsqueeze(-1))

  def _gather_for_envs_2d(
    self,
    attr: str,
    indices: torch.Tensor,
    body_index: int | None = None,
  ) -> torch.Tensor:
    """Gather motion data with per-env 2D indices (N, K)."""
    indices = self.clamp_frame_indices(indices)
    if self.num_motions == 1:
      data = getattr(self.motions[0], attr)
      if body_index is not None:
        return data[indices, body_index]
      return data[indices]

    data0 = getattr(self.motions[0], attr)
    safe_indices = indices[:1].clamp(0, data0.shape[0] - 1)
    if body_index is not None:
      sample = data0[safe_indices, body_index]
    else:
      sample = data0[safe_indices]
    out = torch.empty(
      indices.shape[0], *sample.shape[1:], device=self.device, dtype=sample.dtype
    )
    for i, motion in enumerate(self.motions):
      mask = self.motion_ids == i
      if not mask.any():
        continue
      env_idx = mask.nonzero(as_tuple=True)[0]
      data = getattr(motion, attr)
      max_index = int(self.motion_lengths[i].item()) - 1
      local_indices = torch.clamp(indices[env_idx], 0, max_index)
      if body_index is not None:
        out[env_idx] = data[local_indices, body_index]
      else:
        out[env_idx] = data[local_indices]
    return out

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    self._sample_motion_ids(env_ids)
    n = len(env_ids)

    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      failed_env_ids = env_ids[episode_failed]
      if self.num_motions == 1:
        current_bin_index = torch.clamp(
          (self.time_steps[failed_env_ids] * self.bin_count)
          // max(self.motion_lengths[0].item(), 1),
          0,
          self.bin_count - 1,
        )
        self._current_bin_failed[:] = torch.bincount(
          current_bin_index, minlength=self.bin_count
        )
      else:
        self._current_bin_failed.zero_()
        for motion_id in range(self.num_motions):
          motion_mask = self.motion_ids[failed_env_ids] == motion_id
          if not motion_mask.any():
            continue
          failed = failed_env_ids[motion_mask]
          bin_count = int(self.bin_counts[motion_id].item())
          length = max(int(self.motion_lengths[motion_id].item()), 1)
          bin_idx = torch.clamp(
            (self.time_steps[failed] * bin_count) // length, 0, bin_count - 1
          )
          self._current_bin_failed[motion_id, :bin_count] += torch.bincount(
            bin_idx, minlength=bin_count
          )

    sampled_bins = torch.zeros(n, dtype=torch.long, device=self.device)
    lengths = self._motion_lengths_for_envs(env_ids)
    bin_counts = self._bin_counts_for_envs(env_ids)

    if self.num_motions == 1:
      bin_count = self.bin_count
      sampling_probabilities = (
        self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(bin_count)
      )
      sampling_probabilities = torch.nn.functional.pad(
        sampling_probabilities.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),
        mode="replicate",
      )
      sampling_probabilities = torch.nn.functional.conv1d(
        sampling_probabilities, self.kernel.view(1, 1, -1)
      ).view(-1)
      sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
      sampled_bins = torch.multinomial(sampling_probabilities, n, replacement=True)
      H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
      H_norm = H / math.log(bin_count)
      pmax, imax = sampling_probabilities.max(dim=0)
      top1_bin = imax.float() / bin_count
    else:
      entropy_sum = 0.0
      pmax = torch.tensor(0.0, device=self.device)
      top1_bin = torch.tensor(0.5, device=self.device)
      for motion_id in range(self.num_motions):
        env_mask = self.motion_ids[env_ids] == motion_id
        if not env_mask.any():
          continue
        local_n = int(env_mask.sum().item())
        bin_count = int(self.bin_counts[motion_id].item())
        sampling_probabilities = (
          self.bin_failed_count[motion_id, :bin_count]
          + self.cfg.adaptive_uniform_ratio / float(bin_count)
        )
        sampling_probabilities = torch.nn.functional.pad(
          sampling_probabilities.unsqueeze(0).unsqueeze(0),
          (0, self.cfg.adaptive_kernel_size - 1),
          mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
          sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        local_bins = torch.multinomial(sampling_probabilities, local_n, replacement=True)
        sampled_bins[env_mask] = local_bins
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        entropy_sum += (H / math.log(bin_count)) * local_n
        local_pmax, local_imax = sampling_probabilities.max(dim=0)
        if local_pmax > pmax:
          pmax = local_pmax
          top1_bin = local_imax.float() / bin_count
      H_norm = entropy_sum / n

    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (n,), device=self.device))
      / bin_counts.float()
      * (lengths - 1).float()
    ).long()
    self.time_steps[env_ids] = torch.clamp(self.time_steps[env_ids], min=0)
    self.time_steps[env_ids] = torch.minimum(
      self.time_steps[env_ids],
      self.motion_lengths[self.motion_ids[env_ids]] - 1,
    )

    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = top1_bin

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self._sample_motion_ids(env_ids)
    n = len(env_ids)
    if self.num_motions == 1:
      self.time_steps[env_ids] = torch.randint(
        0, self.motion_lengths[0].item(), (n,), device=self.device
      )
    else:
      for motion_id in range(self.num_motions):
        env_mask = self.motion_ids[env_ids] == motion_id
        if not env_mask.any():
          continue
        local_env_ids = env_ids[env_mask]
        max_t = int(self.motion_lengths[motion_id].item())
        self.time_steps[local_env_ids] = torch.randint(
          0, max_t, (len(local_env_ids),), device=self.device
        )
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
      self._sample_motion_ids(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1
    if self.num_motions == 1:
      motion_end = self.time_steps >= self.motion_lengths[0]
    else:
      motion_end = self.time_steps >= self.motion_lengths[self.motion_ids]
    env_ids = torch.where(motion_end)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)
    self.time_steps = self.clamp_frame_indices(self.time_steps)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.asset_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.body_pos_w[visualizer.env_idx, 0].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = (
        self.body_quat_w[visualizer.env_idx, 0].cpu().numpy()
      )
      qpos[joint_q_adr] = self.joint_pos[visualizer.env_idx].cpu().numpy()

      visualizer.add_ghost_mesh(qpos, model=self._ghost_model)

    elif self.cfg.viz.mode == "frames":
      desired_body_pos = self.body_pos_w[visualizer.env_idx].cpu().numpy()
      desired_body_quat = self.body_quat_w[visualizer.env_idx]
      desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

      current_body_pos = self.robot_body_pos_w[visualizer.env_idx].cpu().numpy()
      current_body_quat = self.robot_body_quat_w[visualizer.env_idx]
      current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

      for i, body_name in enumerate(self.cfg.body_names):
        visualizer.add_frame(
          position=desired_body_pos[i],
          rotation_matrix=desired_body_rotm[i],
          scale=0.08,
          label=f"desired_{body_name}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=current_body_pos[i],
          rotation_matrix=current_body_rotm[i],
          scale=0.12,
          label=f"current_{body_name}",
        )

      desired_anchor_pos = self.anchor_pos_w[visualizer.env_idx].cpu().numpy()
      desired_anchor_quat = self.anchor_quat_w[visualizer.env_idx]
      desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=desired_anchor_pos,
        rotation_matrix=desired_rotation_matrix,
        scale=0.1,
        label="desired_anchor",
        axis_colors=_DESIRED_FRAME_COLORS,
      )

      current_anchor_pos = self.robot_anchor_pos_w[visualizer.env_idx].cpu().numpy()
      current_anchor_quat = self.robot_anchor_quat_w[visualizer.env_idx]
      current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=current_anchor_pos,
        rotation_matrix=current_rotation_matrix,
        scale=0.15,
        label="current_anchor",
      )

  def _check_and_resample_fps(self, motion_file: str, step_dt: float) -> str:
    """Check if motion file fps matches env step_dt, resample if needed.
    
    Args:
      motion_file: Path to the motion npz file
      step_dt: Environment step_dt (seconds per step)
    
    Returns:
      Path to the motion file (original or resampled)
    """
    # Calculate target fps from step_dt
    target_fps = 1.0 / step_dt
    
    # Load npz to check fps
    with np.load(motion_file) as data:
      if 'fps' not in data:
        raise ValueError(f"Motion file {motion_file} does not contain 'fps' key")
      
      # Some motion files store `fps` as a (1,) ndarray instead of a scalar.
      fps_arr = np.asarray(data["fps"])
      if fps_arr.size != 1:
        raise ValueError(
          f"Motion file {motion_file} has invalid 'fps' shape {fps_arr.shape} "
          f"(expected size==1)."
        )
      input_fps = float(fps_arr.reshape(()).item())
    
    # Check if fps matches (with small tolerance for floating point)
    fps_tolerance = 0.01  # 0.01 Hz tolerance
    if abs(input_fps - target_fps) < fps_tolerance:
      # Fps matches, use original file
      return motion_file
    
    # Fps doesn't match, need to resample
    print(f"[INFO] Motion file fps ({input_fps:.2f} Hz) doesn't match env step_dt "
          f"({step_dt:.4f} s, {target_fps:.2f} Hz). Resampling...")
    
    # Generate output file path
    input_path = Path(motion_file)
    # Create filename like: original_name_50fps.npz
    output_path = input_path.parent / f"{input_path.stem}_{int(target_fps)}fps{input_path.suffix}"
    
    # Check if resampled file already exists
    if output_path.exists():
      print(f"[INFO] Resampled file already exists: {output_path}")
      # Verify the existing file has correct fps
      with np.load(output_path) as existing_data:
        if 'fps' in existing_data:
          fps_arr = np.asarray(existing_data["fps"])
          if fps_arr.size != 1:
            raise ValueError(
              f"Existing resampled file {output_path} has invalid 'fps' shape "
              f"{fps_arr.shape} (expected size==1)."
            )
          existing_fps = float(fps_arr.reshape(()).item())
          if abs(existing_fps - target_fps) < fps_tolerance:
            print(f"[INFO] Using existing resampled file with fps {existing_fps:.2f} Hz")
            return str(output_path)
          else:
            print(f"[WARN] Existing resampled file has wrong fps ({existing_fps:.2f} Hz), "
                  f"re-resampling to {target_fps:.2f} Hz...")
        else:
          print(f"[WARN] Existing resampled file missing fps key, re-resampling...")
    
    # Resample the motion file
    print(f"[INFO] Resampling {motion_file} to {target_fps:.2f} Hz...")
    resample_npz(
      input_file=motion_file,
      output_file=str(output_path),
      target_fps=target_fps,
    )
    print(f"[INFO] Resampled motion saved to: {output_path}")
    
    return str(output_path)


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  """Motion tracking command configuration.

  ``motion_file`` may be a single ``.npz`` path or a directory containing multiple
  ``.npz`` files (all motions are sampled uniformly per environment at reset).
  """
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  asset_name: str
  class_type: type[CommandTerm] = MotionCommand
  qd_mask: tuple[float, ...] | None = None
  """Per-joint multiplier on reference **joint velocity** only (notation ``qd`` / ``qdot``).
  Does **not** multiply ``joint_pos`` (``q``). Length equals num joints; ``None`` means all ones."""
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)
