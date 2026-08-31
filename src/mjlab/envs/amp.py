"""AMP (Adversarial Motion Priors) support for manager-based RL envs.

Provides disc_obs (discriminator observation) from robot state history,
get_disc_obs_space(), and fetch_disc_obs_demo() for use with AMP-style training
(e.g. MimicKit amp_agent, or amp-rsl-rl).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_mul,
  subtract_frame_transforms,
)
from mjlab.utils.spaces import Box

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


FALL_DIRECTION_NAMES = (
  "forward",
  "forward_left",
  "left",
  "backward_left",
  "backward",
  "backward_right",
  "right",
  "forward_right",
)
_FALL_DIRECTION_TO_INDEX = {
  name: index for index, name in enumerate(FALL_DIRECTION_NAMES)
}


def _fall_direction_one_hot(
  direction_names: list[str] | tuple[str, ...],
  device: str,
) -> torch.Tensor:
  """Convert canonical fall direction names to categorical one-hot labels."""
  indexes = []
  for name in direction_names:
    try:
      indexes.append(_FALL_DIRECTION_TO_INDEX[name])
    except KeyError as exc:
      supported = ", ".join(FALL_DIRECTION_NAMES)
      raise ValueError(
        f"Unsupported AMP fall direction '{name}'. Expected one of: {supported}."
      ) from exc
  return torch.nn.functional.one_hot(
    torch.tensor(indexes, device=device, dtype=torch.long),
    num_classes=len(FALL_DIRECTION_NAMES),
  ).float()


def _quantize_fall_direction(direction_xy: torch.Tensor) -> torch.Tensor:
  """Quantize planar directions to the nearest of eight 45-degree classes."""
  angle = torch.atan2(direction_xy[:, 1], direction_xy[:, 0])
  direction_index = torch.round(angle / (torch.pi / 4.0)).long() % len(
    FALL_DIRECTION_NAMES
  )
  return torch.nn.functional.one_hot(
    direction_index, num_classes=len(FALL_DIRECTION_NAMES)
  ).to(dtype=direction_xy.dtype)


def _quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
  """Quaternion to 6D rotation (first two columns of R). (..., 4) -> (..., 6)."""
  mat = matrix_from_quat(quat)
  return mat[..., :2].reshape(*quat.shape[:-1], 6)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Extract yaw (around Z) from quaternion (w, x, y, z)."""
  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return torch.atan2(siny_cosp, cosy_cosp)


def _heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
  """Inverse of yaw-only quaternion. quat (N, 4) wxyz."""
  from mjlab.utils.lab_api.math import quat_conjugate, quat_from_euler_xyz

  yaw = _yaw_from_quat(quat)
  q_yaw_inv = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), -yaw)
  return quat_conjugate(q_yaw_inv)


def compute_disc_obs(
  ref_root_pos: torch.Tensor,
  ref_root_quat: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  global_obs: bool = False,
  root_height_obs: bool = True,
  include_root_xy: bool = True,
  include_root_rot: bool = True,
  include_root_vel: bool = True,
  include_projected_gravity: bool = False,
  extra_body_pos_w: torch.Tensor | None = None,
  extra_body_quat_w: torch.Tensor | None = None,
  fall_direction_obs: torch.Tensor | None = None,
) -> torch.Tensor:
  """Compute discriminator observation from history of states.

  All inputs except ref_* have shape (num_envs, num_steps, ...). ref_* (num_envs, ...).
  Output (num_envs, disc_dim).
  """
  n, t = root_pos.shape[0], root_pos.shape[1]
  ref_pos = ref_root_pos.unsqueeze(1)

  root_pos_rel = root_pos - ref_pos

  if not global_obs:
    heading_inv = _heading_quat_inv(ref_root_quat)
    heading_inv_expand = heading_inv.unsqueeze(1).expand(n, t, 4)
    root_pos_rel_flat = root_pos_rel.reshape(-1, 3)
    heading_flat = heading_inv_expand.reshape(-1, 4)
    root_pos_rel = quat_apply_inverse(heading_flat, root_pos_rel_flat).reshape(n, t, 3)
    heading_inv_expand = heading_inv.unsqueeze(1).expand(n, t, 4)
    root_quat_local = quat_mul(heading_inv_expand, root_quat)
    if include_root_vel:
      root_lin_vel = quat_apply_inverse(
        heading_flat, root_lin_vel.reshape(-1, 3)
      ).reshape(n, t, 3)
      root_ang_vel = quat_apply_inverse(
        heading_flat, root_ang_vel.reshape(-1, 3)
      ).reshape(n, t, 3)
  else:
    root_quat_local = root_quat

  root_pos_terms: list[torch.Tensor] = []
  if include_root_xy:
    root_pos_terms.append(root_pos_rel[..., :2])
  if root_height_obs:
    # Use actual root height rather than height relative to the reference
    # frame. With num_disc_obs_steps == 1, the relative z would otherwise be
    # identically zero and provide no fall-state information.
    root_pos_terms.append(root_pos[..., 2:3])

  root_rot_6d = _quat_to_6d(root_quat_local.reshape(-1, 4)).reshape(n, t, 6)
  joint_pos_exp = (
    joint_pos
    if joint_pos.dim() >= 3
    else joint_pos.unsqueeze(1).expand(n, t, joint_pos.shape[-1])
  )
  joint_vel_exp = (
    joint_vel
    if joint_vel.dim() >= 3
    else joint_vel.unsqueeze(1).expand(n, t, joint_vel.shape[-1])
  )
  pos_obs_parts = [*root_pos_terms]
  if include_root_rot:
    pos_obs_parts.append(root_rot_6d)
  pos_obs_parts.append(joint_pos_exp)
  if extra_body_pos_w is not None:
    if extra_body_quat_w is None:
      raise ValueError("extra_body_pos_w requires extra_body_quat_w.")
    n_b = extra_body_pos_w.shape[2]
    flat_n = n * t * n_b
    ap = root_pos[:, :, None, :].expand(n, t, n_b, 3).reshape(flat_n, 3)
    aq = root_quat[:, :, None, :].expand(n, t, n_b, 4).reshape(flat_n, 4)
    bp = extra_body_pos_w.reshape(flat_n, 3)
    bq = extra_body_quat_w.reshape(flat_n, 4)
    pos_b, _ = subtract_frame_transforms(ap, aq, bp, bq)
    pos_obs_parts.append(pos_b.reshape(n, t, n_b * 3))
  pos_obs = torch.cat(pos_obs_parts, dim=-1)
  vel_obs_parts: list[torch.Tensor] = []
  if include_projected_gravity:
    gravity_w = torch.zeros((n, t, 3), device=root_pos.device, dtype=root_pos.dtype)
    gravity_w[..., 2] = -1.0
    projected_gravity = quat_apply_inverse(
      root_quat_local.reshape(-1, 4), gravity_w.reshape(-1, 3)
    ).reshape(n, t, 3)
    vel_obs_parts.append(projected_gravity)
  if include_root_vel:
    vel_obs_parts.extend([root_lin_vel, root_ang_vel])
  vel_obs_parts.append(joint_vel_exp)
  vel_obs = torch.cat(vel_obs_parts, dim=-1)
  disc_obs = torch.cat([pos_obs, vel_obs], dim=-1).reshape(n, -1)
  if fall_direction_obs is not None:
    if fall_direction_obs.shape != (n, len(FALL_DIRECTION_NAMES)):
      raise ValueError(
        "fall_direction_obs must have shape "
        f"({n}, {len(FALL_DIRECTION_NAMES)}), got "
        f"{tuple(fall_direction_obs.shape)}."
      )
    disc_obs = torch.cat([disc_obs, fall_direction_obs], dim=-1)
  return disc_obs


def calc_disc_obs_dim(
  num_disc_obs_steps: int,
  num_joints: int,
  root_height_obs: bool = True,
  include_root_xy: bool = True,
  include_root_rot: bool = True,
  include_root_vel: bool = True,
  include_projected_gravity: bool = False,
  num_disc_body_pos_b: int = 0,
  include_fall_direction_obs: bool = False,
) -> int:
  """Discriminator observation dimension."""
  pos_dim = num_joints
  if include_root_xy:
    pos_dim += 2
  if root_height_obs:
    pos_dim += 1
  if include_root_rot:
    pos_dim += 6
  pos_dim += 3 * num_disc_body_pos_b
  vel_dim = (
    num_joints
    + (6 if include_root_vel else 0)
    + (3 if include_projected_gravity else 0)
  )
  direction_dim = len(FALL_DIRECTION_NAMES) if include_fall_direction_obs else 0
  return num_disc_obs_steps * (pos_dim + vel_dim) + direction_dim


@dataclass
class AMPCfg:
  """Configuration for AMP in an env."""

  num_disc_obs_steps: int = 2
  asset_name: str = "robot"
  root_body_name: str = "LINK_BASE"
  """Body used as the AMP root for pos/quat/vel semantics."""
  motion_file: str | list[str] | None = None
  """Path to one csv or list of csv paths for reference motions."""
  global_obs: bool = False
  root_height_obs: bool = True
  include_root_xy: bool = True
  """Whether to include root relative x/y in discriminator observation."""
  include_root_rot: bool = True
  """Whether to include root 6D orientation in discriminator observation."""
  include_root_vel: bool = True
  """Whether to include root linear/angular velocity in discriminator observation."""
  include_projected_gravity: bool = False
  """Whether to include projected gravity in discriminator observation."""
  disc_body_pos_b_link_names: tuple[str, ...] = ()
  """Extra link positions in anchor frame, appended to disc obs."""
  include_fall_direction_obs: bool = False
  """Append one categorical 8-way fall-direction observation to disc obs."""
  motion_fall_directions: tuple[str, ...] = ()
  """Expert direction label for each motion file, in motion_file order."""
  fall_direction_confirm_steps: int = 5
  """Consecutive informative control updates required before episode locking."""
  fall_direction_min_tilt_rad: float = 0.15
  """Minimum torso tilt from upright for a reliable direction signal."""
  fall_direction_min_speed: float = 0.2
  """Minimum horizontal root speed (m/s) when tilt is not yet informative."""
  fall_direction_min_displacement: float = 0.03
  """Minimum horizontal displacement (m) used as the last direction fallback."""
  fall_direction_min_coherence: float = 0.8
  """Minimum mean unit-vector length in the confirmation window (0, 1]."""


class AMPHelper:
  """Maintains state history and computes disc_obs for an RL env. Used when cfg.amp is set."""

  def __init__(self, env: ManagerBasedRlEnv, cfg: AMPCfg) -> None:
    self._env = env
    self._cfg = cfg
    self._device = env.device
    self._num_envs = env.num_envs
    if cfg.include_fall_direction_obs:
      if cfg.fall_direction_confirm_steps < 1:
        raise ValueError("fall_direction_confirm_steps must be positive.")
      if not 0.0 < cfg.fall_direction_min_tilt_rad < math.pi / 2:
        raise ValueError("fall_direction_min_tilt_rad must be in (0, pi/2).")
      if (
        cfg.fall_direction_min_speed <= 0.0
        or cfg.fall_direction_min_displacement <= 0.0
      ):
        raise ValueError(
          "Fall direction speed/displacement thresholds must be positive."
        )
      if not 0.0 < cfg.fall_direction_min_coherence <= 1.0:
        raise ValueError("fall_direction_min_coherence must be in (0, 1].")
    self._fall_direction_min_lean = math.sin(cfg.fall_direction_min_tilt_rad)
    robot = env.scene[cfg.asset_name]
    self._robot = robot
    self._root_body_idx = robot.body_names.index(cfg.root_body_name)
    self._extra_body_idx = tuple(
      robot.body_names.index(n) for n in cfg.disc_body_pos_b_link_names
    )
    self._num_disc_body_pos_b = len(self._extra_body_idx)
    self._num_joints = robot.data.joint_pos.shape[1]
    self._default_joint_pos = robot.data.default_joint_pos.clone()
    self._disc_dim = calc_disc_obs_dim(
      cfg.num_disc_obs_steps,
      self._num_joints,
      cfg.root_height_obs,
      cfg.include_root_xy,
      cfg.include_root_rot,
      cfg.include_root_vel,
      cfg.include_projected_gravity,
      self._num_disc_body_pos_b,
      cfg.include_fall_direction_obs,
    )
    n = cfg.num_disc_obs_steps
    self._hist_root_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_root_quat = CircularBuffer(n, self._num_envs, self._device)
    self._hist_root_lin = CircularBuffer(n, self._num_envs, self._device)
    self._hist_root_ang = CircularBuffer(n, self._num_envs, self._device)
    self._hist_joint_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_joint_vel = CircularBuffer(n, self._num_envs, self._device)
    self._hist_extra_body_pos = CircularBuffer(n, self._num_envs, self._device)
    self._hist_extra_body_quat = CircularBuffer(n, self._num_envs, self._device)
    self._disc_obs_buf = torch.zeros(
      (self._num_envs, self._disc_dim), device=self._device, dtype=torch.float32
    )
    self._episode_root_pos_w = torch.zeros(
      self._num_envs, 3, device=self._device, dtype=torch.float32
    )
    self._episode_heading_cos = torch.ones(
      self._num_envs, device=self._device, dtype=torch.float32
    )
    self._episode_heading_sin = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.float32
    )
    self._direction_reference_pending = torch.ones(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._direction_reference_dirty = True
    self._fall_direction_locked = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._fall_direction_sum_xy = torch.zeros(self._num_envs, 2, device=self._device)
    self._fall_direction_count = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.long
    )
    self._fall_direction_obs = torch.zeros(
      self._num_envs, len(FALL_DIRECTION_NAMES), device=self._device
    )
    self._fall_direction_obs[:, 0] = 1.0
    self._demo_data: list[dict[str, torch.Tensor]] | None = None
    self._demo_disc_obs: torch.Tensor | None = None
    self._demo_pair_states: torch.Tensor | None = None
    self._demo_pair_next_states: torch.Tensor | None = None
    if cfg.motion_file:
      paths = (
        cfg.motion_file
        if isinstance(cfg.motion_file, (list, tuple))
        else [cfg.motion_file]
      )
      self._load_demos(paths)

  def _load_demos(self, paths: list[str]) -> None:
    """Load demos from one or more motion files (npz). Each file becomes one motion."""
    if self._cfg.include_fall_direction_obs:
      if len(self._cfg.motion_fall_directions) != len(paths):
        raise ValueError(
          "AMP motion_fall_directions must contain exactly one direction for "
          f"each motion file: got {len(self._cfg.motion_fall_directions)} labels "
          f"for {len(paths)} files."
        )
      direction_labels = _fall_direction_one_hot(
        self._cfg.motion_fall_directions, self._device
      )
    else:
      direction_labels = None

    self._demo_data = []
    for motion_index, path in enumerate(paths):
      demo = self._load_one_demo(path)
      if direction_labels is not None:
        time_steps = demo["root_pos"].shape[1]
        demo["fall_direction_obs"] = (
          direction_labels[motion_index].view(1, 1, -1).expand(1, time_steps, -1)
        )
      self._demo_data.append(demo)
    self._build_demo_cache()

  def _load_one_demo(self, path: str) -> dict[str, torch.Tensor]:
    """Load a single motion demo from .npz."""
    if path.endswith(".npz"):
      # NPZ path: same semantics as tracking MotionLoader (joint_pos, joint_vel,
      # and either root_* or body_* arrays for determining root state).
      data = np.load(path)
      joint_pos = torch.from_numpy(data["joint_pos"]).float().to(self._device)
      joint_vel = torch.from_numpy(data["joint_vel"]).float().to(self._device)
      if joint_pos.ndim == 2:
        joint_pos = joint_pos.unsqueeze(0)
        joint_vel = joint_vel.unsqueeze(0)
      T = joint_pos.shape[1]

      if "root_pos" in data:
        root_pos = torch.from_numpy(data["root_pos"]).float().to(self._device)
        root_quat = torch.from_numpy(data["root_quat"]).float().to(self._device)
        root_lin = torch.from_numpy(data["root_lin_vel"]).float().to(self._device)
        root_ang = torch.from_numpy(data["root_ang_vel"]).float().to(self._device)
        if root_pos.ndim == 2:
          root_pos = root_pos.unsqueeze(0)
          root_quat = root_quat.unsqueeze(0)
          root_lin = root_lin.unsqueeze(0)
          root_ang = root_ang.unsqueeze(0)
      elif "body_pos_w" in data:
        body_pos = np.asarray(data["body_pos_w"])
        body_quat = np.asarray(data["body_quat_w"])
        if body_pos.ndim == 2:
          body_pos = body_pos[:, np.newaxis, :]
          body_quat = body_quat[:, np.newaxis, :]
        root_idx = self._root_body_idx
        root_pos = (
          torch.from_numpy(body_pos[:, root_idx, :])
          .float()
          .to(self._device)
          .unsqueeze(0)
        )
        root_quat = (
          torch.from_numpy(body_quat[:, root_idx, :])
          .float()
          .to(self._device)
          .unsqueeze(0)
        )
        if "body_lin_vel_w" in data and "body_ang_vel_w" in data:
          body_lin = np.asarray(data["body_lin_vel_w"])
          body_ang = np.asarray(data["body_ang_vel_w"])
          if body_lin.ndim == 2:
            body_lin = body_lin[:, np.newaxis, :]
            body_ang = body_ang[:, np.newaxis, :]
          root_lin = (
            torch.from_numpy(body_lin[:, root_idx, :])
            .float()
            .to(self._device)
            .unsqueeze(0)
          )
          root_ang = (
            torch.from_numpy(body_ang[:, root_idx, :])
            .float()
            .to(self._device)
            .unsqueeze(0)
          )
        else:
          root_lin = torch.zeros(1, T, 3, device=self._device)
          root_ang = torch.zeros(1, T, 3, device=self._device)
        out: dict[str, torch.Tensor] = {
          "root_pos": root_pos,
          "root_quat": root_quat,
          "root_lin_vel": root_lin,
          "root_ang_vel": root_ang,
          "joint_pos": joint_pos,
          "joint_vel": joint_vel,
        }
        if self._cfg.disc_body_pos_b_link_names:
          body_count = body_pos.shape[1]
          if self._root_body_idx >= body_count:
            raise ValueError(
              f"Motion npz '{path}' body count ({body_count}) is smaller than "
              f"root body index {self._root_body_idx}."
            )
          for bi in self._extra_body_idx:
            if bi >= body_count:
              raise ValueError(
                f"Motion npz '{path}' body count ({body_count}) is smaller than "
                f"required disc body index {bi}."
              )
          idx_np = np.asarray(self._extra_body_idx, dtype=np.int64)
          out["extra_body_pos_w"] = (
            torch.from_numpy(body_pos[:, idx_np, :])
            .float()
            .to(self._device)
            .unsqueeze(0)
          )
          out["extra_body_quat_w"] = (
            torch.from_numpy(body_quat[:, idx_np, :])
            .float()
            .to(self._device)
            .unsqueeze(0)
          )
        return out
      else:
        root_pos = torch.zeros(1, T, 3, device=self._device)
        root_quat = torch.zeros(1, T, 4, device=self._device)
        root_quat[:, :, 0] = 1.0
        root_lin = torch.zeros(1, T, 3, device=self._device)
        root_ang = torch.zeros(1, T, 3, device=self._device)

      out = {
        "root_pos": root_pos,
        "root_quat": root_quat,
        "root_lin_vel": root_lin,
        "root_ang_vel": root_ang,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
      }
      if self._cfg.disc_body_pos_b_link_names:
        raise ValueError(
          f"Motion npz '{path}' must provide body_pos_w/body_quat_w when "
          "disc_body_pos_b_link_names is configured."
        )
      return out
    raise ValueError(
      f"Unsupported AMP motion file format '{path}'. Expected a .npz file."
    )

  def _pad_demo_motion(
    self, demo: dict[str, torch.Tensor], target_len: int
  ) -> dict[str, torch.Tensor]:
    """Pad a demo to target_len by repeating the last frame."""
    cur_len = demo["root_pos"].shape[1]
    if cur_len >= target_len:
      return demo
    pad = target_len - cur_len
    padded: dict[str, torch.Tensor] = {}
    for key, value in demo.items():
      padded[key] = torch.cat(
        [value, value[:, -1:].expand(-1, pad, value.shape[-1])], dim=1
      )
    return padded

  def _compute_demo_disc_sequence(
    self, demo: dict[str, torch.Tensor]
  ) -> torch.Tensor | None:
    """Precompute disc_obs for every valid history window in one demo."""
    n_steps = self._cfg.num_disc_obs_steps
    demo = self._pad_demo_motion(demo, n_steps)
    seq_len = demo["root_pos"].shape[1]
    num_windows = seq_len - n_steps + 1
    if num_windows <= 0:
      return None

    def _windows(x: torch.Tensor) -> torch.Tensor:
      return torch.stack([x[0, i : i + n_steps] for i in range(num_windows)], dim=0)

    root_pos = _windows(demo["root_pos"])
    root_quat = _windows(demo["root_quat"])
    root_lin = _windows(demo["root_lin_vel"])
    root_ang = _windows(demo["root_ang_vel"])
    default_joint_pos = self._default_joint_pos[0:1]
    joint_pos = _windows(demo["joint_pos"] - default_joint_pos.unsqueeze(1))
    joint_vel = _windows(demo["joint_vel"])
    extra_kw: dict[str, torch.Tensor | None] = {
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    if self._cfg.disc_body_pos_b_link_names:
      extra_kw["extra_body_pos_w"] = _windows(demo["extra_body_pos_w"])
      extra_kw["extra_body_quat_w"] = _windows(demo["extra_body_quat_w"])
    fall_direction_obs = None
    if self._cfg.include_fall_direction_obs:
      # The categorical expert label is constant over a motion. Append it once
      # per discriminator sample rather than once per history frame.
      fall_direction_obs = _windows(demo["fall_direction_obs"])[:, -1]
    return compute_disc_obs(
      ref_root_pos=root_pos[:, -1],
      ref_root_quat=root_quat[:, -1],
      root_pos=root_pos,
      root_quat=root_quat,
      root_lin_vel=root_lin,
      root_ang_vel=root_ang,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      global_obs=self._cfg.global_obs,
      root_height_obs=self._cfg.root_height_obs,
      include_root_xy=self._cfg.include_root_xy,
      include_root_rot=self._cfg.include_root_rot,
      include_root_vel=self._cfg.include_root_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      fall_direction_obs=fall_direction_obs,
      **extra_kw,
    )

  def _build_demo_cache(self) -> None:
    """Precompute demo discriminator observations once at load time."""
    if not self._demo_data:
      self._demo_disc_obs = None
      self._demo_pair_states = None
      self._demo_pair_next_states = None
      return

    disc_sequences: list[torch.Tensor] = []
    pair_states: list[torch.Tensor] = []
    pair_next_states: list[torch.Tensor] = []
    for demo in self._demo_data:
      disc_seq = self._compute_demo_disc_sequence(demo)
      if disc_seq is None or disc_seq.numel() == 0:
        continue
      disc_sequences.append(disc_seq)
      if disc_seq.shape[0] > 1:
        pair_states.append(disc_seq[:-1])
        pair_next_states.append(disc_seq[1:])

    self._demo_disc_obs = torch.cat(disc_sequences, dim=0) if disc_sequences else None
    self._demo_pair_states = torch.cat(pair_states, dim=0) if pair_states else None
    self._demo_pair_next_states = (
      torch.cat(pair_next_states, dim=0) if pair_next_states else None
    )

  def update(self, env_ids: torch.Tensor | None = None) -> None:
    """Append current robot state to history and update disc_obs buffer."""
    r = self._robot.data
    root_pos = r.body_link_pos_w[:, self._root_body_idx]
    root_quat = r.body_link_quat_w[:, self._root_body_idx]
    root_lin = r.body_link_lin_vel_w[:, self._root_body_idx]
    root_ang = r.body_link_ang_vel_w[:, self._root_body_idx]
    if self._cfg.include_fall_direction_obs and self._direction_reference_dirty:
      # Reset already tells us whether references need refreshing. A Python
      # flag avoids synchronizing CUDA with a per-step ``if pending.any()``.
      pending = self._direction_reference_pending
      self._episode_root_pos_w.copy_(
        torch.where(pending[:, None], root_pos, self._episode_root_pos_w)
      )
      yaw = _yaw_from_quat(root_quat)
      self._episode_heading_cos.copy_(
        torch.where(pending, torch.cos(yaw), self._episode_heading_cos)
      )
      self._episode_heading_sin.copy_(
        torch.where(pending, torch.sin(yaw), self._episode_heading_sin)
      )
      pending.zero_()
      self._direction_reference_dirty = False
    jpos = r.joint_pos - self._default_joint_pos
    jvel = r.joint_vel
    self._hist_root_pos.append(root_pos)
    self._hist_root_quat.append(root_quat)
    self._hist_root_lin.append(root_lin)
    self._hist_root_ang.append(root_ang)
    self._hist_joint_pos.append(jpos)
    self._hist_joint_vel.append(jvel)
    if self._cfg.disc_body_pos_b_link_names:
      idx = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      extra_pos = r.body_link_pos_w.index_select(1, idx)
      extra_quat = r.body_link_quat_w.index_select(1, idx)
      self._hist_extra_body_pos.append(extra_pos)
      self._hist_extra_body_quat.append(extra_quat)
    if not self._hist_root_pos.is_initialized:
      return
    buf = self._hist_root_pos.buffer
    t = buf.shape[1]
    if t < self._cfg.num_disc_obs_steps:
      return
    ref_pos = self._hist_root_pos.buffer[:, -1]
    ref_quat = self._hist_root_quat.buffer[:, -1]
    extra_kw: dict[str, torch.Tensor | None] = {
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    if self._cfg.disc_body_pos_b_link_names:
      extra_kw["extra_body_pos_w"] = self._hist_extra_body_pos.buffer
      extra_kw["extra_body_quat_w"] = self._hist_extra_body_quat.buffer
    fall_direction_obs = None
    if self._cfg.include_fall_direction_obs:
      fall_direction_obs = self._compute_policy_fall_direction(
        root_pos, root_quat, root_lin
      )
    self._disc_obs_buf[:] = compute_disc_obs(
      ref_root_pos=ref_pos,
      ref_root_quat=ref_quat,
      root_pos=self._hist_root_pos.buffer,
      root_quat=self._hist_root_quat.buffer,
      root_lin_vel=self._hist_root_lin.buffer,
      root_ang_vel=self._hist_root_ang.buffer,
      joint_pos=self._hist_joint_pos.buffer,
      joint_vel=self._hist_joint_vel.buffer,
      global_obs=self._cfg.global_obs,
      root_height_obs=self._cfg.root_height_obs,
      include_root_xy=self._cfg.include_root_xy,
      include_root_rot=self._cfg.include_root_rot,
      include_root_vel=self._cfg.include_root_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      fall_direction_obs=fall_direction_obs,
      **extra_kw,
    )

  def _compute_policy_fall_direction(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    """Confirm the initial fall direction, then keep it fixed until reset.

    Average unit directions before quantization, so neighboring categories can
    alternate at a 22.5-degree boundary without preventing confirmation.
    """
    up_axis_b = torch.zeros_like(root_pos_w)
    up_axis_b[:, 2] = 1.0
    up_axis_w = quat_apply(root_quat_w, up_axis_b)
    displacement_w = root_pos_w - self._episode_root_pos_w

    cos_yaw = self._episode_heading_cos
    sin_yaw = self._episode_heading_sin

    def _to_episode_xy(vector_w: torch.Tensor) -> torch.Tensor:
      local_x = cos_yaw * vector_w[:, 0] + sin_yaw * vector_w[:, 1]
      local_y = -sin_yaw * vector_w[:, 0] + cos_yaw * vector_w[:, 1]
      return torch.stack([local_x, local_y], dim=-1)

    lean_xy = _to_episode_xy(up_axis_w)
    velocity_xy = _to_episode_xy(root_lin_vel_w)
    displacement_xy = _to_episode_xy(displacement_w)

    # Only sufficiently strong signals may confirm a direction. In particular,
    # neither tiny reset tilts nor the stationary forward fallback can lock it.
    lean_clear = (
      torch.linalg.vector_norm(lean_xy, dim=-1) >= self._fall_direction_min_lean
    )
    velocity_clear = (
      torch.linalg.vector_norm(velocity_xy, dim=-1)
      >= self._cfg.fall_direction_min_speed
    )
    displacement_clear = (
      torch.linalg.vector_norm(displacement_xy, dim=-1)
      >= self._cfg.fall_direction_min_displacement
    )
    motion_xy = torch.where(velocity_clear[:, None], velocity_xy, displacement_xy)
    direction_xy = torch.where(lean_clear[:, None], lean_xy, motion_xy)
    informative = lean_clear | velocity_clear | displacement_clear
    collecting = informative & ~self._fall_direction_locked
    unit_xy = direction_xy / torch.linalg.vector_norm(
      direction_xy, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    self._fall_direction_sum_xy.copy_(
      torch.where(collecting[:, None], self._fall_direction_sum_xy + unit_xy, 0.0)
    )
    self._fall_direction_count.copy_(
      torch.where(collecting, self._fall_direction_count + 1, 0)
    )
    sum_norm = torch.linalg.vector_norm(self._fall_direction_sum_xy, dim=-1)
    coherence = sum_norm / self._fall_direction_count.clamp_min(1)
    window_full = self._fall_direction_count >= self._cfg.fall_direction_confirm_steps
    confirmed = window_full & (coherence >= self._cfg.fall_direction_min_coherence)

    # Until confirmation this is a provisional one-hot label. Only the discrete
    # result reaches D; the continuous evidence remains private to this helper.
    direction_xy = torch.where(
      (collecting & (sum_norm > 1e-6))[:, None],
      self._fall_direction_sum_xy,
      direction_xy,
    )
    no_direction = torch.linalg.vector_norm(direction_xy, dim=-1) < 1e-6
    direction_xy[:, 0] = torch.where(no_direction, 1.0, direction_xy[:, 0])
    self._fall_direction_obs.copy_(
      torch.where(
        self._fall_direction_locked[:, None],
        self._fall_direction_obs,
        _quantize_fall_direction(direction_xy),
      )
    )
    self._fall_direction_locked |= confirmed

    # Contradictory evidence (e.g. opposite directions) starts a fresh short
    # window rather than keeping stale evidence across the whole episode.
    retry = window_full & ~confirmed
    self._fall_direction_sum_xy.copy_(
      torch.where(retry[:, None], 0.0, self._fall_direction_sum_xy)
    )
    self._fall_direction_count.copy_(torch.where(retry, 0, self._fall_direction_count))
    return self._fall_direction_obs

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset history for given envs (or all)."""
    self._direction_reference_dirty = True
    ids = slice(None) if env_ids is None else env_ids
    self._fall_direction_locked[ids] = False
    self._fall_direction_sum_xy[ids] = 0.0
    self._fall_direction_count[ids] = 0
    self._fall_direction_obs[ids] = 0.0
    self._fall_direction_obs[ids, 0] = 1.0
    if env_ids is None:
      self._direction_reference_pending[:] = True
      self._hist_root_pos.reset(None)
      self._hist_root_quat.reset(None)
      self._hist_root_lin.reset(None)
      self._hist_root_ang.reset(None)
      self._hist_joint_pos.reset(None)
      self._hist_joint_vel.reset(None)
      self._hist_extra_body_pos.reset(None)
      self._hist_extra_body_quat.reset(None)
    else:
      self._direction_reference_pending[env_ids] = True
      self._hist_root_pos.reset(env_ids)
      self._hist_root_quat.reset(env_ids)
      self._hist_root_lin.reset(env_ids)
      self._hist_root_ang.reset(env_ids)
      self._hist_joint_pos.reset(env_ids)
      self._hist_joint_vel.reset(env_ids)
      self._hist_extra_body_pos.reset(env_ids)
      self._hist_extra_body_quat.reset(env_ids)

  def get_disc_obs(self) -> torch.Tensor:
    """Current disc_obs. Shape (num_envs, disc_dim)."""
    return self._disc_obs_buf

  def get_disc_obs_space(self) -> Box:
    """Gym-style Box space for disc_obs."""
    return Box(
      shape=(self._disc_dim,),
      low=-float("inf"),
      high=float("inf"),
      dtype="float32",
    )

  def fetch_disc_obs_demo(self, num_samples: int) -> torch.Tensor:
    """Sample num_samples demo disc_obs for discriminator training. Shape (num_samples, disc_dim)."""
    if self._demo_disc_obs is not None and self._demo_disc_obs.shape[0] > 0:
      indices = torch.randint(
        0, self._demo_disc_obs.shape[0], (num_samples,), device=self._device
      )
      return self._demo_disc_obs[indices]

    n_steps = self._cfg.num_disc_obs_steps
    # No motion file: synthetic standing (default pose, zero vel), use env 0 as ref
    root_pos = (
      self._robot.data.body_link_pos_w[0:1, self._root_body_idx]
      .unsqueeze(1)
      .expand(1, n_steps, 3)
    )
    root_quat = (
      self._robot.data.body_link_quat_w[0:1, self._root_body_idx]
      .unsqueeze(1)
      .expand(1, n_steps, 4)
    )
    root_lin = torch.zeros(1, n_steps, 3, device=self._device)
    root_ang = torch.zeros(1, n_steps, 3, device=self._device)
    joint_pos = torch.zeros(1, n_steps, self._num_joints, device=self._device)
    joint_vel = torch.zeros(1, n_steps, self._num_joints, device=self._device)
    ref_pos = root_pos[:, -1]
    ref_quat = root_quat[:, -1]
    extra_kw: dict[str, torch.Tensor | None] = {
      "extra_body_pos_w": None,
      "extra_body_quat_w": None,
    }
    k = self._num_disc_body_pos_b
    if k:
      r0 = self._robot.data
      idx = torch.tensor(self._extra_body_idx, device=self._device, dtype=torch.long)
      extra_kw["extra_body_pos_w"] = (
        r0.body_link_pos_w[0:1]
        .index_select(1, idx)
        .unsqueeze(1)
        .expand(1, n_steps, k, 3)
      )
      extra_kw["extra_body_quat_w"] = (
        r0.body_link_quat_w[0:1]
        .index_select(1, idx)
        .unsqueeze(1)
        .expand(1, n_steps, k, 4)
      )
    one = compute_disc_obs(
      ref_root_pos=ref_pos,
      ref_root_quat=ref_quat,
      root_pos=root_pos,
      root_quat=root_quat,
      root_lin_vel=root_lin,
      root_ang_vel=root_ang,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      global_obs=self._cfg.global_obs,
      root_height_obs=self._cfg.root_height_obs,
      include_root_xy=self._cfg.include_root_xy,
      include_root_rot=self._cfg.include_root_rot,
      include_root_vel=self._cfg.include_root_vel,
      include_projected_gravity=self._cfg.include_projected_gravity,
      fall_direction_obs=(
        _fall_direction_one_hot(("forward",), self._device)
        if self._cfg.include_fall_direction_obs
        else None
      ),
      **extra_kw,
    )
    return one.expand(num_samples, -1)

  def fetch_disc_obs_demo_pairs(
    self, num_pairs: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample num_pairs consecutive (s_t, s_{t+1}) from cached demo pairs."""
    if (
      self._demo_pair_states is not None
      and self._demo_pair_next_states is not None
      and self._demo_pair_states.shape[0] > 0
    ):
      indices = torch.randint(
        0, self._demo_pair_states.shape[0], (num_pairs,), device=self._device
      )
      return self._demo_pair_states[indices], self._demo_pair_next_states[indices]

    if self._demo_data is None or len(self._demo_data) == 0:
      single = self.fetch_disc_obs_demo(1)
      return single.expand(num_pairs, -1), single.expand(num_pairs, -1)
    single = self.fetch_disc_obs_demo(1)
    return single.expand(num_pairs, -1), single.expand(num_pairs, -1)
