"""Resample recorded motion states in physical time, before device upload."""

from __future__ import annotations

import math

import numpy as np


def _slerp(a: np.ndarray, b: np.ndarray, fraction: np.ndarray) -> np.ndarray:
  """Vectorized shortest-arc interpolation of wxyz quaternions."""
  norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
  norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
  if (
    not np.isfinite(norm_a).all()
    or not np.isfinite(norm_b).all()
    or (norm_a < 1e-8).any()
    or (norm_b < 1e-8).any()
  ):
    raise ValueError("Motion contains invalid quaternions.")
  a, b = a / norm_a, b / norm_b
  dot = np.sum(a * b, axis=-1, keepdims=True)
  b = np.where(dot < 0.0, -b, b)
  dot = np.clip(np.abs(dot), 0.0, 1.0)
  angle = np.arccos(dot)
  sin_angle = np.maximum(np.sin(angle), 1e-8)
  spherical = (
    np.sin((1.0 - fraction) * angle) * a
    + np.sin(fraction * angle) * b
  ) / sin_angle
  linear = (1.0 - fraction) * a + fraction * b
  result = np.where(dot > 0.9995, linear, spherical)
  return result / np.linalg.norm(result, axis=-1, keepdims=True)


def resample_motion_arrays(
  data: dict[str, np.ndarray], target_dt: float
) -> dict[str, np.ndarray]:
  """Align NPZ states to the policy clock without changing motion speed.

  Interpolate positions and recorded velocities linearly and orientations using
  SLERP. Velocities retain their physical units; they are not scaled by the fps
  ratio. Output times are k * target_dt within the recording: a final fractional
  interval is omitted rather than stretching time or extrapolating the motion.
  Supports joint/root arrays [T, D] or [1, T, D], and body arrays [T, B, D].
  """
  if "fps" not in data:
    raise ValueError("AMP motion must provide fps to align expert and policy time.")
  fps_values = np.asarray(data["fps"]).reshape(-1)
  if fps_values.size != 1:
    raise ValueError("Motion fps must contain exactly one value.")
  source_fps = float(fps_values[0])
  if not math.isfinite(source_fps) or source_fps <= 0.0:
    raise ValueError("Motion fps must be finite and positive.")
  if not math.isfinite(target_dt) or target_dt <= 0.0:
    raise ValueError("Target motion timestep must be finite and positive.")

  joint_pos = data["joint_pos"]
  if joint_pos.ndim not in (2, 3) or (
    joint_pos.ndim == 3 and joint_pos.shape[0] != 1
  ):
    raise ValueError("Motion joint_pos must have shape [T, J] or [1, T, J].")
  num_frames = joint_pos.shape[-2]
  if num_frames == 0:
    raise ValueError("Motion must contain at least one frame.")
  if math.isclose(source_fps * target_dt, 1.0, rel_tol=1e-7):
    return data

  duration = (num_frames - 1) / source_fps
  target_count = math.floor(duration / target_dt + 1e-9) + 1
  source_index = np.minimum(
    np.arange(target_count, dtype=np.float64) * target_dt * source_fps,
    num_frames - 1,
  )
  left = np.floor(source_index).astype(np.int64)
  right = np.minimum(left + 1, num_frames - 1)
  result = dict(data)
  for key in (
    "joint_pos", "joint_vel", "root_pos", "root_quat",
    "root_lin_vel", "root_ang_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
  ):
    if key not in data:
      continue
    value = np.asarray(data[key])
    axis = 1 if not key.startswith("body_") and value.ndim == 3 else 0
    frames = np.moveaxis(value, axis, 0)
    if frames.shape[0] != num_frames:
      raise ValueError(f"Motion {key} frame count does not match joint_pos.")
    fraction = (source_index - left).reshape(
      (target_count,) + (1,) * (frames.ndim - 1)
    )
    a, b = frames[left], frames[right]
    if "quat" in key:
      if frames.shape[-1] != 4:
        raise ValueError(f"Motion {key} must contain wxyz quaternions.")
      interpolated = _slerp(a, b, fraction)
    else:
      interpolated = a + fraction * (b - a)
    result[key] = np.moveaxis(
      interpolated.astype(np.result_type(value.dtype, np.float32)), 0, axis
    )
  result["fps"] = np.asarray(1.0 / target_dt)
  return result
