from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
  quat_apply_inverse,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)

def projected_gravity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Projected gravity vector in robot anchor frame.

  Returns shape (N, 3).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  g_w = env.scene[command.cfg.asset_name].data.gravity_vec_w  # (N, 3)
  g_robot_b = quat_apply_inverse(command.robot_anchor_quat_w, g_w)
  return g_robot_b


def projected_gravity_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Difference between projected gravity (in anchor frames) of reference and robot.

  Returns shape (N, 3).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  g_w = env.scene[command.cfg.asset_name].data.gravity_vec_w  # (N, 3)
  # print(g_w)
  g_ref_b = quat_apply_inverse(command.anchor_quat_w, g_w)
  g_robot_b = quat_apply_inverse(command.robot_anchor_quat_w, g_w)
  return g_robot_b - g_ref_b


def reference_torso_ori_multi_frame(
  env: ManagerBasedRlEnv, command_name: str, horizon: int = 10
) -> torch.Tensor:
  """Reference anchor (torso) orientation over next `horizon` frames (exclude current).

  Returns concatenated first two columns of rotation matrices per future frame:
  shape (N, horizon * 6).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  t = command.time_steps  # (N,)
  T = command.motion.time_step_total
  # future indices t+1..t+horizon
  device = t.device
  future_steps = torch.arange(1, horizon + 1, device=device).unsqueeze(0)  # (1,H)
  idx = torch.clamp(t.unsqueeze(1) + future_steps, 0, T - 1)  # (N,H)
  # Gather future quats at anchor body index; command.motion.body_quat_w: (T, B, 4)
  anchor_idx = command.motion_anchor_body_index
  future_quat = command.motion.body_quat_w[idx, anchor_idx]  # (N,H,4)
  # Normalize quaternion to avoid invalid rotation matrices
  norm = torch.linalg.vector_norm(future_quat, dim=-1, keepdim=True)
  future_quat = future_quat / (norm + 1e-9)
  mat = matrix_from_quat(future_quat)  # (N,H,3,3)
  cols = mat[..., :2]  # (N,H,3,2)
  return cols.reshape(env.num_envs, -1)


def future_frames_generated_commands(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """返回未来9帧的关节位置和速度目标，按帧顺序堆叠（第一帧所有数据，第二帧所有数据...）"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.future_frames_command


def reference_anchor_future_xy_10(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """参考全局 anchor 未来10帧的 XY 位置（不含当前帧），输出 [N, 10*2]."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  t = command.time_steps  # [N]
  T = command.motion.time_step_total
  steps = torch.arange(1, 11, device=t.device).unsqueeze(
    0
  )  # [1,10] -> future 10 frames
  idx = torch.clamp(t.unsqueeze(1) + steps, 0, T - 1)  # [N,10]
  # 取参考的 anchor body 在全局的位姿序列；body_pos_w: (T,B,3)
  anchor_idx = command.motion_anchor_body_index
  pos_w = command.motion.body_pos_w[idx, anchor_idx, :]  # [N,10,3]
  xy = pos_w[..., :2]  # [N,10,2]
  return xy.reshape(env.num_envs, -1)


def reference_anchor_ori_current_future_10(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """参考 anchor（躯干）orientation 当前帧+未来帧共10步，输出 [N, 10*6].

  返回当前帧和未来9帧的anchor orientation，每帧用旋转矩阵的前两列表示（6个值）。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  t = command.time_steps  # [N]
  T = command.motion.time_step_total
  # 当前帧+未来9帧 = 10帧
  steps = torch.arange(0, 10, device=t.device).unsqueeze(0)  # [1,10]
  idx = torch.clamp(t.unsqueeze(1) + steps, 0, T - 1)  # [N,10]
  # 取参考的 anchor body 在全局的姿态序列；body_quat_w: (T,B,4)
  anchor_idx = command.motion_anchor_body_index
  future_quat = command.motion.body_quat_w[idx, anchor_idx]  # [N,10,4]
  # 归一化四元数，避免无效的旋转矩阵
  norm = torch.linalg.vector_norm(future_quat, dim=-1, keepdim=True)
  future_quat = future_quat / (norm + 1e-9)
  mat = matrix_from_quat(future_quat)  # [N,10,3,3]
  cols = mat[..., :2]  # [N,10,3,2] - 取前两列
  return cols.reshape(env.num_envs, -1)  # [N, 10*6]


def reference_anchor_quat_current_future_10(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """参考 anchor（躯干）四元数 当前帧+未来帧共10步，输出 [N, 10*4].

  返回当前帧和未来9帧的anchor四元数（w, x, y, z 顺序）。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  t = command.time_steps  # [N]
  T = command.motion.time_step_total
  # 当前帧+未来9帧 = 10帧
  steps = torch.arange(0, 10, device=t.device).unsqueeze(0)  # [1,10]
  idx = torch.clamp(t.unsqueeze(1) + steps, 0, T - 1)  # [N,10]
  # 取参考的 anchor body 在全局的姿态四元数序列；body_quat_w: (T,B,4)
  anchor_idx = command.motion_anchor_body_index
  quat = command.motion.body_quat_w[idx, anchor_idx]  # [N,10,4]
  # 归一化四元数，保证数值稳定
  norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
  quat = quat / (norm + 1e-9)
  return quat.reshape(env.num_envs, -1)  # [N, 10*4]


def reference_feet_rel_pos_current_future_10(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """参考脚部相对位置（当前+未来共10帧），每帧 v = pL - pR（世界系），输出 [N, 10*3]."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  names = command.cfg.body_names
  device = (
    command.device
    if hasattr(command, "device")
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
  )
  try:
    li = names.index("LINK_ANKLE_ROLL_L")
    ri = names.index("LINK_ANKLE_ROLL_R")
  except ValueError:
    return torch.zeros((env.num_envs, 10 * 3), device=device)
  t = command.time_steps  # [N]
  T = command.motion.time_step_total
  steps = torch.arange(0, 10, device=t.device).unsqueeze(0)  # [1,10]
  idx = torch.clamp(t.unsqueeze(1) + steps, 0, T - 1)  # [N,10]
  pL = command.motion.body_pos_w[idx, li, :]  # [N,10,3]
  pR = command.motion.body_pos_w[idx, ri, :]  # [N,10,3]
  v = pL - pR  # [N,10,3]
  return v.reshape(env.num_envs, -1)


def history_observations(
  env: ManagerBasedRlEnv, history_steps: int = 4
) -> torch.Tensor:
  """获取过去N步的观测历史，包含指定的观测分量。

  历史观测包含以下分量：
  - base_ang_vel: 基座角速度
  - joint_pos: 关节位置（相对）
  - joint_vel: 关节速度（相对）
  - actions: 上一步动作
  - projected_gravity_error: 投影重力误差

  Args:
    env: 环境实例
    history_steps: 历史步数，默认4步

  Returns:
    shape (N, history_steps * obs_dim) 的张量
  """
  # 获取当前观测
  current_obs = _get_current_history_obs(env)

  # 初始化历史缓存和上一步观测缓存（如果尚未初始化）
  if env._obs_history is None or env._prev_obs is None:
    obs_dim = current_obs.shape[1]
    env._obs_history = torch.zeros(
      (env.num_envs, history_steps, obs_dim), device=current_obs.device
    )
    env._prev_obs = torch.zeros_like(current_obs)
    # 初始化时，用当前观测回填所有历史槽位
    env._obs_history[:] = current_obs.unsqueeze(1).repeat(1, history_steps, 1)
    return env._obs_history.reshape(env.num_envs, -1)

  # 检查是否有环境需要重置历史
  reset_mask = env.episode_length_buf == 0

  # 对于重置的环境，用当前观测回填所有历史槽位
  if reset_mask.any():
    env._obs_history[reset_mask] = current_obs[reset_mask].unsqueeze(1).repeat(
      1, history_steps, 1
    )

  # 对于非重置的环境，将上一步的观测加入历史
  non_reset_mask = ~reset_mask
  if non_reset_mask.any():
    # 将历史向前移动：丢弃最旧的，添加上一步的观测
    env._obs_history[non_reset_mask, :-1] = env._obs_history[non_reset_mask, 1:].clone()
    env._obs_history[non_reset_mask, -1] = env._prev_obs[non_reset_mask]

  # 更新上一步观测为当前观测（为下一次调用准备）
  env._prev_obs = current_obs.clone()

  # 返回历史观测（不包含当前观测）
  return env._obs_history.reshape(env.num_envs, -1)


def _get_current_history_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """获取当前步的历史观测分量（不包含历史）

  包含以下观测分量：
  - base_ang_vel: 基座角速度（带噪声 -0.2 到 0.2）
  - joint_pos: 关节位置（相对，带噪声 -0.02 到 0.02）
  - joint_vel: 关节速度（相对，带噪声 -0.5 到 0.5）
  - actions: 上一步动作（无噪声）
  - projected_gravity_error: 投影重力误差（带噪声 -0.01 到 0.01）
  """
  obs_list = []

  asset = env.scene["robot"]

  # 1. 基座角速度（带噪声 -0.2 到 0.2）
  base_ang_vel = asset.data.root_link_ang_vel_b
  base_ang_vel_noise = (
    torch.rand(base_ang_vel.shape, device=base_ang_vel.device) * 0.4 - 0.2
  )
  base_ang_vel_noisy = base_ang_vel + base_ang_vel_noise
  obs_list.append(base_ang_vel_noisy)

  # 2. 关节位置（相对默认位置，带噪声 -0.02 到 0.02）
  joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
  joint_pos_noise = torch.rand(joint_pos.shape, device=joint_pos.device) * 0.04 - 0.02
  joint_pos_noisy = joint_pos + joint_pos_noise
  obs_list.append(joint_pos_noisy)

  # 3. 关节速度（相对默认速度，带噪声 -0.5 到 0.5）
  joint_vel = asset.data.joint_vel - asset.data.default_joint_vel
  joint_vel_noise = torch.rand(joint_vel.shape, device=joint_vel.device) * 1.0 - 0.5
  joint_vel_noisy = joint_vel + joint_vel_noise
  obs_list.append(joint_vel_noisy)

  # 4. 上一步动作（无噪声）
  obs_list.append(env.action_manager.action)

  # 5. 投影重力误差（带噪声 -0.01 到 0.01）
  proj_gravity_error = projected_gravity_error(env, "motion")
  proj_gravity_error_noise = (
    torch.rand(proj_gravity_error.shape, device=proj_gravity_error.device) * 0.02 - 0.01
  )
  proj_gravity_error_noisy = proj_gravity_error + proj_gravity_error_noise
  obs_list.append(proj_gravity_error_noisy)

  return torch.cat(obs_list, dim=1)


def _get_current_basic_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """获取当前步的基础观测量（不包含历史）- 保持向后兼容"""
  return _get_current_history_obs(env)


def generated_commands_with_scale(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_scale: float = 1.0,
  vel_scale: float = 0.05,
) -> torch.Tensor:
  """返回当前帧的关节位置和速度目标，分别应用scale。

  ``qd_mask``（若配置）仅乘在速度半段 ``joint_vel`` 上，不乘 ``joint_pos``。

  Args:
    env: 环境实例
    command_name: 命令名称
    pos_scale: 位置的缩放系数，默认1.0
    vel_scale: 速度的缩放系数，默认0.05

  Returns:
    shape (N, num_joints * 2) 的张量，前半部分是位置，后半部分是速度
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_pos = command.joint_pos * pos_scale
  joint_vel = command.apply_qd_mask_to_vel(command.joint_vel * vel_scale)
  return torch.cat([joint_pos, joint_vel], dim=1)


def future_frames_generated_commands_with_scale(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_scale: float = 1.0,
  vel_scale: float = 0.05,
) -> torch.Tensor:
  """返回未来9帧的关节位置和速度目标，分别应用scale，按帧顺序堆叠。

  ``qd_mask``（若配置）仅乘在每帧的速度部分，不乘位置部分。

  Args:
    env: 环境实例
    command_name: 命令名称
    pos_scale: 位置的缩放系数，默认1.0
    vel_scale: 速度的缩放系数，默认0.05

  Returns:
    shape (N, 9 * num_joints * 2) 的张量
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  # 获取未来9帧的索引
  future_steps = torch.arange(1, 10, device=command.time_steps.device).unsqueeze(0)
  future_indices = command.time_steps.unsqueeze(1) + future_steps

  # 处理边界情况
  max_valid_index = command.motion.time_step_total - 1
  future_indices = torch.clamp(future_indices, 0, max_valid_index)

  # 获取未来9帧的关节位置和速度
  future_pos_frames = command.motion.joint_pos[future_indices] * pos_scale
  future_vel_frames = command.apply_qd_mask_to_vel(
    command.motion.joint_vel[future_indices] * vel_scale
  )

  # 按帧顺序堆叠
  frame_data_list = []
  for frame_idx in range(9):
    frame_pos = future_pos_frames[:, frame_idx, :]
    frame_vel = future_vel_frames[:, frame_idx, :]
    frame_data = torch.cat([frame_pos, frame_vel], dim=1)
    frame_data_list.append(frame_data)

  return torch.cat(frame_data_list, dim=1)


##
# Residual action support observations.
##


def joint_pos_error_from_motion(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Get joint position error (current - reference) for residual mode.

  This is the tracking error that the policy should correct.

  Args:
    env: The environment
    command_name: Name of the motion command

  Returns:
    torch.Tensor: Joint position error [num_envs, num_joints]
                  Positive = current joint position > reference
                  Negative = current joint position < reference
  """
  from mjlab.entity import Entity

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene["robot"]

  # Current joint positions
  current_joint_pos = asset.data.joint_pos

  # Reference joint positions from motion
  reference_joint_pos = command.joint_pos

  # Return tracking error: current - reference
  return current_joint_pos - reference_joint_pos


def joint_vel_error_from_motion(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Get joint velocity error (current - reference) for residual mode.

  Args:
    env: The environment
    command_name: Name of the motion command

  Returns:
    torch.Tensor: Joint velocity error [num_envs, num_joints]
  """
  from mjlab.entity import Entity

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset: Entity = env.scene["robot"]

  # Current joint velocities
  current_joint_vel = asset.data.joint_vel

  # Reference joint velocities from motion
  reference_joint_vel = command.joint_vel

  # Return tracking error: current - reference
  return current_joint_vel - reference_joint_vel


def base_height_diff_10frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Compute the difference between current robot base height and reference base height
  for current frame and next 4 future frames (total 5 frames).

  Returns:
    torch.Tensor: Shape [num_envs, 5] containing height differences
                  (robot_z - reference_z) for current + 4 future frames
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  # Get current robot base z position
  robot_base_z = command.robot_anchor_pos_w[:, 2:3]  # [N, 1]

  # Get current frame and future 4 frames (total 5 frames)
  current_time = command.time_steps  # [N]
  future_steps = torch.arange(0, 10, device=current_time.device).unsqueeze(0)  # [1, 5] (0,1,2,3,4)
  frame_indices = current_time.unsqueeze(1) + future_steps  # [N, 5]

  # Handle boundary: clamp to valid indices
  max_valid_index = command.motion.time_step_total - 1
  frame_indices = torch.clamp(frame_indices, 0, max_valid_index)

  # Get reference base positions for these frames
  # motion.body_pos_w shape: [total_frames, num_bodies, 3]
  # Need to add environment origin offset to match robot coordinates
  ref_body_pos_w = command.motion.body_pos_w[frame_indices, command.motion_anchor_body_index] + command._env.scene.env_origins.unsqueeze(1)  # [N, 5, 3]
  ref_base_z = ref_body_pos_w[:, :, 2]  # [N, 5]
  height_diffs = ref_base_z 
  # Compute differences: robot_z - reference_z
  # height_diffs = robot_base_z - ref_base_z  # [N, 5]

  return height_diffs.view(env.num_envs, -1)  # [N, 5]
