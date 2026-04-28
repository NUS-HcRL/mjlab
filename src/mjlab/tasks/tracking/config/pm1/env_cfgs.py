"""PM1 flat tracking environment configurations."""

from pathlib import Path

from mjlab.asset_zoo.robots import (
  PM_ACTION_SCALE,
  PM_LOWER_BODY_JOINT_NAMES,
  PM_MAX_LOWER_BODY_TORQUE,
  PM_QD_MASK,
  PM_ROBOT_CFG,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 护具 map 数据目录（默认站立系下的查表文件放于此，reward 中力衰减可引用）
PROTECTOR_MAP_DIR = Path(__file__).resolve().parent / "protector_map"

def pm1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """创建 PM1 平地跟踪任务配置。
  
  Args:
    has_state_estimation: 如果为 True，包含 base_lin_vel 和 motion_anchor_pos_b 观测。
                          如果为 False，移除这些观测（用于没有状态估计的系统）。
    play: 如果为 True，配置为播放模式（无限episode，无噪声，无随机化）。
  
  Returns:
    ManagerBasedRlEnvCfg: 配置好的 PM1 跟踪任务环境。
  """
  cfg = make_tracking_env_cfg()

  # 设置 PM1 机器人配置
  cfg.scene.entities = {"robot": PM_ROBOT_CFG}

  ##
  # 接触传感器配置
  ##
  
  # PM1 机器人自碰撞检测
  # 检测机器人各身体部件之间的碰撞
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # 检测所有机器人身体部件与地面的最大接触力
  # 排除 ankle_pitch 和 ankle_roll（脚部预期会接触地面）
  body_contact_force_cfg = ContactSensorCfg(
    name="body_contact_force",
    primary=ContactMatch(
      mode="body",
      pattern=r"^LINK_.*$",
      entity="robot",
      exclude=(
        "LINK_ANKLE_PITCH_L",
        "LINK_ANKLE_PITCH_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("force", "found", "pos"),
    reduce="maxforce",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg, body_contact_force_cfg,)

  ##
  # 动作配置
  ##
  
  # 根据 PM1 机器人的扭矩限制和刚度设置每个关节的动作缩放
  # PM_ACTION_SCALE 计算公式为：0.25 * effort_limit / stiffness（每个关节）
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM_ACTION_SCALE

  ##
  # 运动命令配置
  ##
  
  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # 设置锚点身体为基座链接，用于运动跟踪
  motion_cmd.anchor_body_name = "LINK_BASE"
  # 定义运动命令中要跟踪的身体链接
  # 注释掉的链接不用于跟踪（例如，某些 pitch/yaw 关节）
  motion_cmd.body_names = (
    "LINK_BASE",
    # "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    # "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    # "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    # "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    # "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    # "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    # "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    # "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    # "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    # "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_YAW",
  )
  motion_cmd.qd_mask = PM_QD_MASK
  # 与 rl_dance_runner 一致：sum(|tau_0:12|) 超限时等比缩小（见 ManagerBasedRlEnv）
  cfg.max_lower_body_torque = PM_MAX_LOWER_BODY_TORQUE
  cfg.lower_body_joint_names = PM_LOWER_BODY_JOINT_NAMES

  ##
  # 域随机化事件
  ##
  
  # 训练时随机化脚部摩擦系数
  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^collision_(left|right)_foot(_toe)?$"
  # 训练时随机化基座质心位置
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_BASE",)

  ##
  # 终止条件
  ##
  
  # 如果末端执行器（踝关节）位置偏离目标太远，终止 episode
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
  )
  # 头部冲击过大时终止（避免手撑地后头部轻微贴地被误杀）
  cfg.terminations["forbidden_body_contact_force"].params["body_names"] = ("LINK_HEAD_YAW", "LINK_TORSO_YAW", "LINK_ELBOW_END_L", "LINK_ELBOW_END_R")
  cfg.terminations["forbidden_body_contact_force"].params["force_threshold"] = 1000.0

  ##
  # 奖励：reduce_contact_force 使用护具 map + 力衰减公式（传入 protector_map_dir 等参数）
  ##
  cfg.rewards["reduce_contact_force"].params["protector_map_dir"] = PROTECTOR_MAP_DIR
  cfg.rewards["reduce_contact_force"].params["force_params_path"] = PROTECTOR_MAP_DIR / "fitted_parameters.json"
  cfg.rewards["reduce_contact_force"].params["asset_cfg"] = SceneEntityCfg("robot")
  cfg.rewards["reduce_contact_force"].params["density_protector"] = 0.4

  ##
  # 查看器配置
  ##
  
  # 设置查看器跟随基座链接
  cfg.viewer.body_name = "LINK_BASE"

  ##
  # PM1 机器人传感器名称修复
  ##
  
  # 修复 PM1 机器人的传感器名称（与 G1 使用不同的传感器名称）
  # PM1 使用: imu_link_linear_velocity, imu_angular_velocity
  # G1 使用: imu_lin_vel, imu_ang_vel
  if "base_lin_vel" in cfg.observations["policy"].terms:
    cfg.observations["policy"].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in cfg.observations["policy"].terms:
    cfg.observations["policy"].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"
  if "base_lin_vel" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = "robot/imu_link_linear_velocity"
  if "base_ang_vel" in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_angular_velocity"

  ##
  # 观测修改
  ##
  
  # 如果没有状态估计，修改观测
  # 移除需要状态估计的 base_lin_vel 和 motion_anchor_pos_b
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  ##
  # 播放模式配置
  ##
  
  # 应用播放模式覆盖设置（用于推理/评估）
  if play:
    # 无限 episode 长度（无自动终止）
    cfg.episode_length_s = int(1e9)

    # 禁用观测噪声/损坏，用于干净的推理
    cfg.observations["policy"].enable_corruption = False
    # 播放时禁用随机推动与 reset 初速度
    cfg.events.pop("push_robot", None)
    cfg.events.pop("reset_base_velocity", None)

    # 禁用 RSI（随机状态初始化）随机化
    # episode 开始时无随机姿态/速度变化
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    # 始终从运动文件的开头开始
    motion_cmd.sampling_mode = "start"

  return cfg
