#!/usr/bin/env bash
# 顺序执行 kapp_armlink 八方向训练任务：前一个跑完后自动跑下一个
# 用法: ./run_train_sequential_kapp_armlink_8dir.sh

set -e  # 任一命令失败则退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_CMD="python -m mjlab.scripts.train"
CONFIG="Mjlab-Tracking-Flat-PM1"

# 跌倒动作接触较多；为每个环境预留接触和约束容量，避免 MuJoCo 丢弃数据
# 不使用护具 map；当前项目安装后由 TrainConfig 提供此参数
COMMON_ARGS="--env.scene.num-envs 4096 --env.sim.nconmax 96 --env.sim.njmax 640 --agent.max_iterations 10000 --use-protector-map False"

# kapp_armlink 八方向 motion 文件
MOTIONS=(
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_forward_row0753.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_backward_row0606.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_left_row0385.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_right_row0373.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_forward_left_row0882.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_forward_right_row1471.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_backward_left_row0537.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_backward_right_row0923.npz"
)

echo "======== 共 ${#MOTIONS[@]} 个任务，按顺序执行 ========"

for i in "${!MOTIONS[@]}"; do
  motion="${MOTIONS[$i]}"
  echo ""
  echo "======== [$((i+1))/${#MOTIONS[@]}] 开始: $motion ========"
  $RUN_CMD "$CONFIG" --motion-file "$motion" $COMMON_ARGS
  echo "======== [$((i+1))/${#MOTIONS[@]}] 完成: $motion ========"
done

echo ""
echo "======== 全部 ${#MOTIONS[@]} 个任务已执行完毕 ========"
