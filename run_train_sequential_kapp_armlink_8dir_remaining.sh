#!/usr/bin/env bash
# 接续训练剩余 3 个 kapp_armlink 方向（forward_right / backward_left / backward_right）
# 用法: ./run_train_sequential_kapp_armlink_8dir_remaining.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_CMD="python -m mjlab.scripts.train"
CONFIG="Mjlab-Tracking-Flat-PM1"

# 轨迹约 1.2s：timeout 取 2.0s（略长于单段动作），避免一集内反复 resample 跌倒末态
# 无护具；跌倒接触容量加大
COMMON_ARGS="--env.scene.num-envs 4096 --env.episode-length-s 2.0 --env.sim.nconmax 96 --env.sim.njmax 640 --agent.max_iterations 10000 --use-protector-map False"

MOTIONS=(
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_forward_right_row1471.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_backward_left_row0537.npz"
  "motion_file/pm_fall4:v0/kapp_armlink_8dir_mjlab/kapp_armlink_backward_right_row0923.npz"
)

echo "======== 剩余 ${#MOTIONS[@]} 个任务，按顺序执行 ========"

for i in "${!MOTIONS[@]}"; do
  motion="${MOTIONS[$i]}"
  echo ""
  echo "======== [$((i+1))/${#MOTIONS[@]}] 开始: $motion ========"
  $RUN_CMD "$CONFIG" --motion-file "$motion" $COMMON_ARGS
  echo "======== [$((i+1))/${#MOTIONS[@]}] 完成: $motion ========"
done

echo ""
echo "======== 剩余 ${#MOTIONS[@]} 个任务已执行完毕 ========"
