#!/usr/bin/env bash
# 顺序执行 8 个方向 fine-tune：前一个跑完后自动跑下一个
# 用法: 直接编辑本文件中的 RUN_PATHS 后执行
#   bash run_train_sequential_finetune.sh

set -e  # 任一命令失败则退出（若希望某个失败后继续跑后面的，可注释掉本行）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_CMD="python -m mjlab.scripts.train"
CONFIG="Mjlab-Tracking-Flat-PM1"
# 护具 TSV 文件名（相对 config/pm1/protector_map/），与 train 的 --protector-map-front/back 一致
# PROTECTOR_MAP_FRONT="yz_map_front_elbow_knee.tsv"
# PROTECTOR_MAP_BACK="yz_map_back_elbow.tsv"
# COMMON_ARGS="--env.scene.num-envs 4096 --agent.max_iterations 10000 --protector-map-front ${PROTECTOR_MAP_FRONT} --protector-map-back ${PROTECTOR_MAP_BACK}"

# 不用护具 map 时：注释掉上面 PROTECTOR_MAP_* 与 COMMON_ARGS 一行，改用下面这行（reduce_contact_force 用原始接触力）：
COMMON_ARGS="--env.scene.num-envs 4096 --agent.max_iterations 10000 --use-protector-map False"
# 仍走 map 逻辑但厚度为 0 时，可改用：yz_map_front_zero.tsv / yz_map_back_zero.tsv 作为 PROTECTOR_MAP_*。

# 8 个方向对应的 wandb run path（必须与 MOTIONS 顺序一致）
RUN_PATHS=(
  "1205492990-nus/mjlab/vxhg287u"  # front
  "1205492990-nus/mjlab/myqvuj04"  # back
  "1205492990-nus/mjlab/b4nltr5v"  # left
  "1205492990-nus/mjlab/mj34ew0r"  # right
  "1205492990-nus/mjlab/ds60vcdq"  # left_front
  "1205492990-nus/mjlab/vpx28hnf"  # left_back
  "1205492990-nus/mjlab/cfc0q2jq"  # right_front
  "1205492990-nus/mjlab/ik29m1md"  # right_back
)

# 要顺序执行的 motion 文件列表（可自行增删改）
MOTIONS=(
  "motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toBack_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toLeft_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toRight_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toLeftFront_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toLeftBack_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toRightFront_1_converted_50fps.npz"
  "motion_file/pm_fall4:v0/toRightBack_1_converted_50fps.npz"
)

echo "======== 共 ${#MOTIONS[@]} 个任务，按顺序执行 ========"

for i in "${!MOTIONS[@]}"; do
  motion="${MOTIONS[$i]}"
  run_path="${RUN_PATHS[$i]}"
  echo ""
  echo "======== [$((i+1))/${#MOTIONS[@]}] 开始: $motion ========"
  echo "======== 使用 run_path: $run_path ========"
  $RUN_CMD "$CONFIG" --motion-file "$motion" --agent.resume True --wandb-run-path "$run_path" $COMMON_ARGS
  echo "======== [$((i+1))/${#MOTIONS[@]}] 完成: $motion ========"
done

echo ""
echo "======== 全部 ${#MOTIONS[@]} 个任务已执行完毕 ========"
