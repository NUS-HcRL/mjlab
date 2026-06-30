# 训练
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/dance1_subject2_yaw0.npz --env.scene.num-envs 4096 --agent.max_iterations 10000

## 多 motion 训练（目录下全部 npz，一个 policy mimic 所有动作）
# --motion-file 可指向单个 .npz 或包含多个 .npz 的目录；目录下会加载全部 motion，每个 env reset 时随机采样一条。
# 单条 motion 常用 10000 iter；78 条 dodge 建议 25000~30000（动作多、每条见到的样本更少，iter 需加大；不够可 resume 继续训）。

# 单卡（默认 GPU 0）
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 25000

# 4 卡（GPU 0~3；每张卡各跑 num-envs 路并行，总并行 env = 4 × num-envs）
# 示例：4 × 512 = 2048 总 env，与单卡 4096 接近；OOM 可再降到 256~512
MUJOCO_GL=egl python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian \
  --gpu-ids 0 1 2 3 \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 25000

# 8 卡 3090（每张卡各跑 num-envs 路并行，总并行 env = 8 × num-envs）
# --gpu-ids all 等价于 0 1 2 3 4 5 6 7；若单卡 4096 会 OOM，可改为 512~1024（8 卡合计约 4096~8192 env）
MUJOCO_GL=egl python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 25000

## 多 motion 恢复训练
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian \
  --gpu-ids all \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 30000 \
  --agent.resume True \
  --wandb-run-path <entity>/mjlab/<run-id>

## 多 motion 演示（play 时指定单条 npz 测某条 motion；或仍传目录看多 env 随机不同 motion）
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian/street_avoid_car_000_stand_R_001__A428.npz \
  --wandb-run-path <entity>/mjlab/<run-id>

## 测某一条 dodge
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --wandb-run-path 1205492990-nus/mjlab/x8r42v0r \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian/street_avoid_car_000_stand_R_001__A428.npz

## 看多条 dodge 
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --wandb-run-path 1205492990-nus/mjlab/x8r42v0r \
  --motion-file motion_file/pm_fall4:v0/pm01_dodge_npz_aoqian \
  --num-envs 16 \
  --viewer viser

## 恢复训练 - 从 WandB 恢复（推荐）
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/dance1_subject2.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --wandb-run-path 1205492990-nus/mjlab/gqb1hfyv

## 恢复训练 - 从本地文件系统恢复
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/motion.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --agent.load-run "2025-12-14_17-37-01" \
  --agent.load-checkpoint "model_1000.pt"

# 演示
## 用wandb文件
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/dance1_subject2_yaw0_50fps.npz --wandb-run-path 1205492990-nus/mjlab/7uy2uvny
## 用本地pt
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz \
  --checkpoint-file motion_file/pm_fall4:v0/pt/toBack_2.pt

python -m mjlab.scripts.force Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Forward_1_converted.npz --wandb-run-path 1205492990-nus/mjlab/vboc51sb

python -m mjlab.scripts.force Mjlab-Falling-Flat-PM1-AMP --wandb-run-path e1519767-national-university-of-singapore/mjlab/t3s98zq5

纯mimic向前摔：1205492990-nus/mjlab/oaxwus98
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Front_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/oaxwus98

纯mimic向后摔：1205492990-nus/mjlab/vgc12h8p
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/vgc12h8p

纯mimic向左摔：1205492990-nus/mjlab/s2gtqz6w
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Left_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/s2gtqz6w

纯mimic向右摔：1205492990-nus/mjlab/ca1kdhmc
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Right_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/ca1kdhmc

纯mimic（从左前）向右后摔：
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/

纯mimic（从左后）向右前摔：
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/

纯mimic（从右前）向左后摔：
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/

纯mimic（从右后）向左前摔：1205492990-nus/mjlab/hjjmk8ib
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/hjjmk8ib

--wandb-run-path 1205492990-nus/mjlab/6icim82d

## MNN 模型转换

### 安装 MNN

MNN 转换工具通常已安装在系统中（`/usr/local/bin/MNNConvert`）。如果未安装，可以从源码编译：

```bash
# 克隆 MNN 仓库
git clone https://github.com/alibaba/MNN.git
cd MNN

# 编译（需要 CMake）
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j4

# 编译完成后，MNNConvert 位于 build/MNNConvert
# 可以复制到系统路径或添加到 PATH
sudo cp MNNConvert /usr/local/bin/
```

或者如果系统已安装，直接使用：
```bash
which MNNConvert  # 检查是否已安装
```

### 使用方式

#### 方式 1: 批量转换脚本（推荐）

使用批量转换脚本转换目录下所有 ONNX 文件：

```bash
# 转换指定目录下的所有 ONNX 文件
python convert_onnx_to_mnn_batch.py --input_dir motion_file/pm_fall4:v0/onnx

# 输出会生成同目录、同名的 .mnn，例如 model.onnx -> model.mnn
python convert_onnx_to_mnn_batch.py --input_file motion_file/pm_fall4:v0/onnx/toFront_chr_1.onnx

# 转换训练日志中的 ONNX 文件
python convert_onnx_to_mnn_batch.py --input_dir logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01
```

#### 方式 2: 使用 onnx_to_mnn.py 脚本

转换单个 ONNX 文件：

```bash
# 从 ONNX 文件转换
python -m mjlab.scripts.onnx_to_mnn \
  --input_file motion_file/pm_fall4:v0/onnx/dance2.onnx \
  --output_file motion_file/pm_fall4:v0/onnx/dance2.mnn
```

#### 方式 3: 直接使用 MNNConvert 命令行工具

```bash
cd ~/engineai/MNN/build && \
./MNNConvert \
  -f ONNX \
  --modelFile /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx \
  --MNNModel /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/model.mnn \
  --bizCode MNN
```

或者如果已安装到系统路径：

```bash
MNNConvert \
  -f ONNX \
  --modelFile model.onnx \
  --MNNModel model.mnn \
  --bizCode MNN
```

### 检查 ONNX 模型信息

```bash
python inspect_onnx.py /home/wang22/engineai/mjlab/logs/rsl_rl/pm1_tracking/2025-12-14_17-37-01/2025-12-14_17-37-01.onnx
```

#### 方式 4: 批量转换 PT 文件（自动查找对应 ONNX）

如果 `.pt` 文件在子目录中，脚本会自动在父目录查找对应的 `.onnx` 文件：

```bash
# 转换 pt 目录下的所有 .pt 文件（会自动查找对应的 ONNX）
python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0/pt

# 直接转换 ONNX 文件
python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0 --file-type onnx

# 指定输出目录
python -m mjlab.scripts.pt_to_mnn_batch --input-dir motion_file/pm_fall4:v0/pt --output-dir output_mnn
```

## NPZ 文件工具

### 查看 NPZ 文件 FPS

```bash
# 查看单个文件的 FPS
python -m mjlab.scripts.check_npz_fps motion_file/pm_fall4:v0/motion.npz

# 查看目录下所有 npz 文件的 FPS
python -m mjlab.scripts.check_npz_fps --input-dir motion_file/pm_fall4:v0
```

### 将 NPZ 文件转换为 CSV

#### 使用自定义列顺序（推荐）

按照指定格式生成单个 CSV 文件：

```bash
# 转换单个文件
python -m mjlab.scripts.npz_to_csv motion_file/pm_fall4:v0/motion.npz --custom-order

# 批量转换目录下所有 npz 文件
python -m mjlab.scripts.npz_to_csv --input-dir motion_file/pm_fall4:v0 --custom-order

# 指定输出目录
python -m mjlab.scripts.npz_to_csv --input-dir motion_file/pm_fall4:v0 --custom-order --output-dir output_csv
```

#### 分别保存每个数组为独立 CSV

```bash
# 转换单个文件（每个数组保存为单独的 CSV）
python -m mjlab.scripts.npz_to_csv motion_file/pm_fall4:v0/motion.npz

# 批量转换
python -m mjlab.scripts.npz_to_csv --input-dir motion_file/pm_fall4:v0
```
