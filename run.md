# 如果切换分支，需要重新 pip install -e .

# 安装
```bash
# 安装 conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda update -n base -c defaults conda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes

# 创建环境
conda create -n mjlab python=3.13

# 安装环境
# 使用 PyPI 版本（与5070笔记本一致）
# 注意：mujoco-warp 0.0.1 版本已修复 wp.math.sqrt 问题，使用 wp.sqrt，无需修补
pip install warp-lang==1.10.1
pip install mujoco==3.3.7
pip install mujoco-warp==0.0.1
pip install rsl-rl-lib==3.2.0
pip install MNN
pip install -e .

# 登录 wandb
export WANDB_ENTITY=1205492990-nus
export WANDB_PROJECT=mjlab
export WANDB_API_KEY=eb307b6cd96b693d24910f18a15b65ce95a61d90
# 或者 
wandb login eb307b6cd96b693d24910f18a15b65ce95a61d90
```

# 训练
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz --env.scene.num-envs 4096 --agent.max_iterations 10000

## 无护具（不做 map 衰减）
```bash
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --use-protector-map False

python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --wandb-run-path 1205492990-nus/mjlab/kdfee61g  

```

## 通过xml改link软硬，目前是给膝、肘处link统一更改
```bash
# 修改 src/mjlab/asset_zoo/robots/engineai_pm01/pm01_8.py    SOLREF_CONTACT_SOFT_6mm 改大
```

## 护具 map（前/后两个 TSV 名）
默认使用 map，无需写 `--use-protector-map True`；仅无护具时需传 `--use-protector-map False`。  
文件放在 `src/mjlab/tasks/tracking/config/pm1/protector_map/` 下，参数只写**文件名**（或绝对路径）。不传时默认 `yz_map_front.tsv` / `yz_map_back.tsv`。
```bash
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/Front_1_converted_50fps.npz \
  --protector-map-front yz_map_front_zero.tsv \
  --protector-map-back yz_map_back_zero.tsv \
  --env.scene.num-envs 4096 --agent.max_iterations 10000

python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 14000 \
  --agent.resume True \
  --protector-map-front yz_map_front_20260331_222436.tsv \
  --protector-map-back yz_map_back_20260331_222436.tsv \
  --wandb-run-path 1205492990-nus/mjlab/kdfee61g  
```
只改一侧时可只传一个参数，另一侧仍用默认名。

uv run train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/Front_1_converted_50fps.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000

## 顺序训练（多任务依次跑）

使用 `run_train_sequential.sh`：前一个任务跑完后自动跑下一个。可在脚本里改 `MOTIONS`（8 个 motion）、以及 `PROTECTOR_MAP_FRONT` / `PROTECTOR_MAP_BACK`（护具 TSV 名，默认 `yz_map_front.tsv` / `yz_map_back.tsv`）。

```bash
./run_train_sequential.sh
# 或
bash run_train_sequential.sh
```

脚本默认任一任务失败即退出。若希望某个失败后仍继续跑后面的，注释掉脚本中的 `set -e` 即可。

## 恢复训练 - 从 WandB 恢复（推荐）
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/motion.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --wandb-run-path e1519767-national-university-of-singapore/mjlab/run-id

## 恢复训练 - 从本地文件系统恢复
python -m mjlab.scripts.train Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/motion.npz \
  --env.scene.num-envs 4096 \
  --agent.max_iterations 10000 \
  --agent.resume True \
  --agent.load-run "2025-12-14_17-37-01" \
  --agent.load-checkpoint "model_1000.pt"

```bash
./run_train_sequential_finetune.sh
```

# 演示
## play 时护具 map（与 train 参数一致）
默认使用 map；指定前后 TSV 名（相对 `protector_map/`）；无护具加 `--use-protector-map False`。
```bash
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz \
  --wandb-run-path 1205492990-nus/mjlab/t46cmn6z \
  --protector-map-front yz_map_front_zero.tsv \
  --protector-map-back yz_map_back_zero.tsv

python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/toFront_1_converted_50fps.npz \
  --wandb-run-path 1205492990-nus/mjlab/u0uzc0vi \
  --use-protector-map False
```

## 用wandb文件
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/motion.npz --wandb-run-path 1205492990-nus/mjlab/vboc51sb
## 用本地pt
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 \
  --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz \
  --checkpoint-file motion_file/pm_fall4:v0/pt/toBack_2.pt

python -m mjlab.scripts.force Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Forward_1_converted.npz --wandb-run-path 1205492990-nus/mjlab/vboc51sb

纯mimic向前摔：1205492990-nus/mjlab/wkl6a3g1   加ForceReward: 1205492990-nus/mjlab/spl7m43l
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Front_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/wkl6a3g1

纯mimic向后摔：1205492990-nus/mjlab/e6vf2gjn   加ForceReward: 1205492990-nus/mjlab/kbw6thv6
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Back_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/e6vf2gjn

纯mimic向左摔：1205492990-nus/mjlab/yffk9hkx   加ForceReward: 1205492990-nus/mjlab/4sm7sf1w
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Left_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/5yjmv32g

纯mimic向右摔：1205492990-nus/mjlab/tx9z9bhl   加ForceReward: 1205492990-nus/mjlab/hajxuepw
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/Right_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/tx9z9bhl

纯mimic（从左前）向右后摔：1205492990-nus/mjlab/jaapk028   加ForceReward: 1205492990-nus/mjlab/t0hoxhos
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/jaapk028

纯mimic（从左后）向右前摔：1205492990-nus/mjlab/hnqysmih   加ForceReward: 1205492990-nus/mjlab/e89hpcao
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/LeftBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/hnqysmih

纯mimic（从右前）向左后摔：1205492990-nus/mjlab/hcu3gh3w   加ForceReward: 1205492990-nus/mjlab/e89hpcao
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightFront_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/hcu3gh3w

纯mimic（从右后）向左前摔：1205492990-nus/mjlab/hsbeg4f9   加ForceReward: 1205492990-nus/mjlab/4tzprhip 
python -m mjlab.scripts.play Mjlab-Tracking-Flat-PM1 --motion-file motion_file/pm_fall4:v0/RightBack_1_converted_50fps.npz --wandb-run-path 1205492990-nus/mjlab/hsbeg4f9



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
  --input_file motion_file/pm_fall4:v0/onnx/toFront_4_force.onnx \
  --output_file motion_file/pm_fall4:v0/onnx/toFront_4_force.mnn
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
