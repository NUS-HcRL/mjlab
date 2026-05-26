#!/usr/bin/env python3
"""从 AMP/RSL-RL 的 .pt checkpoint 导出 ONNX（仅 actor + obs normalizer）。

用法:
  conda activate mjlab
  python export_amp_pt_to_onnx.py motion_file/pm_fall4:v0/pt/amp_x2.pt

  # 指定输出目录/文件名
  python export_amp_pt_to_onnx.py path/to/model.pt -o motion_file/pm_fall4:v0/onnx -f amp_x2.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from mjlab.utils.lab_api.rl.exporter import export_policy_as_onnx


def _infer_dims(state_dict: dict) -> tuple[int, int, int]:
  actor_in = int(state_dict["actor.0.weight"].shape[1])
  critic_in = int(state_dict["critic.0.weight"].shape[1])
  num_actions = int(state_dict["actor.6.weight"].shape[0])
  return actor_in, critic_in, num_actions


def export_pt_to_onnx(
  pt_path: Path,
  out_dir: Path,
  onnx_name: str,
  *,
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
) -> Path:
  ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
  if "model_state_dict" not in ckpt:
    raise KeyError(f"{pt_path} 缺少 model_state_dict，是否为 AMP/RSL-RL checkpoint？")
  sd = ckpt["model_state_dict"]
  actor_in, critic_in, num_actions = _infer_dims(sd)

  obs = TensorDict(
    {"policy": torch.zeros(1, actor_in), "critic": torch.zeros(1, critic_in)},
    batch_size=[1],
  )
  obs_groups = {"policy": ["policy"], "critic": ["critic"]}
  actor_critic = ActorCritic(
    obs,
    obs_groups,
    num_actions=num_actions,
    actor_obs_normalization=True,
    critic_obs_normalization=True,
    actor_hidden_dims=actor_hidden_dims,
    critic_hidden_dims=critic_hidden_dims,
    activation="elu",
    init_noise_std=0.05,
    noise_std_type="log",
  )
  actor_critic.load_state_dict(sd, strict=True)
  actor_critic.eval()

  out_dir.mkdir(parents=True, exist_ok=True)
  export_policy_as_onnx(
    actor_critic,
    str(out_dir),
    normalizer=actor_critic.actor_obs_normalizer,
    filename=onnx_name,
  )
  return out_dir / onnx_name


def main() -> None:
  parser = argparse.ArgumentParser(description="Export AMP .pt checkpoint to ONNX")
  parser.add_argument("pt_file", type=Path, help="Path to model_*.pt or amp_x2.pt")
  parser.add_argument(
    "-o",
    "--out-dir",
    type=Path,
    default=None,
    help="Output directory (default: <pt_dir>/../onnx)",
  )
  parser.add_argument(
    "-f",
    "--filename",
    type=str,
    default=None,
    help="ONNX filename (default: <pt_stem>.onnx)",
  )
  args = parser.parse_args()

  pt_path = args.pt_file.resolve()
  if not pt_path.exists():
    raise SystemExit(f"文件不存在: {pt_path}")

  out_dir = args.out_dir or (pt_path.parent.parent / "onnx")
  onnx_name = args.filename or f"{pt_path.stem}.onnx"

  onnx_path = export_pt_to_onnx(pt_path, out_dir.resolve(), onnx_name)
  print(f"✓ ONNX: {onnx_path}")
  print("  输入 obs: [1, actor_dim], 输出 actions: [1, num_actions]")
  print("  MNN: python convert_onnx_to_mnn_batch.py --input_file", onnx_path)


if __name__ == "__main__":
  main()
