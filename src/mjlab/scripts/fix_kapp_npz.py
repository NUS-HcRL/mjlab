"""Convert KAPP motion files to mjlab's body-array layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BODY_ARRAY_KEYS = (
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def remove_world_body(
  input_file: str | Path,
  output_file: str | Path,
) -> Path:
  """Remove KAPP's leading ``world`` entry while preserving all other NPZ data."""
  input_path = Path(input_file)
  output_path = Path(output_file)

  with np.load(input_path, allow_pickle=False) as source:
    if "body_names" not in source:
      raise ValueError(f"{input_path}: 缺少 body_names，无法安全确认 world 索引")

    body_names = source["body_names"]
    if body_names.ndim != 1 or len(body_names) == 0 or str(body_names[0]) != "world":
      raise ValueError(f"{input_path}: body_names[0] 不是 world，无需或无法转换")

    body_count = len(body_names)
    missing_keys = [key for key in BODY_ARRAY_KEYS if key not in source]
    if missing_keys:
      raise ValueError(f"{input_path}: 缺少必要字段 {missing_keys}")

    converted = {key: source[key] for key in source.files}
    for key in BODY_ARRAY_KEYS:
      array = source[key]
      if array.ndim < 2 or array.shape[1] != body_count:
        raise ValueError(
          f"{input_path}: {key} 的 body 维度 {array.shape} 与 body_names "
          f"数量 {body_count} 不一致"
        )
      converted[key] = array[:, 1:].copy()

    converted["body_names"] = body_names[1:].copy()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **converted)
  return output_path


def main() -> None:
  parser = argparse.ArgumentParser(
    description="删除 KAPP NPZ body 数组中位于索引 0 的 world 节点",
  )
  parser.add_argument("inputs", nargs="+", type=Path, help="待转换的 KAPP NPZ 文件")
  parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="转换后文件的输出目录（文件名保持不变）",
  )
  args = parser.parse_args()

  for input_path in args.inputs:
    output_path = args.output_dir / input_path.name
    remove_world_body(input_path, output_path)
    print(f"{input_path} -> {output_path}")


if __name__ == "__main__":
  main()
