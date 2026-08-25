"""Download the latest W&B run history and summarize AMP training signals."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ENTITY = "e1519767-national-university-of-singapore"
DEFAULT_PROJECT = "mjlab"
DEFAULT_GROUP = "pm1_falling_amp"
DEFAULT_OUT_DIR = Path("outputs") / "wandb_runs"

AMP_METRICS = (
  "Loss/amp_loss",
  "Loss/grad_pen_loss",
  "Loss/policy_pred",
  "Loss/expert_pred",
  "Loss/accuracy_policy",
  "Loss/accuracy_expert",
  "Loss/mean_kl_divergence",
  "Debug/ratio_mean",
  "Debug/ratio_std",
  "Debug/actor_grad_norm",
  "Debug/policy_amp_input_norm",
  "Debug/expert_amp_input_norm",
  "Debug/policy_expert_pair_distance",
  "Policy/mean_noise_std",
  "Train/mean_reward",
  "Train/mean_task_reward",
  "Train/mean_style_reward",
  "Train/mean_effective_task_reward",
  "Train/mean_effective_style_reward",
  "Train/mean_effective_mixed_reward",
  "Train/style_reward_std",
  "Train/style_balance_scale",
  "Train/task_weight_scale",
)

ENV_METRIC_PREFIXES = (
  "Episode_Reward/",
  "Episode_Termination/",
  "Metrics/",
  "Train/",
)


@dataclass(frozen=True)
class MetricStats:
  name: str
  count: int
  first: float | None
  last: float | None
  mean_tail: float | None
  min_tail: float | None
  max_tail: float | None


def _json_default(value: Any) -> Any:
  if isinstance(value, (dt.date, dt.datetime)):
    return value.isoformat()
  if isinstance(value, Path):
    return str(value)
  return str(value)


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
  return {k: v for k, v in config.items() if not k.startswith("_")}


def _safe_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  if not math.isfinite(out):
    return None
  return out


def _slug(value: str) -> str:
  return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _run_path(entity: str, project: str, run_id: str) -> str:
  return f"{entity}/{project}/{run_id}"


def _select_latest_run(
  api: Any,
  entity: str,
  project: str,
  group: str | None,
  name_contains: str | None,
) -> Any:
  filters: dict[str, Any] = {}
  if group:
    filters["group"] = group
  runs = list(
    api.runs(
      f"{entity}/{project}",
      filters=filters or None,
      order="-created_at",
      per_page=50,
    )
  )
  if name_contains:
    runs = [run for run in runs if name_contains in (run.name or "")]
  if not runs:
    details = [f"project={entity}/{project}"]
    if group:
      details.append(f"group={group}")
    if name_contains:
      details.append(f"name_contains={name_contains}")
    raise RuntimeError(f"No W&B runs found for {', '.join(details)}.")
  return runs[0]


def _history_keys(run: Any, extra_keys: Sequence[str]) -> list[str]:
  keys = ["_step", *AMP_METRICS]
  summary_keys = set(getattr(run.summary, "_json_dict", {}) or {})
  for key in sorted(summary_keys):
    if key.startswith(ENV_METRIC_PREFIXES) and key not in keys:
      keys.append(key)
  for key in extra_keys:
    if key and key not in keys:
      keys.append(key)
  return [key for key in keys if key == "_step" or key in summary_keys or "/" in key]


def _download_history(run: Any, keys: Sequence[str], samples: int) -> list[dict[str, Any]]:
  if samples > 0:
    return [
      dict(row)
      for row in run.history(
        samples=samples,
        keys=[key for key in keys if key != "_step"],
        pandas=False,
      )
    ]

  rows_by_step: dict[int, dict[str, Any]] = {}
  metric_keys = [key for key in keys if key != "_step"]
  for key in metric_keys:
    metric_rows = 0
    for row in run.scan_history(keys=["_step", key], page_size=1000):
      step = row.get("_step")
      if step is None:
        continue
      step_int = int(step)
      rows_by_step.setdefault(step_int, {"_step": step_int})[key] = row.get(key)
      metric_rows += 1
      if samples > 0 and metric_rows >= samples:
        break

  rows = [rows_by_step[step] for step in sorted(rows_by_step)]
  return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> None:
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(keys), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _metric_stats(
  rows: Sequence[dict[str, Any]],
  metric: str,
  tail: int,
) -> MetricStats:
  values = [_safe_float(row.get(metric)) for row in rows]
  values = [v for v in values if v is not None]
  tail_values = values[-tail:] if tail > 0 else values
  return MetricStats(
    name=metric,
    count=len(values),
    first=values[0] if values else None,
    last=values[-1] if values else None,
    mean_tail=(sum(tail_values) / len(tail_values)) if tail_values else None,
    min_tail=min(tail_values) if tail_values else None,
    max_tail=max(tail_values) if tail_values else None,
  )


def _fmt(value: float | None) -> str:
  if value is None:
    return "n/a"
  return f"{value:.5g}"


def _diagnostic_notes(stats_by_name: dict[str, MetricStats]) -> list[str]:
  notes: list[str] = []
  policy_pred = stats_by_name.get("Loss/policy_pred")
  expert_pred = stats_by_name.get("Loss/expert_pred")
  acc_policy = stats_by_name.get("Loss/accuracy_policy")
  acc_expert = stats_by_name.get("Loss/accuracy_expert")
  grad_pen = stats_by_name.get("Loss/grad_pen_loss")
  style_scale = stats_by_name.get("Train/style_balance_scale")
  ratio_std = stats_by_name.get("Debug/ratio_std")
  noise = stats_by_name.get("Policy/mean_noise_std")

  if policy_pred and expert_pred and policy_pred.mean_tail is not None and expert_pred.mean_tail is not None:
    gap = expert_pred.mean_tail - policy_pred.mean_tail
    if gap < 0.08:
      notes.append(
        "Discriminator separation is weak in the tail. Consider stronger disc updates: "
        "`disc_epochs=2`, `disc_lr=2.5e-4`, or lower `disc_grad_penalty`."
      )
    elif gap > 0.55:
      notes.append(
        "Discriminator separation is very large. If style reward becomes spiky, consider "
        "`disc_input_noise_std=0.02..0.05` or lower discriminator capacity."
      )
    else:
      notes.append("Discriminator separation looks usable; tune reward mixing before changing disc capacity.")

  if acc_policy and acc_expert and acc_policy.mean_tail is not None and acc_expert.mean_tail is not None:
    mean_acc = 0.5 * (acc_policy.mean_tail + acc_expert.mean_tail)
    if mean_acc > 0.9:
      notes.append("Tail discriminator accuracy is high; watch for saturated style rewards.")
    elif mean_acc < 0.58:
      notes.append("Tail discriminator accuracy is near chance; the AMP signal may be too weak.")

  if grad_pen and grad_pen.mean_tail is not None and grad_pen.mean_tail > 1.0:
    notes.append("Gradient penalty is prominent in the tail; `disc_grad_penalty=10` may be too restrictive.")

  if style_scale and style_scale.mean_tail is not None:
    if style_scale.mean_tail > 3.8:
      notes.append("Style balance scale is near the upper clip; style reward is likely too small versus task reward.")
    elif style_scale.mean_tail < 0.3:
      notes.append("Style balance scale is near the lower clip; style reward may dominate task reward.")

  if ratio_std and ratio_std.mean_tail is not None and ratio_std.mean_tail > 0.25:
    notes.append("PPO ratio std is high; policy updates may be aggressive for the current reward mix.")

  if noise and noise.mean_tail is not None and noise.mean_tail < 0.03:
    notes.append("Policy noise is very low; if behavior collapses early, try a little entropy or higher init noise.")

  return notes


def _write_diagnostics(
  path: Path,
  run: Any,
  rows: Sequence[dict[str, Any]],
  stats: Iterable[MetricStats],
  tail: int,
) -> None:
  stats_list = list(stats)
  stats_by_name = {item.name: item for item in stats_list}
  lines = [
    "# W&B AMP Diagnostics",
    "",
    f"- run: `{run.path[-1]}`",
    f"- name: `{run.name}`",
    f"- state: `{run.state}`",
    f"- url: {run.url}",
    f"- history_rows: `{len(rows)}`",
    f"- tail_window: `{tail}`",
    "",
    "## Metric Tail Summary",
    "",
    "| metric | count | first | last | tail_mean | tail_min | tail_max |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for item in stats_list:
    if item.count == 0:
      continue
    lines.append(
      f"| `{item.name}` | {item.count} | {_fmt(item.first)} | {_fmt(item.last)} | "
      f"{_fmt(item.mean_tail)} | {_fmt(item.min_tail)} | {_fmt(item.max_tail)} |"
    )

  notes = _diagnostic_notes(stats_by_name)
  lines.extend(["", "## Tuning Notes", ""])
  if notes:
    lines.extend(f"- {note}" for note in notes)
  else:
    lines.append("- Not enough AMP metrics were present to make a tuning call.")
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Download latest W&B run data and summarize AMP training metrics.",
  )
  parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
  parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
  parser.add_argument("--group", default=DEFAULT_GROUP, help="W&B group/experiment name. Use empty string to disable.")
  parser.add_argument("--run-path", default=None, help="Explicit run path: entity/project/run_id.")
  parser.add_argument("--name-contains", default=None, help="Optional run-name substring filter.")
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--samples", type=int, default=0, help="Max history rows to download; 0 means all.")
  parser.add_argument("--tail", type=int, default=200, help="Tail window for diagnostics.")
  parser.add_argument("--extra-key", action="append", default=[], help="Additional W&B history key to download.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  group = args.group or None

  import wandb

  api = wandb.Api()
  if args.run_path:
    run = api.run(args.run_path)
  else:
    run = _select_latest_run(
      api=api,
      entity=args.entity,
      project=args.project,
      group=group,
      name_contains=args.name_contains,
    )

  entity, project, run_id = run.path
  out_dir = args.out_dir / _slug(project) / _slug(run.name or run_id)
  out_dir.mkdir(parents=True, exist_ok=True)

  keys = _history_keys(run, args.extra_key)
  rows = _download_history(run, keys, args.samples)

  _write_csv(out_dir / "history.csv", rows, keys)
  (out_dir / "run.json").write_text(
    json.dumps(
      {
        "path": _run_path(entity, project, run_id),
        "name": run.name,
        "state": run.state,
        "group": getattr(run, "group", None),
        "created_at": getattr(run, "created_at", None),
        "updated_at": getattr(run, "updated_at", None),
        "url": run.url,
      },
      indent=2,
      default=_json_default,
    ),
    encoding="utf-8",
  )
  (out_dir / "summary.json").write_text(
    json.dumps(dict(run.summary), indent=2, default=_json_default),
    encoding="utf-8",
  )
  (out_dir / "config.json").write_text(
    json.dumps(_clean_config(dict(run.config)), indent=2, default=_json_default),
    encoding="utf-8",
  )

  stats = [
    _metric_stats(rows, metric, args.tail)
    for metric in keys
    if metric != "_step"
  ]
  _write_diagnostics(out_dir / "diagnostics.md", run, rows, stats, args.tail)

  print(f"Downloaded W&B run: {_run_path(entity, project, run_id)}")
  print(f"Output directory: {out_dir}")
  print(f"History rows: {len(rows)}")
  print(f"Diagnostics: {out_dir / 'diagnostics.md'}")


if __name__ == "__main__":
  main()
