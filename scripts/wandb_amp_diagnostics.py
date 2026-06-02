"""Standalone launcher for the W&B AMP diagnostics helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  module_path = repo_root / "src" / "mjlab" / "scripts" / "wandb_amp_diagnostics.py"
  spec = importlib.util.spec_from_file_location("_mjlab_wandb_amp_diagnostics", module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load diagnostics module from {module_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  module.main()


if __name__ == "__main__":
  main()
