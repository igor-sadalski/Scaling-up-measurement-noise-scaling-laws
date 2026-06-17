"""Single source of truth for absolute paths used across the repo.

All paths can be overridden by env vars so the same code runs unchanged on
any machine. Defaults assume:
    - The `scaling_laws` package is installed editable (`pip install -e .`)
      from this repo, so `__file__` lives inside the source tree and
      `REPO_ROOT` resolves to the repo root.
    - The user's input data lives at `~/noise_scaling/data/`.
    - The STATE conda env is at `~/miniconda3/envs/state/`.

Override any of these by exporting the corresponding env var before launching
a script. See README.md → "Environment variables" for the full contract.
"""
from __future__ import annotations

import os
from pathlib import Path

# paths.py → scaling_laws (inner pkg) → src → scaling_laws (outer dir) → repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

# In-repo locations (no env override needed by default)
STATE_PACKAGE_DIR = Path(os.environ.get(
    "STATE_PACKAGE_DIR", str(REPO_ROOT / "STATE" / "state")
))
STATE_DEFAULTS_YAML = STATE_PACKAGE_DIR / "src" / "state" / "configs" / "state-defaults.yaml"
GENEFORMER_DIR = REPO_ROOT / "Geneformer"

# Out-of-repo locations (must be machine-specific)
STATE_PYTHON = Path(os.environ.get(
    "STATE_PYTHON",
    str(Path.home() / "miniconda3" / "envs" / "state" / "bin" / "python"),
))
DATA_DIR = Path(os.environ.get(
    "NOISE_SCALING_DATA_DIR", str(Path.home() / "noise_scaling" / "data")
))
OUTPUT_BASE = Path(os.environ.get(
    "NOISE_SCALING_OUTPUT_BASE", str(DATA_DIR / "other")
))


if __name__ == "__main__":
    # `python -m scaling_laws.paths` prints the resolved values + existence
    # check. Use this on a new machine to verify the env is wired correctly.
    rows = [
        ("REPO_ROOT", REPO_ROOT),
        ("STATE_PACKAGE_DIR", STATE_PACKAGE_DIR),
        ("STATE_DEFAULTS_YAML", STATE_DEFAULTS_YAML),
        ("GENEFORMER_DIR", GENEFORMER_DIR),
        ("STATE_PYTHON", STATE_PYTHON),
        ("DATA_DIR", DATA_DIR),
        ("OUTPUT_BASE", OUTPUT_BASE),
    ]
    width = max(len(name) for name, _ in rows)
    for name, p in rows:
        marker = "OK " if p.exists() else "MISSING"
        print(f"  [{marker}] {name:<{width}} = {p}")
