"""Clear all MERFISH STATE results (model outputs)."""

import shutil
from pathlib import Path

from scaling_laws.paths import DATA_DIR
path_to_data_dir = str(DATA_DIR)

sizes = [100, 203, 414, 843, 1716, 3494, 7113, 14480, 29475, 60000]
qualities = [0.027248, 0.0406617, 0.0606789, 0.0905502, 0.1351267, 0.2016475, 0.3009156, 0.4490518, 0.6701133, 1.0]

removed = 0

# 1. Remove results/State directories
for size in sizes:
    for quality in qualities:
        state_dir = Path(path_to_data_dir) / "merfish" / str(size) / str(quality) / "results" / "State"
        if state_dir.exists():
            shutil.rmtree(state_dir)
            print(f"Removed {state_dir}")
            removed += 1

print(f"\nDone. Removed {removed} items.")
