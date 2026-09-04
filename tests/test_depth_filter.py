from pathlib import Path
import importlib.util

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vision" / "filter_masks_by_depth.py"
SPEC = importlib.util.spec_from_file_location("filter_masks_by_depth", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_keeps_near_component_and_removes_far_component():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[4:6, 4:6] = 255
    depth = np.full((6, 6), 5.0, dtype=float)
    depth[1:3, 1:3] = 1.0
    depth[4:6, 4:6] = 9.0

    filtered, summary, rows = MODULE.filter_single_mask(
        mask=mask,
        depth=depth,
        keep_percentile=50,
        connectivity=8,
        min_component_area=1,
        positive_only=True,
    )

    assert np.all(filtered[1:3, 1:3] == 255)
    assert np.all(filtered[4:6, 4:6] == 0)
    assert summary["num_components_kept"] == 1
    assert summary["num_components_removed"] == 1
    assert {row["status"] for row in rows} == {"kept", "removed_far"}
