from pathlib import Path
import importlib.util

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vision" / "extract_instance_color_features.py"
SPEC = importlib.util.spec_from_file_location("extract_instance_color_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_channel_statistics_are_deterministic():
    stats = MODULE.channel_stats(np.array([0, 10, 20, 30], dtype=np.uint8), "x")
    assert stats["x_mean"] == 15.0
    assert stats["x_min"] == 0.0
    assert stats["x_max"] == 30.0
    assert stats["x_median"] == 15.0
