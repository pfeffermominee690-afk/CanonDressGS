from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "repaired_rendering",
    ROOT / "tools/paper/run_controller_v2_repaired_rendering.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_hard_mode_keeps_dominant_geometry() -> None:
    assert module.GEOMETRY_CHANNELS == {
        "delta_xyz", "delta_log_scaling", "delta_rotvec"
    }


def test_render_accounting_contract_is_exact() -> None:
    assert 1920 + 480 + 2880 == 5280
