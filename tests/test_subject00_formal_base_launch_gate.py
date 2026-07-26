from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    REPO_ROOT
    / "tools"
    / "second_identity"
    / "validate_subject00_formal_base_launch_gate.py"
)


def test_subject00_formal_base_101245_launch_gate_bundle() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SUBJECT00_FORMAL_BASE_101245_LAUNCH_GATE_PASS" in result.stdout
