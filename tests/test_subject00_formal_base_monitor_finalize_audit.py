from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "tools"
    / "second_identity"
    / "audit_subject00_formal_base_101245_monitor_finalize.py"
)


def test_subject00_formal_base_monitor_finalize_static_check() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "static-check"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SUBJECT00_FORMAL_BASE_101245_MONITOR_FINALIZE_STATIC_PASS" in result.stdout
