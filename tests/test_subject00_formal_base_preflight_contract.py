from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    REPO_ROOT
    / "tools"
    / "second_identity"
    / "validate_subject00_formal_base_preflight.py"
)


def test_subject00_formal_base_preflight_contract_bundle() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SUBJECT00_FORMAL_BASE_PREFLIGHT_CONTRACT_BUNDLE_PASS" in result.stdout
