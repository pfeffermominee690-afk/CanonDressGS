from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    REPO_ROOT
    / "tools"
    / "second_identity"
    / "validate_avatarrex_cloud_duplicate_deletion_authorization.py"
)


def test_avatarrex_cloud_duplicate_deletion_authorization_bundle() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "AVATARREX_CLOUD_DUPLICATE_DELETION_AUTHORIZATION_PASS" in result.stdout
