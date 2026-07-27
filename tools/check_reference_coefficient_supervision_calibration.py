from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    test_file = PROJECT_ROOT / "tests/test_reference_coefficient_supervision_calibration.py"
    specification = importlib.util.spec_from_file_location("calibration_tests", test_file)
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load calibration tests")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    tests = sorted(
        (name, value) for name, value in vars(module).items()
        if name.startswith("test_") and callable(value)
    )
    results = {}
    for name, function in tests:
        if name == "test_historical_outputs_immutable":
            with tempfile.TemporaryDirectory() as directory:
                function(Path(directory))
        else:
            function()
        results[name] = "PASS"
    print(json.dumps({
        "status": "PASS", "test_file": str(test_file),
        "test_count": len(results), "tests": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
