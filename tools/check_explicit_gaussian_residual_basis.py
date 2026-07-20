from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    path = PROJECT_ROOT / "tests/test_explicit_gaussian_residual_basis.py"
    spec = importlib.util.spec_from_file_location("explicit_basis_contract_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load explicit-basis contract tests")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tests = [(name, value) for name, value in vars(module).items() if name.startswith("test_") and callable(value)]
    failures = []
    for name, test in sorted(tests):
        if inspect.signature(test).parameters:
            failures.append((name, "test unexpectedly requires fixtures")); continue
        try:
            test()
        except Exception:
            failures.append((name, traceback.format_exc()))
    if failures:
        for name, detail in failures: print(f"FAIL {name}\n{detail}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS all {len(tests)} explicit Gaussian residual-basis contract tests")


if __name__ == "__main__":
    main()
