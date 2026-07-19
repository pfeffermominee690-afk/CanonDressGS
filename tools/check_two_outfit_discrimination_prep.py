from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Approx:
    def __init__(self, expected, rel: float = 1e-6, abs_: float = 1e-12) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs_

    def __eq__(self, actual) -> bool:
        tolerance = max(self.abs, self.rel * abs(float(self.expected)))
        return abs(float(actual) - float(self.expected)) <= tolerance


def _pytest_compatibility_module() -> types.ModuleType:
    module = types.ModuleType("pytest")

    def fixture(*args, **kwargs):
        del args, kwargs
        return lambda function: function

    module.fixture = fixture
    module.approx = lambda expected, **kwargs: _Approx(
        expected, rel=float(kwargs.get("rel", 1e-6)), abs_=float(kwargs.get("abs", 1e-12))
    )
    return module


def main() -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.modules["pytest"] = _pytest_compatibility_module()
    path = PROJECT_ROOT / "tests" / "test_two_outfit_discrimination_prep.py"
    spec = importlib.util.spec_from_file_location("two_outfit_prep_contract_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load two-outfit prep contract tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = [
        value for name, value in module.__dict__.items()
        if name.startswith("test_") and inspect.isfunction(value)
    ]
    for test in tests:
        with tempfile.TemporaryDirectory(prefix=f"{test.__name__}_") as directory:
            temp_path = Path(directory)
            arguments = {}
            for name in inspect.signature(test).parameters:
                if name == "tmp_path":
                    arguments[name] = temp_path
                elif name == "two_outfit_manifest":
                    arguments[name] = module.two_outfit_manifest(temp_path)
                else:
                    raise RuntimeError(f"unsupported local test fixture: {name}")
            test(**arguments)
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} two-outfit preparation contract tests")


if __name__ == "__main__":
    main()
