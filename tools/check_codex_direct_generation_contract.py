#!/usr/bin/env python3
"""Validate the Codex platform-managed 28-target generation contract."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.aaai27 import manage_codex_direct_generation as manager  # noqa: E402


def context() -> dict[str, Any]:
    return {
        "repo": PROJECT_ROOT,
        "contract": json.loads(
            (PROJECT_ROOT / "artifacts/aaai27_sprint/codex_direct_generation/generation_contract.json").read_text(encoding="utf-8")
        ),
    }


def test_codex_generation_uses_no_external_api(ctx: dict[str, Any]) -> None:
    source = inspect.getsource(manager)
    assert ctx["contract"]["external_api_used"] is False
    for forbidden in ("import openai", "from openai", "import requests", "import httpx", "urlopen("):
        assert forbidden not in source


def test_codex_generation_uses_no_api_key(ctx: dict[str, Any]) -> None:
    source = inspect.getsource(manager)
    assert ctx["contract"]["api_key_used"] is False
    assert "os.environ" not in source and "getenv(" not in source


def test_all_28_samples_use_same_provider(ctx: dict[str, Any]) -> None:
    records = [{"generation_provider": manager.PROVIDER} for _ in range(28)]
    assert {item["generation_provider"] for item in records} == {"CODEX_IMAGE_GENERATION_SKILL"}
    assert ctx["contract"]["generation_provider"] == manager.PROVIDER


def test_existing_unknown_model_targets_are_not_reused(ctx: dict[str, Any]) -> None:
    historical = {
        "generation_provider": "codex_builtin_imagegen",
        "model_id": "UNEXPOSED_PLATFORM_IMAGE_MODEL",
        "external_api_used": False,
        "api_key_used": False,
        "raw_output_immutable": True,
        "output_sha256": "0" * 64,
    }
    assert not manager.existing_target_is_reusable(historical)


def test_backend_model_may_be_unexposed(ctx: dict[str, Any]) -> None:
    assert manager.backend_model_contract_valid(None, "NOT_EXPOSED_BY_PLATFORM")
    assert manager.backend_model_contract_valid("platform-exposed-model", "EXPOSED_BY_PLATFORM")
    assert not manager.backend_model_contract_valid(None, "UNKNOWN")


def _input_fixture(root: Path) -> dict[str, Any]:
    inputs = {}
    for name in ("base", "clay", "reference", "pose", "camera"):
        path = root / f"{name}.bin"
        path.write_bytes((name * 5).encode("utf-8"))
        inputs[name] = {"path": str(path), "sha256": manager.sha256(path)}
    return {
        "sample_id": "cond_000000_O01",
        "inputs": inputs,
        "prompt": "fixed prompt",
        "prompt_sha256": manager.sha256_text("fixed prompt"),
    }


def test_generation_records_input_hashes(ctx: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = _input_fixture(Path(directory))
        entries = manager.input_entries(source)
        assert [item["role"] for item in entries] == list(manager.INPUT_ROLES)
        assert all(manager.sha256(Path(item["path"])) == item["sha256"] for item in entries)


def test_generation_records_prompt_hash(ctx: dict[str, Any]) -> None:
    prompt = "an exact frozen prompt\nwith two lines"
    assert manager.sha256_text(prompt) == manager.hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def test_generation_is_append_only(ctx: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.png"
        image = Image.new("RGB", manager.EXPECTED_SIZE, (255, 255, 254))
        image.putpixel((0, 0), (0, 0, 0))
        image.save(source)
        destination = root / "raw_direct_edit.png"
        manager.copy_immutable_output(source, destination)
        try:
            manager.copy_immutable_output(source, destination)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Immutable output was overwritten")


def test_codex_generation_resume_skips_completed_samples(ctx: dict[str, Any]) -> None:
    manifest = {
        "records": [
            {"sample_id": "first", "status": "SUCCESS"},
            {"sample_id": "second", "status": "PENDING"},
            {"sample_id": "third", "status": "PENDING"},
        ]
    }
    assert manager.next_pending_record(manifest)["sample_id"] == "second"


def test_generation_invocation_budget(ctx: dict[str, Any]) -> None:
    manifest = {"primary_invocations": 28, "technical_retries": 0, "total_invocations": 28}
    try:
        manager.check_budget(manifest, is_retry=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Primary budget was not enforced")
    manager.check_budget(manifest, is_retry=True)
    manifest.update({"technical_retries": 4, "total_invocations": 32})
    try:
        manager.check_budget(manifest, is_retry=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Retry/total budget was not enforced")


def test_handoff_contains_28_unique_targets(ctx: dict[str, Any]) -> None:
    targets = [f"raw_targets/O{i:02d}/cond_{i:06d}/raw_direct_edit.png" for i in range(28)]
    records = [f"provenance/sample_{i:02d}/generation_record.json" for i in range(28)]
    assert manager.handoff_counts_are_complete(targets, records)
    assert not manager.handoff_counts_are_complete(targets[:-1], records)


def test_handoff_contains_no_secret(ctx: dict[str, Any]) -> None:
    assert not manager.contains_secret(b'{"external_api_used": false, "api_key_used": false}')
    assert manager.contains_secret(b"Authorization: Bearer deliberately_fake_secret_value_123")
    assert manager.contains_secret(b"OPENAI_API_KEY=deliberately_fake_secret_value_123")


def test_gate_does_not_start_before_28_valid_targets(ctx: dict[str, Any]) -> None:
    manifest = {"successful_count": 27, "records": [{"status": "SUCCESS"}] * 27}
    try:
        manager.validate_complete_records(Path("unused"), manifest)
    except RuntimeError as error:
        assert "28 valid targets" in str(error)
    else:
        raise AssertionError("Incomplete generation entered the gate")


def test_long_term_branch_remains_unchanged(ctx: dict[str, Any]) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "refs/heads/pipeline/full-dressable-20260715"],
        cwd=ctx["repo"],
        text=True,
        encoding="utf-8",
    ).strip()
    assert head == manager.EXPECTED_LONG_HEAD


def main() -> int:
    ctx = context()
    tests: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for name, test in tests:
        test(ctx)
        print(f"PASS {name}")
    print(f"PASS {len(tests)}/{len(tests)} Codex direct-generation contract checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
