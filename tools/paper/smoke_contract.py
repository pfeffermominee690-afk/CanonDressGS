from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


SMOKE_MARKER = "SMOKE ONLY — NOT PAPER RESULTS"
NOT_FOR_PAPER_NUMBERS = "NOT_FOR_PAPER_NUMBERS"
SMOKE_ROOT_NAME = "AAAI27-SEEN-OUTFIT-PAPER-SMOKE"
FORMAL_ROOT_NAME = "AAAI27-SEEN-OUTFIT-PAPER"
SMOKE_STATES = (
    "NOT_RUN",
    "PREFLIGHT_PASS",
    "RUNNING",
    "TRAINED",
    "EVALUATED",
    "MANUAL_REVIEW_REQUIRED",
    "SMOKE_ACCEPTED",
    "FAILED",
)
SMOKE_ATTEMPT_DIRECTORIES = (
    "contract",
    "preflight",
    "logs",
    "checkpoints",
    "raw_metrics",
    "evaluated_metrics",
    "visuals",
    "provenance",
    "final_adjudication",
)
EXPECTED_SMOKE_IDS = (
    "S-B0",
    "S-B1",
    "S-B2",
    "S-B3",
    "S-B4",
    "S-B5",
    "S-OURS",
    "S-A1",
    "S-A5",
    "S-A7",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(repo_root: Path, relative_path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{relative_path}"], cwd=repo_root, text=True
    ).strip()


def load_smoke_registry(path: Path) -> dict[str, Any]:
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_smoke_registry(registry)
    return registry


def validate_smoke_registry(registry: Mapping[str, Any]) -> None:
    experiments = list(registry.get("experiments", []))
    identifiers = [item.get("smoke_experiment_id") for item in experiments]
    if identifiers != list(EXPECTED_SMOKE_IDS):
        raise ValueError("smoke registry order or membership differs from the frozen matrix")
    if registry.get("output_root_name") != SMOKE_ROOT_NAME:
        raise ValueError("smoke output root is not isolated")
    if registry.get("forbidden_state") != "PAPER_FINAL":
        raise ValueError("smoke registry must explicitly forbid PAPER_FINAL")
    if tuple(registry.get("states", ())) != SMOKE_STATES:
        raise ValueError("smoke state machine differs from the registered contract")
    for experiment in experiments:
        if experiment.get("smoke_only") is not True:
            raise ValueError("every smoke entry must be smoke_only")
        if experiment.get("paper_final_eligible") is not False:
            raise ValueError("smoke entries cannot be paper-final eligible")
        if experiment.get("status") != "NOT_RUN":
            raise ValueError("checked-in smoke registry must remain NOT_RUN")
        if experiment.get("seed") != 0:
            raise ValueError("smoke registry must use seed 0")
        expected_steps = 20 if experiment["smoke_experiment_id"] == "S-OURS" else (
            0 if experiment["smoke_experiment_id"] in {"S-B0", "S-B1", "S-B2"} else 5
        )
        if experiment.get("optimizer_steps") != expected_steps:
            raise ValueError("smoke optimizer budget differs from the registered matrix")
        if FORMAL_ROOT_NAME + "/" in str(experiment.get("expected_output_root", "")):
            raise ValueError("smoke entry points at the formal paper output root")


def smoke_output_root(output_root: Path) -> Path:
    root = output_root.resolve() / SMOKE_ROOT_NAME
    if root.name != SMOKE_ROOT_NAME or root.name == FORMAL_ROOT_NAME:
        raise ValueError("refusing a non-isolated smoke output root")
    return root


def next_attempt_path(experiment_root: Path) -> Path:
    for number in range(1, 10000):
        candidate = experiment_root / f"attempt_{number:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("smoke attempt namespace exhausted")


def create_smoke_attempt(experiment_root: Path) -> Path:
    attempt = next_attempt_path(experiment_root)
    attempt.mkdir(parents=True, exist_ok=False)
    for name in SMOKE_ATTEMPT_DIRECTORIES:
        (attempt / name).mkdir()
    return attempt


def write_status(attempt: Path, status: str, **extra: Any) -> None:
    if status not in SMOKE_STATES:
        raise ValueError(f"invalid smoke state: {status}")
    payload = {
        "status": status,
        "smoke_only": True,
        "paper_final_eligible": False,
        "artifact_marker": SMOKE_MARKER,
        "paper_numbers_status": NOT_FOR_PAPER_NUMBERS,
        **extra,
    }
    temporary = attempt / "RUN_STATUS.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(attempt / "RUN_STATUS.json")


def assert_no_formal_destination(path: Path) -> None:
    parts = {part.lower() for part in path.resolve().parts}
    if FORMAL_ROOT_NAME.lower() in parts or "paper_ready" in parts:
        raise ValueError("smoke artifacts may not enter formal paper destinations")
