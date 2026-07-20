from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


STATUSES = (
    "NOT_RUN", "PREFLIGHT_PASS", "RUNNING", "TRAINED", "EVALUATED",
    "MANUAL_REVIEW_REQUIRED", "PAPER_FINAL", "FAILED", "HISTORICAL_EVIDENCE",
)
TRANSITIONS = {
    "NOT_RUN": {"PREFLIGHT_PASS", "FAILED"},
    "PREFLIGHT_PASS": {"RUNNING", "FAILED"},
    "RUNNING": {"TRAINED", "FAILED"},
    "TRAINED": {"EVALUATED", "FAILED"},
    "EVALUATED": {"MANUAL_REVIEW_REQUIRED", "FAILED"},
    "MANUAL_REVIEW_REQUIRED": {"PAPER_FINAL", "FAILED"},
    "PAPER_FINAL": set(),
    "FAILED": set(),
    "HISTORICAL_EVIDENCE": set(),
}


def transition_is_valid(old: str, new: str) -> bool:
    return old in TRANSITIONS and new in TRANSITIONS[old]


def validate_paper_final_evidence(evidence: Mapping[str, Any] | None) -> None:
    required = {
        "evaluator_pass": True,
        "artifacts_complete": True,
        "manual_adjudication_present": True,
        "explicit_confirmation": True,
    }
    if evidence is None or any(evidence.get(key) is not value for key, value in required.items()):
        raise ValueError("PAPER_FINAL requires evaluator PASS, complete artifacts, manual adjudication, and explicit confirmation")


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(dict(payload), stream, sort_keys=False, allow_unicode=True)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def transition_registry(
    registry_path: Path,
    experiment_id: str,
    new_status: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    matches = [item for item in registry["experiments"] if item["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise KeyError(f"experiment id must resolve exactly once: {experiment_id}")
    experiment = matches[0]
    old_status = experiment["status"]
    if not experiment.get("executable", False) or old_status == "HISTORICAL_EVIDENCE":
        raise ValueError("historical evidence is immutable and non-executable")
    if not transition_is_valid(old_status, new_status):
        raise ValueError(f"invalid registry transition: {old_status} -> {new_status}")
    if new_status == "PAPER_FINAL":
        validate_paper_final_evidence(evidence)
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment.setdefault("status_history", []).append({
        "from": old_status, "to": new_status, "timestamp": timestamp,
        "evidence": dict(evidence or {}),
    })
    experiment["status"] = new_status
    atomic_write_yaml(registry_path, registry)
    return experiment
