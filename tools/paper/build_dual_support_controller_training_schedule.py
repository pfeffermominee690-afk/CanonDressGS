"""Generate the frozen 64-batch/300-step controller data-order artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.dual_support_controller_training_schedule import (  # noqa: E402
    build_schedule,
)


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def atomic_json(path: Path, value: Any, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite schedule artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-repository-root", type=Path, required=True)
    parser.add_argument("--runtime-output-root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    manifest_path = args.training_manifest.resolve()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = build_schedule(manifest)

    cycle_artifact = {
        "training_cycle_sha256": bundle.cycle_sha256,
        "scientific_hash_policy": "CANONICAL_JSON_UTF8_LF_SORT_KEYS_NO_ABSOLUTE_PATHS",
        "scientific_cycle": bundle.scientific_cycle,
    }
    schedule_artifact = {
        "training_300_step_data_order_sha256": bundle.schedule_sha256,
        "training_cycle_sha256": bundle.cycle_sha256,
        "scientific_hash_policy": "CANONICAL_JSON_UTF8_LF_SORT_KEYS_NO_ABSOLUTE_PATHS",
        "scientific_schedule": bundle.scientific_schedule,
    }
    summary = dict(bundle.summary)
    summary.update(
        {
            "task_id": "AAAI27-DUAL-SUPPORT-CONTROLLER-TRAINING-CONTRACT-REPAIR-001",
            "label": "RESEARCH METHOD CONTRACT — NOT PAPER FINAL",
            "source_head": "174655ce6aabcae5d60ce45f3b4be319eb71e914",
            "source_training_manifest_sha256_lf": lf_sha256(manifest_path),
            "source_protocol_sha256_lf_at_generation": lf_sha256(protocol_path),
            "runtime_manifest_participates_in_scientific_hash": False,
            "controller_training_contract": "REPAIRED_AND_FROZEN",
            "next_task": "TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_FROM_REPAIRED_CONTRACT",
            "next_task_started": False,
        }
    )
    runtime_root = args.runtime_repository_root
    runtime_output = args.runtime_output_root
    runtime_manifest = {
        "schema_version": "canondressgs.research.dual_support_controller_training_schedule_runtime.v1",
        "task_id": "AAAI27-DUAL-SUPPORT-CONTROLLER-TRAINING-CONTRACT-REPAIR-001",
        "repository_root": str(runtime_root),
        "training_manifest": str(
            runtime_root
            / "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json"
        ),
        "protocol": str(
            runtime_root
            / "paper_protocol/reviewer_risk/dual_support_controller_protocol.yaml"
        ),
        "cycle_manifest": str(
            runtime_root
            / "paper_protocol/reviewer_risk/dual_support_controller_training_cycle.json"
        ),
        "schedule_manifest": str(
            runtime_root
            / "paper_protocol/reviewer_risk/dual_support_controller_training_schedule_300_steps.json"
        ),
        "formal_output_root": str(runtime_output),
        "training_cycle_sha256": bundle.cycle_sha256,
        "training_300_step_data_order_sha256": bundle.schedule_sha256,
        "participates_in_scientific_hash": False,
        "training_execution_authorized_by_this_manifest": False,
        "paper_final": False,
    }

    outputs = {
        "dual_support_controller_training_cycle.json": cycle_artifact,
        "dual_support_controller_training_schedule_300_steps.json": schedule_artifact,
        "dual_support_controller_training_schedule_runtime.json": runtime_manifest,
        "dual_support_controller_training_schedule_summary.json": summary,
    }
    for name, value in outputs.items():
        atomic_json(output_dir / name, value, replace=args.replace)
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_cycle_sha256": bundle.cycle_sha256,
                "training_300_step_data_order_sha256": bundle.schedule_sha256,
                **bundle.summary["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
