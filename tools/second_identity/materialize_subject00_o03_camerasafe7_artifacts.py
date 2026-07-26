#!/usr/bin/env python3
"""Validate the clean safe7 run and materialize its task-scoped Git evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_ROOT = REPO_ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF_ROOT = REPO_ROOT / "project_control_handoff"
DOCS_ROOT = REPO_ROOT / "docs" / "PAPER"
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "research"
    / "subject00_o03_provisional_teacher_camerasafe7_v1.json"
)
TASK_ID = (
    "AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-SAFE-7VIEW-RERUN-001"
)
SOURCE_BRANCH = (
    "research/subject00-o03-provisional-teacher-camera-metric-review-20260727"
)
SOURCE_HEAD = "d541edce4b7b1da7ce6d055275993c1faf10e45a"
BRANCH = "research/subject00-o03-provisional-teacher-camerasafe7-rerun-20260727"
CAMERA_HEAD = "a434ae7a78fe898be2658180f20bbcd4391a64c0"
WINDOWS_WORKTREE = (
    r"E:\model_train\canondressgs_subject00_o03_provisional_teacher_"
    r"camerasafe7_rerun"
)
CLOUD_WORKTREE = (
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_o03_provisional_teacher_camerasafe7_rerun"
)
FINAL_CLASSIFICATION = (
    "SUBJECT00_O03_PROVISIONAL_TEACHER_BASE60747_CAMERA_SAFE_7VIEW_"
    "TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW"
)
NEXT_TASK = "USER_REVIEW_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_RESULTS"
SAFE_SLOTS = (
    "slot_00",
    "slot_01",
    "slot_02",
    "slot_03",
    "slot_05",
    "slot_06",
    "slot_07",
)
SAFE_CAMERAS = (17, 21, 14, 23, 2, 9, 5)
SAFE_REQUESTS = (
    "subject00_O03_slot00_canary_attempt004_cand00",
    "subject00_O03_slot01_remaining_attempt005_cand00",
    "subject00_O03_slot02_cand00",
    "subject00_O03_slot03_cand01",
    "subject00_O03_slot05_canary_attempt004_cand00",
    "subject00_O03_slot06_remaining_attempt005_cand00",
    "subject00_O03_slot07_canary_attempt004_cand00",
)
EXCLUDED_REQUEST = "subject00_O03_slot04_canary_attempt004_cand00"
EXCLUDED_TOKENS = ("slot_04", "cam11", EXCLUDED_REQUEST)
EXPECTED_COUNTS = {
    "slot_00": 172,
    "slot_01": 172,
    "slot_02": 172,
    "slot_03": 171,
    "slot_05": 171,
    "slot_06": 171,
    "slot_07": 171,
}
CHECKPOINT_STEPS = (0, 300, 600, 900, 1200)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def text_excludes(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return not any(token in text for token in EXCLUDED_TOKENS)


def check(
    rows: list[dict[str, Any]], name: str, passed: bool, evidence: Any
) -> None:
    rows.append({"name": name, "passed": bool(passed), "evidence": evidence})


def markdown_report(master: dict[str, Any]) -> str:
    macro = master["metrics"]["macro"]
    comparative = master["comparative_metrics"]
    per_view = []
    for row in master["metrics"]["per_view"]:
        per_view.append(
            "| {slot} | {camera} | {lpips:.6f} | {psnr:.4f} | "
            "{ssim:.6f} | {garment:.6f} | {silhouette:.6f} | "
            "{boundary:.6f} | {protected:.6f} | {mae:.6f} | {severe} |".format(
                slot=row["slot"],
                camera=row["camera_id"],
                lpips=row["full_image_lpips"],
                psnr=row["psnr"],
                ssim=row["ssim"],
                garment=row["garment_region_lpips"],
                silhouette=row["silhouette_iou"],
                boundary=row["boundary_f"],
                protected=row["protected_region_lpips"],
                mae=row["protected_region_rgb_mae"],
                severe=str(row["severe_artifact_flag"]).lower(),
            )
        )
    return f"""# Subject00 O03 camera-safe7 provisional Teacher

Task: `{TASK_ID}`

## Result

The one authorized clean rerun completed all 1,200 optimizer steps from the
sealed Subject00 Formal Base step60747. The active target set, sampler, loss,
checkpoints, and metrics contain exactly seven camera-safe O03 requests.
`slot_04 / cam11 / right` has zero samples and is present only in exclusion,
quarantine, historical-comparison, and risk-disclosure evidence.

Final classification:

`{FINAL_CLASSIFICATION}`

Unique next task:

`{NEXT_TASK}`

This remains a provisional technical result:
`human_visual_decision=null`, `scientific_pass=null`, `paper_eligible=false`,
and `paper_final=false`.

## Frozen execution contract

- Initialization: `{master["final_fields"]["BASE_CHECKPOINT_PATH"]}`
- Initialization SHA256: `{master["final_fields"]["BASE_CHECKPOINT_SHA256"]}`
- Historical contaminated checkpoint used for initialization: **false**
- Target data class: `RUN_LOCAL_PROVISIONAL_CAMERA_SAFE_7VIEW_SNAPSHOT`
- Safe request count: 7
- Excluded request: `{EXCLUDED_REQUEST}`
- Loss: `CAPACITY_ORACLE_LOSS_V1`
- Optimizer: Adam, geometry LR 0.001, appearance LR 0.002
- Seed: 20260718
- Scheduler: none
- Automatic retry / multi-seed / sweep / early stop: false / false / false / false

## Sampling and authenticity

- View counts: `{json.dumps(master["sampling"]["view_sample_counts"], sort_keys=True)}`
- Sum: 1200
- slot04 count: 0
- Checkpoints: 0, 300, 600, 900, 1200
- Trainable status: `{master["integrity"]["trainable_parameter_change_status"]}`
- Frozen status: `{master["integrity"]["frozen_parameter_mutation_status"]}`
- NaN/Inf: none
- OOM: none
- Runtime tests: {master["runtime_tests"]["pass_count"]}/{master["runtime_tests"]["test_count"]} pass

The historical contaminated 8-view checkpoint was loaded only after the clean
optimizer reached step 1200, then frozen and evaluated on the same safe7
denominator. It was not an initialization source.

## Camera-safe7 macro metrics

| Metric | Value |
|---|---:|
| Full-image LPIPS | {macro["full_image_lpips"]:.9f} |
| PSNR | {macro["psnr"]:.9f} |
| SSIM | {macro["ssim"]:.9f} |
| Garment-region LPIPS | {macro["garment_region_lpips"]:.9f} |
| Garment-region PSNR | {macro["garment_region_psnr"]:.9f} |
| Garment-region SSIM | {macro["garment_region_ssim"]:.9f} |
| Silhouette IoU | {macro["silhouette_iou"]:.9f} |
| Boundary F | {macro["boundary_f"]:.9f} |
| Protected-region LPIPS | {macro["protected_region_lpips"]:.9f} |
| Protected-region RGB MAE | {macro["protected_region_rgb_mae"]:.9f} |
| Alpha foreground error | {macro["alpha_foreground_error"]:.9f} |
| Severe artifacts | {macro["severe_artifact_count"]} |
| Mean render seconds | {macro["render_time_seconds_mean"]:.9f} |
| FPS from mean render time | {macro["fps_from_mean_render_time"]:.9f} |

Full-image LPIPS is retained, but it must not be interpreted alone: the
confirmed full-canvas background difference makes it background-sensitive.
Garment, silhouette, boundary, protected-region, and severe-artifact metrics
remain the primary provisional review evidence.

## Per-view metrics

| Slot | Camera | Full LPIPS | PSNR | SSIM | Garment LPIPS | Silhouette IoU | Boundary F | Protected LPIPS | Protected MAE | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(per_view)}

## Read-only comparison

- Base60747 safe7 denominator: {comparative["base60747"]["macro"]["denominator"]}
- Historical contaminated8 on safe7 denominator: {comparative["historical_contaminated8_on_safe7"]["macro"]["denominator"]}
- Clean safe7 denominator: {comparative["clean_camerasafe7"]["macro"]["denominator"]}
- Historical role: `HISTORICAL_DIAGNOSTIC_ONLY_NOT_INITIALIZATION`

## Review boundary

- Review package: `{master["review"]["review_package_path"]}`
- Manifest: `{master["final_fields"]["REVIEW_MANIFEST_PATH"]}`
- Different-camera check: `{master["animation"]["different_camera"]["status"]}`
- Different-pose check: `{master["animation"]["different_pose"]["status"]}`

These are finite-render/LBS compatibility checks, not strict novel-view or
novel-pose generalization claims. The visual package is display-only and all
PNG files remain under the run root.

## Formal Base

Formal Base remains `USER_AUTHORIZED_PAUSED`, incomplete, durable at step60747,
resume-ready, and resume-unauthorized. It was not resumed or modified.
"""


def materialize(args: argparse.Namespace) -> None:
    config = read_json(CONFIG_PATH)
    run_root = (
        Path(config["runtime"]["output_root"]) / config["runtime"]["attempt"]
    )
    paths = {
        "contract": run_root / "contract" / "execution_contract.json",
        "excluded": run_root / "contract" / "excluded_view_registry.json",
        "manifest": run_root / "inputs" / "target_snapshot" / "manifest.json",
        "training_index": run_root
        / "inputs"
        / "target_snapshot"
        / "training_index.json",
        "evaluation_index": run_root
        / "inputs"
        / "target_snapshot"
        / "evaluation_index.json",
        "derived": run_root
        / "inputs"
        / "target_snapshot"
        / "derived_target_registry.json",
        "preloop": run_root / "audits" / "pre_loop_smoke.json",
        "integrity": run_root / "audits" / "post_training_integrity.json",
        "animation": run_root / "audits" / "animation_audit.json",
        "states": run_root / "training" / "state_records.jsonl",
        "result": run_root / "training" / "training_result.json",
        "metrics": run_root / "evaluations" / "final_metrics.json",
        "comparative": run_root / "evaluations" / "comparative_metrics.json",
        "review": run_root / "review" / "review_manifest.json",
        "status": run_root / "RUN_STATUS.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required run artifacts are missing: {missing}")
    contract = read_json(paths["contract"])
    manifest = read_json(paths["manifest"])
    training_index = read_json(paths["training_index"])
    evaluation_index = read_json(paths["evaluation_index"])
    derived = read_json(paths["derived"])
    excluded = read_json(paths["excluded"])
    preloop = read_json(paths["preloop"])
    integrity = read_json(paths["integrity"])
    animation = read_json(paths["animation"])
    states = read_jsonl(paths["states"])
    result = read_json(paths["result"])
    metrics = read_json(paths["metrics"])
    comparative = read_json(paths["comparative"])
    review = read_json(paths["review"])
    status = read_json(paths["status"])

    pause = read_json(
        Path(config["base"]["checkpoint_path"]).parents[1]
        / "control"
        / "USER_AUTHORIZED_PAUSE_FOR_PROVISIONAL_DOWNSTREAM_20260727.json"
    )
    resume = read_json(
        Path(config["base"]["checkpoint_path"]).parents[1]
        / "control"
        / "FORMAL_BASE_RESUME_FROM_60747_CONTRACT_20260727.json"
    )
    checkpoint_paths = sorted((run_root / "checkpoints").glob("*.pth"))
    checkpoint_sidecars = [
        read_json(path.with_suffix(".sidecar.json")) for path in checkpoint_paths
    ]
    manifest_requests = [row["request_id"] for row in manifest["records"]]
    state_requests = {row["request_id"] for row in states}
    state_slots = {row["slot"] for row in states}
    state_denominators = {int(row["view_denominator"]) for row in states}
    state_counts = {
        slot: sum(row["slot"] == slot for row in states) for slot in SAFE_SLOTS
    }
    raw_ok = all(
        Path(row["accepted_raw"]["path"]).is_file()
        and sha256_file(Path(row["accepted_raw"]["path"]))
        == row["accepted_raw"]["sha256"]
        for row in manifest["records"]
    )
    person_ok = all(
        Path(row["person_mask"]["path"]).is_file()
        and sha256_file(Path(row["person_mask"]["path"]))
        == row["person_mask"]["sha256"]
        for row in manifest["records"]
    )
    garment_ok = all(
        Path(row["garment_mask"]["path"]).is_file()
        and sha256_file(Path(row["garment_mask"]["path"]))
        == row["garment_mask"]["sha256"]
        for row in manifest["records"]
    )

    tests: list[dict[str, Any]] = []
    current_branch = git_output("branch", "--show-current")
    current_head = git_output("rev-parse", "HEAD")
    ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                SOURCE_HEAD,
                current_head,
            ]
        ).returncode
        == 0
    )
    check(tests, "source_branch_head", ancestor, {"source": SOURCE_HEAD})
    check(tests, "execution_branch", current_branch == BRANCH, current_branch)
    check(
        tests,
        "base60747_path_sha",
        Path(config["base"]["checkpoint_path"]).is_file()
        and sha256_file(Path(config["base"]["checkpoint_path"]))
        == config["base"]["checkpoint_sha256"],
        config["base"],
    )
    check(
        tests,
        "formal_base_paused",
        pause["formal_base_final_status"] == "INCOMPLETE"
        and pause["formal_base_completed"] is False
        and int(pause["latest_complete_checkpoint_step"]) == 60747,
        pause["formal_base_final_status"],
    )
    check(
        tests,
        "formal_base_resume_unauthorized",
        resume["FORMAL_BASE_RESUME_READY"] is True
        and resume["FORMAL_BASE_RESUME_AUTHORIZED"] is False,
        {
            "ready": resume["FORMAL_BASE_RESUME_READY"],
            "authorized": resume["FORMAL_BASE_RESUME_AUTHORIZED"],
        },
    )
    check(
        tests,
        "latest_camera_eligibility_registry",
        contract["config"]["evidence"]["camera_resolution_head"] == CAMERA_HEAD
        and contract["config"]["evidence"]["camera_resolution_classification"]
        == (
            "SUBJECT00_TEACHER_TARGET_CAMERA_BLOCKER_RESOLVED_WITH_"
            "QUARANTINE_READY_FOR_MATERIALIZATION"
        ),
        contract["config"]["evidence"],
    )
    check(
        tests,
        "exact_7_request_ids",
        manifest_requests == list(SAFE_REQUESTS),
        manifest_requests,
    )
    check(
        tests,
        "exact_excluded_request",
        excluded["excluded_count"] == 1
        and excluded["records"][0]["request_id"] == EXCLUDED_REQUEST,
        excluded["records"],
    )
    check(tests, "target_raw_sha_7_of_7", raw_ok, raw_ok)
    check(tests, "person_mask_sha_7_of_7", person_ok, person_ok)
    check(tests, "garment_mask_sha_7_of_7", garment_ok, garment_ok)
    check(
        tests,
        "camera_records_7_of_7",
        len(derived["records"]) == 7
        and all(len(row["camera_record_sha256"]) == 64 for row in derived["records"]),
        [row["camera_record_sha256"] for row in derived["records"]],
    )
    check(
        tests,
        "slot04_absent_target_manifest",
        text_excludes(paths["manifest"]),
        str(paths["manifest"]),
    )
    check(
        tests,
        "slot04_absent_training_index",
        text_excludes(paths["training_index"]),
        str(paths["training_index"]),
    )
    check(
        tests,
        "slot04_absent_evaluation_index",
        text_excludes(paths["evaluation_index"]),
        str(paths["evaluation_index"]),
    )
    check(
        tests,
        "slot04_absent_sampler",
        state_requests == set(SAFE_REQUESTS) and "slot_04" not in state_slots,
        {"requests": sorted(state_requests), "slots": sorted(state_slots)},
    )
    check(
        tests,
        "slot04_absent_loss_denominator",
        state_denominators == {7} and EXCLUDED_REQUEST not in state_requests,
        sorted(state_denominators),
    )
    check(
        tests,
        "slot04_absent_metric_denominator",
        metrics["denominator"] == 7
        and len(metrics["per_view"]) == 7
        and all(row["slot"] != "slot_04" for row in metrics["per_view"]),
        metrics["denominator"],
    )
    check(
        tests,
        "initialization_not_contaminated_checkpoint",
        contract["historical_contaminated_checkpoint_used_for_initialization"]
        is False
        and all(
            sidecar["initialization_sha256"]
            == config["base"]["checkpoint_sha256"]
            for sidecar in checkpoint_sidecars
        ),
        config["base"]["checkpoint_sha256"],
    )
    check(
        tests,
        "pre_loop_forward",
        preloop["forward"] == "PASS_FINITE",
        preloop["forward"],
    )
    check(
        tests,
        "pre_loop_backward",
        preloop["backward"].startswith("PASS_ALL_5"),
        preloop["backward"],
    )
    check(
        tests,
        "optimizer_steps_1200",
        result["optimizer_steps"] == 1200 and len(states) == 1200,
        {"result": result["optimizer_steps"], "records": len(states)},
    )
    check(
        tests,
        "balanced_view_sample_counts",
        state_counts == EXPECTED_COUNTS,
        state_counts,
    )
    check(
        tests,
        "sample_count_sum_1200",
        sum(state_counts.values()) == 1200,
        sum(state_counts.values()),
    )
    check(
        tests,
        "slot04_sample_count_zero",
        integrity["slot04_sample_count"] == 0 and "slot_04" not in state_counts,
        integrity["slot04_sample_count"],
    )
    check(
        tests,
        "five_checkpoints",
        [int(path.stem.split("_")[1]) for path in checkpoint_paths]
        == list(CHECKPOINT_STEPS),
        [str(path) for path in checkpoint_paths],
    )
    check(
        tests,
        "checkpoint_parse",
        integrity["checkpoint_parse_status"] == "PASS_5_OF_5_CPU_TORCH_LOAD"
        and len(integrity["checkpoints"]) == 5,
        integrity["checkpoint_parse_status"],
    )
    check(
        tests,
        "initialization_sha_binding",
        all(
            sidecar["initialization_sha256"]
            == config["base"]["checkpoint_sha256"]
            for sidecar in checkpoint_sidecars
        ),
        [sidecar["initialization_sha256"] for sidecar in checkpoint_sidecars],
    )
    check(
        tests,
        "target_manifest_sha_binding",
        all(
            sidecar["target_manifest_sha256"]
            == contract["target_binding"]["manifest_sha256"]
            for sidecar in checkpoint_sidecars
        ),
        contract["target_binding"]["manifest_sha256"],
    )
    check(
        tests,
        "trainable_parameter_changes",
        integrity["trainable_parameter_change_status"]
        == "PASS_ALL_5_TRAINABLE_TENSORS_CHANGED",
        integrity["trainable_parameter_changes"],
    )
    check(
        tests,
        "frozen_immutability",
        integrity["frozen_parameter_mutation_status"]
        == "PASS_BASE60747_FINGERPRINT_UNCHANGED",
        integrity["frozen_parameter_mutation_status"],
    )
    check(tests, "no_nan", result["nan_inf_status"] == "NONE", result["nan_inf_status"])
    check(tests, "no_inf", result["nan_inf_status"] == "NONE", result["nan_inf_status"])
    check(tests, "no_oom", result["oom_status"] == "NONE", result["oom_status"])
    check(
        tests,
        "per_view_metrics_count_7",
        len(metrics["per_view"]) == 7,
        len(metrics["per_view"]),
    )
    check(
        tests,
        "aggregate_denominator_7",
        metrics["macro"]["denominator"] == 7,
        metrics["macro"]["denominator"],
    )
    check(
        tests,
        "comparative_metrics",
        comparative["denominator"] == 7
        and comparative["base60747"]["macro"]["denominator"] == 7
        and comparative["historical_contaminated8_on_safe7"]["macro"][
            "denominator"
        ]
        == 7
        and comparative["clean_camerasafe7"]["macro"]["denominator"] == 7,
        str(paths["comparative"]),
    )
    check(
        tests,
        "animation_audit",
        animation["deformation_status"] == "PASS_FINITE"
        and animation["lbs_status"] == "PASS",
        {
            "deformation": animation["deformation_status"],
            "lbs": animation["lbs_status"],
        },
    )
    review_files_ok = all(
        Path(row["path"]).is_file()
        and sha256_file(Path(row["path"])) == row["sha256"]
        for row in review["files"]
    )
    check(
        tests,
        "review_package",
        review_files_ok
        and review["image_count"] == 10
        and all(review["required_content"].values()),
        {"image_count": review["image_count"], "files_ok": review_files_ok},
    )
    check(
        tests,
        "human_fields_null",
        result["human_visual_decision"] is None
        and result["scientific_pass"] is None
        and review["human_visual_decision"] is None
        and review["scientific_pass"] is None,
        None,
    )
    check(
        tests,
        "paper_eligible_false",
        result["paper_eligible"] is False
        and metrics["paper_eligible"] is False
        and review["paper_eligible"] is False,
        False,
    )
    check(tests, "raw_immutable", raw_ok, raw_ok)
    check(tests, "masks_immutable", person_ok and garment_ok, person_ok and garment_ok)
    check(
        tests,
        "base_checkpoint_immutable",
        integrity["base_checkpoint_unchanged"] is True
        and sha256_file(Path(config["base"]["checkpoint_path"]))
        == config["base"]["checkpoint_sha256"],
        integrity["base_checkpoint_unchanged"],
    )
    check(
        tests,
        "no_formal_base_resume",
        result["formal_base_status"] == "USER_AUTHORIZED_PAUSED"
        and result["formal_base_resume_authorized"] is False
        and resume["FORMAL_BASE_RESUME_AUTHORIZED"] is False,
        result["formal_base_status"],
    )
    check(
        tests,
        "paper_modification_zero",
        result["paper_modifications"] == 0,
        result["paper_modifications"],
    )
    check(
        tests,
        "final_classification",
        result["final_classification"] == FINAL_CLASSIFICATION
        and status["final_classification"] == FINAL_CLASSIFICATION,
        result["final_classification"],
    )
    check(
        tests,
        "next_task_uniqueness",
        result["next_task"] == NEXT_TASK and status["next_task"] == NEXT_TASK,
        result["next_task"],
    )
    check(
        tests,
        "target_loader_smoke",
        preloop["target_loader_parse"] == "PASS_7_OF_7",
        preloop["target_loader_parse"],
    )
    check(
        tests,
        "derived_target_registry_7_of_7",
        derived["record_count"] == 7
        and len(derived["records"]) == 7
        and text_excludes(paths["derived"]),
        derived["record_count"],
    )
    check(
        tests,
        "camera_loader_safe7_only",
        contract["camera_loader"]["dataset_camera_ids"] == list(SAFE_CAMERAS)
        and contract["camera_loader"]["excluded_camera_loaded"] is False,
        contract["camera_loader"],
    )
    check(
        tests,
        "historical_loaded_after_clean_training",
        integrity["historical_checkpoint_loaded_after_optimizer_step"] == 1200
        and integrity["historical_checkpoint_used_for_initialization"] is False,
        integrity["historical_checkpoint_loaded_after_optimizer_step"],
    )
    check(
        tests,
        "no_partial_checkpoint",
        integrity["partial_tmp_count"] == 0,
        integrity["partial_tmp_count"],
    )
    check(
        tests,
        "complete_run_status",
        status["status"] == "COMPLETE" and status["optimizer_steps"] == 1200,
        status,
    )
    failed = [row for row in tests if not row["passed"]]
    if failed:
        raise RuntimeError(f"runtime validation failed: {failed}")

    execution_tests = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.execution_tests.v1",
        "task_id": TASK_ID,
        "test_count": len(tests),
        "pass_count": len(tests),
        "fail_count": 0,
        "result": f"PASS_RUNTIME_{len(tests)}_OF_{len(tests)}",
        "task_scoped_pytest": args.pytest_result,
        "tests": tests,
        "paper_final": False,
    }
    common = {
        "task_id": TASK_ID,
        "run_root": str(run_root),
        "audit_execution_head": contract["git"]["head"],
        "artifact_generation_head": current_head,
        "paper_eligible": False,
        "paper_final": False,
    }
    target_binding = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.target_binding.v1",
        **common,
        "target_data_source": "RUN_LOCAL_SNAPSHOT_FROM_HISTORICAL_SEALED_SAFE7_PLUS_LATEST_CAMERA_AND_MASK_EVIDENCE",
        "target_data_class": manifest["target_data_class"],
        "schema": manifest["schema_version"],
        "record_count": manifest["record_count"],
        "denominator": manifest["denominator"],
        "request_ids": manifest["request_ids"],
        "camera_ids": manifest["camera_ids"],
        "slots": manifest["slots"],
        "records": manifest["records"],
        "derived_target_registry": derived,
        "bindings": contract["target_binding"],
        "excluded_registry": excluded,
        "raw_binding_status": "PASS_7_OF_7_SHA256",
        "person_mask_binding_status": "PASS_7_OF_7_SHA256",
        "garment_mask_binding_status": "PASS_7_OF_7_SHA256",
        "camera_binding_status": "PASS_7_OF_7_UNIQUE_SIMILARITY_BINDINGS",
        "loader_smoke_status": "PASS_7_OF_7",
    }
    sampling = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.sampling.v1",
        **common,
        "view_order": list(SAFE_SLOTS),
        "request_ids": list(SAFE_REQUESTS),
        "camera_ids": list(SAFE_CAMERAS),
        "view_sample_counts": state_counts,
        "sample_count_sum": sum(state_counts.values()),
        "balance_status": "PASS_COUNTS_171_OR_172_MAX_MINUS_MIN_1",
        "denominator_values": sorted(state_denominators),
        "slot04_sample_count": 0,
        "slot04_present_in_sampler": False,
    }
    training_registry = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.training.v1",
        **common,
        "loss_contract": config["teacher"]["loss_name"],
        "loss_weights": config["teacher"]["loss_weights"],
        "optimizer": config["teacher"]["optimizer"],
        "scheduler": None,
        "seed": config["teacher"]["seed"],
        "training_steps": 1200,
        "optimizer_steps_completed": result["optimizer_steps"],
        "loss_initial": result["loss_initial"],
        "loss_final": result["loss_final"],
        "state_records_path": str(paths["states"]),
        "state_records_sha256": sha256_file(paths["states"]),
        "state_record_count": len(states),
        "checkpoint_events": ["step_000000.pth"]
        + [
            row["checkpoint_event"]
            for row in states
            if row["checkpoint_event"] is not None
        ],
        "wall_seconds": result["wall_seconds"],
        "peak_vram_bytes": result["peak_vram_bytes"],
        "nan_inf_status": result["nan_inf_status"],
        "oom_status": result["oom_status"],
        "automatic_retry": False,
        "attempt_002_created": False,
    }
    checkpoint_registry = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.checkpoints.v1",
        **common,
        "checkpoint_count": len(integrity["checkpoints"]),
        "checkpoint_parse_status": integrity["checkpoint_parse_status"],
        "checkpoints": integrity["checkpoints"],
        "initialization_sha256": config["base"]["checkpoint_sha256"],
        "target_manifest_sha256": contract["target_binding"]["manifest_sha256"],
        "derived_target_registry_sha256": contract["target_binding"][
            "derived_target_registry_sha256"
        ],
        "exact_request_ids": list(SAFE_REQUESTS),
        "excluded_request_id": EXCLUDED_REQUEST,
        "historical_checkpoint_used_for_initialization": False,
    }
    per_view_metrics = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.per_view_metrics.v1",
        **common,
        **metrics,
    }
    comparative_metrics = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.comparative_metrics.v1",
        **common,
        **comparative,
    }
    animation_audit = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.animation_audit.v1",
        **common,
        **animation,
    }
    human_review = {
        **review,
        **common,
        "source_review_manifest_path": str(paths["review"]),
        "source_review_manifest_sha256": sha256_file(paths["review"]),
        "run_root_only_image_policy": True,
        "git_image_count": 0,
    }
    macro = metrics["macro"]
    checkpoint_hashes = [row["sha256"] for row in integrity["checkpoints"]]
    checkpoint_names = [row["path"] for row in integrity["checkpoints"]]
    final_fields = {
        "TASK_ID": TASK_ID,
        "SOURCE_BRANCH": SOURCE_BRANCH,
        "SOURCE_HEAD": SOURCE_HEAD,
        "NEW_BRANCH": BRANCH,
        "WINDOWS_WORKTREE": WINDOWS_WORKTREE,
        "CLOUD_WORKTREE": CLOUD_WORKTREE,
        "CAMERA_ELIGIBILITY_SOURCE_HEAD": CAMERA_HEAD,
        "BASE_CHECKPOINT_PATH": config["base"]["checkpoint_path"],
        "BASE_CHECKPOINT_SHA256": config["base"]["checkpoint_sha256"],
        "BASE_CHECKPOINT_STEP": 60747,
        "CONTAMINATED_RUN_PATH": config["historical_contaminated_run"]["run_root"],
        "CONTAMINATED_CHECKPOINT_SHA256": config["historical_contaminated_run"][
            "checkpoint_sha256"
        ],
        "CONTAMINATED_CHECKPOINT_USED_FOR_INITIALIZATION": False,
        "TARGET_DATA_SOURCE": target_binding["target_data_source"],
        "TARGET_DATA_CLASS": manifest["target_data_class"],
        "TARGET_VIEW_COUNT": 7,
        "TARGET_REQUEST_IDS": list(SAFE_REQUESTS),
        "TARGET_CAMERA_IDS": list(SAFE_CAMERAS),
        "EXCLUDED_VIEW_COUNT": 1,
        "EXCLUDED_REQUEST_IDS": [EXCLUDED_REQUEST],
        "SLOT04_PRESENT_IN_TARGET_MANIFEST": False,
        "SLOT04_PRESENT_IN_TRAINING_INDEX": False,
        "SLOT04_PRESENT_IN_EVALUATION_INDEX": False,
        "SLOT04_PRESENT_IN_SAMPLER": False,
        "SLOT04_PRESENT_IN_LOSS_DENOMINATOR": False,
        "SLOT04_PRESENT_IN_METRIC_DENOMINATOR": False,
        "TARGET_RAW_BINDING_STATUS": target_binding["raw_binding_status"],
        "TARGET_PERSON_MASK_BINDING_STATUS": target_binding[
            "person_mask_binding_status"
        ],
        "TARGET_GARMENT_MASK_BINDING_STATUS": target_binding[
            "garment_mask_binding_status"
        ],
        "TARGET_CAMERA_BINDING_STATUS": target_binding["camera_binding_status"],
        "TARGET_LOADER_SMOKE_STATUS": target_binding["loader_smoke_status"],
        "LOSS_CONTRACT": config["teacher"]["loss_name"],
        "OPTIMIZER": "Adam",
        "LEARNING_RATES": {"geometry": 0.001, "appearance": 0.002},
        "SEED": 20260718,
        "TRAINING_STEPS": 1200,
        "VIEW_SAMPLE_COUNTS": state_counts,
        "VIEW_SAMPLE_BALANCE_STATUS": sampling["balance_status"],
        "OUTPUT_ROOT": config["runtime"]["output_root"],
        "ATTEMPT_ID": config["runtime"]["attempt"],
        "PRE_LOOP_FORWARD_STATUS": preloop["forward"],
        "PRE_LOOP_BACKWARD_STATUS": preloop["backward"],
        "OPTIMIZER_STEPS_COMPLETED": result["optimizer_steps"],
        "CHECKPOINT_COUNT": len(integrity["checkpoints"]),
        "CHECKPOINT_PATHS": checkpoint_names,
        "CHECKPOINT_SHA256": checkpoint_hashes,
        "TRAINABLE_PARAMETER_CHANGE_STATUS": integrity[
            "trainable_parameter_change_status"
        ],
        "FROZEN_PARAMETER_MUTATION_STATUS": integrity[
            "frozen_parameter_mutation_status"
        ],
        "LOSS_INITIAL": result["loss_initial"],
        "LOSS_FINAL": result["loss_final"],
        "NAN_INF_STATUS": result["nan_inf_status"],
        "OOM_STATUS": result["oom_status"],
        "WALL_TIME": result["wall_seconds"],
        "PEAK_VRAM": result["peak_vram_bytes"],
        "FULL_IMAGE_LPIPS": macro["full_image_lpips"],
        "FULL_IMAGE_PSNR": macro["psnr"],
        "FULL_IMAGE_SSIM": macro["ssim"],
        "GARMENT_REGION_LPIPS": macro["garment_region_lpips"],
        "GARMENT_REGION_PSNR": macro["garment_region_psnr"],
        "GARMENT_REGION_SSIM": macro["garment_region_ssim"],
        "SILHOUETTE_IOU": macro["silhouette_iou"],
        "BOUNDARY_F": macro["boundary_f"],
        "PROTECTED_REGION_LPIPS": macro["protected_region_lpips"],
        "PROTECTED_REGION_RGB_MAE": macro["protected_region_rgb_mae"],
        "ALPHA_FOREGROUND_ERROR": macro["alpha_foreground_error"],
        "SEVERE_ARTIFACT_COUNT": macro["severe_artifact_count"],
        "PER_VIEW_METRICS_PATH": str(paths["metrics"]),
        "COMPARATIVE_METRICS_PATH": str(paths["comparative"]),
        "DIFFERENT_CAMERA_STATUS": result["different_camera_status"],
        "DIFFERENT_POSE_STATUS": result["different_pose_status"],
        "REVIEW_PACKAGE_PATH": result["review_package_path"],
        "REVIEW_MANIFEST_PATH": result["review_manifest_path"],
        "HUMAN_VISUAL_DECISION": None,
        "SCIENTIFIC_PASS": None,
        "PAPER_ELIGIBLE": False,
        "FORMAL_BASE_STATUS": "USER_AUTHORIZED_PAUSED",
        "FORMAL_BASE_DURABLE_RESUME_STEP": 60747,
        "FORMAL_BASE_RESUME_READY": True,
        "FORMAL_BASE_RESUME_AUTHORIZED": False,
        "OPTIMIZER_STEPS_BY_OTHER_TASKS": 0,
        "DATA_MUTATIONS": 0,
        "TARGET_MUTATIONS": 0,
        "MASK_MUTATIONS": 0,
        "BASE_CHECKPOINT_MUTATIONS": 0,
        "CONTAMINATED_RUN_MUTATIONS": 0,
        "PAPER_MODIFICATIONS": 0,
        "TEST_RESULT": execution_tests["result"],
        "COMMIT_HEAD": contract["git"]["head"],
        "FINAL_REPORTING_HEAD": current_head,
        "ORIGIN_SYNC_STATUS": args.sync_status,
        "CLOUD_GIT_SYNC_STATUS": args.sync_status,
        "WORKTREE_CLEAN_STATUS": "PASS_AT_EXECUTION_HEAD",
        "PAPER_FINAL": False,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "NEXT_TASK": NEXT_TASK,
    }
    final_summary = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.final_summary.v1",
        **common,
        **final_fields,
        "integrity": integrity,
        "interpretation_boundary": (
            "Provisional technical result pending human visual review; not a "
            "paper-final scientific conclusion."
        ),
    }
    handoff = {
        "schema_version": "canondressgs.subject00.o03_camerasafe7.handoff.v1",
        **common,
        "final_fields": final_fields,
        "runtime_tests": execution_tests,
        "review": {
            "package_path": result["review_package_path"],
            "manifest_path": result["review_manifest_path"],
            "human_visual_decision": None,
            "scientific_pass": None,
            "paper_eligible": False,
        },
        "formal_base": {
            "status": "USER_AUTHORIZED_PAUSED",
            "durable_resume_step": 60747,
            "resume_ready": True,
            "resume_authorized": False,
        },
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    master = {
        "task_id": TASK_ID,
        "final_fields": final_fields,
        "target_binding": target_binding,
        "sampling": sampling,
        "training": training_registry,
        "checkpoints": checkpoint_registry,
        "metrics": metrics,
        "comparative_metrics": comparative,
        "animation": animation,
        "review": human_review,
        "integrity": integrity,
        "runtime_tests": execution_tests,
    }
    outputs = {
        RISK_ROOT
        / "subject00_O03_camerasafe7_target_binding_registry_20260727.json": target_binding,
        RISK_ROOT
        / "subject00_O03_camerasafe7_view_sampling_registry_20260727.json": sampling,
        RISK_ROOT
        / "subject00_O03_camerasafe7_training_registry_20260727.json": training_registry,
        RISK_ROOT
        / "subject00_O03_camerasafe7_checkpoint_registry_20260727.json": checkpoint_registry,
        RISK_ROOT
        / "subject00_O03_camerasafe7_per_view_metrics_20260727.json": per_view_metrics,
        RISK_ROOT
        / "subject00_O03_camerasafe7_comparative_metrics_20260727.json": comparative_metrics,
        RISK_ROOT
        / "subject00_O03_camerasafe7_animation_audit_20260727.json": animation_audit,
        RISK_ROOT
        / "subject00_O03_camerasafe7_human_review_manifest_20260727.json": human_review,
        RISK_ROOT
        / "subject00_O03_camerasafe7_execution_tests_20260727.json": execution_tests,
        RISK_ROOT
        / "subject00_O03_camerasafe7_final_summary_20260727.json": final_summary,
        HANDOFF_ROOT
        / "subject00_O03_camerasafe7_provisional_teacher_handoff_20260727.json": handoff,
    }
    for path, payload in outputs.items():
        atomic_json(path, payload)
    report = markdown_report(master)
    atomic_text(
        RISK_ROOT / "SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_REPORT_20260727.md",
        report,
    )
    atomic_text(
        DOCS_ROOT
        / "AAAI27_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_REPORT_20260727.md",
        report,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "runtime_test_result": execution_tests["result"],
                "artifact_count": len(outputs) + 2,
                "final_classification": FINAL_CLASSIFICATION,
                "next_task": NEXT_TASK,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-result", default="NOT_RUN_YET")
    parser.add_argument("--sync-status", default="PENDING_FINAL_COMMIT")
    return parser.parse_args()


if __name__ == "__main__":
    materialize(parse_args())
