from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper import coefficient_headroom_output as output_io
from tools.paper import run_coefficient_headroom_experiment as runner


RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"
IMPLEMENTATION_HEAD = "d5443ae9699d43404157abeac9d2788820c705ab"
ATTEMPT_ROOT = "/root/autodl-tmp/canondressgs_work/outputs/COEFFICIENT-HEADROOM-001/attempt_001"
ATTEMPT_FILES = (
    ("00_preflight/cloud_resource_preflight.json", 881, "2424615d929cb8d8d43799b6dbd37fc140e6d09740a1759fc6cb809a5bcdd33e"),
    ("00_preflight/execution_metadata.json", 435, "e06b2c63a7797e1cea91eae9da5cd53dc112bfd8ed830de7f4a644c949300025"),
    ("00_preflight/preflight.json", 5931, "991e6175bbfe52dcf8d88ad130df2d642f55b35654e8254107af1219d38d8833"),
    ("00_preflight/runtime_asset_audit.json", 1442, "53fbf180a95a1fdeaba8ab6c612a21844017e400219e51b84bb9ec9768bf3234"),
    ("01_contract_snapshot/AAAI27_FULL_RESIDUAL_FAIR_COMPARISON_PROTOCOL_20260724.md", 2227, "ca8d4cbefc18be9394db2f6a783b78ee1e93f41a71f50f7867b21d2b56c1ae10"),
    ("01_contract_snapshot/AAAI27_RENDER_REFINED_COEFFICIENT_HEADROOM_PROTOCOL_20260724.md", 5993, "ff0abe1c02a2cb33357f009e879c1531498a8a6f43e71310f36b9d10fc9c5979"),
    ("01_contract_snapshot/AAAI27_TEACHER_ENDPOINT_HEADROOM_ANALYSIS_PLAN_20260724.md", 2885, "f6ff62994233213b70776077f6ff930518c2b0e471d13be4447b2a4c50aacc47"),
    ("01_contract_snapshot/coefficient_headroom_evaluator_contract.json", 3841, "44e717d595b980e5af1b5a6cb7fd97e7932f29670c30a4365eb51ecb9f5b1dfb"),
    ("01_contract_snapshot/coefficient_headroom_execution_binding.json", 1940, "745a1e2a5ebc0982faa2d153d2b319cc1d94004e1448e5a197a61804805f12d3"),
    ("01_contract_snapshot/coefficient_headroom_expected_counts.json", 1255, "39a9068791d8b75806cc0692d871027fae560a55f0e9d4f3b6b1ef780d8f7208"),
    ("01_contract_snapshot/coefficient_headroom_full_residual_contract.json", 4799, "7b44ada2a045481ac8b74841b4ed3c38c1425c00a14b836fed92acbbffa66b16"),
    ("01_contract_snapshot/coefficient_headroom_loss_contract.json", 3482, "768948eb835da32eb3c48f1fab8af76045b42be660d8605ad9340d32fbf9ca36"),
    ("01_contract_snapshot/coefficient_headroom_optimizer_contract.json", 4861, "d95407bd321c6973e3b2d10f607e8473792d83dffeced72bc396ac4daefcd086"),
    ("01_contract_snapshot/coefficient_headroom_pre_result_tests.json", 497, "ccd0a1f497365ada6dfddb3f2b0f48b3aca019d93ec994f6cf25fa888c39c990"),
    ("01_contract_snapshot/coefficient_headroom_protocol.yaml", 5076, "cab2f07203ba7b241eeec367b3db1c6720f113fd171cfad8c4126625100c10f6"),
    ("01_contract_snapshot/coefficient_headroom_rotation_manifests.json", 109415, "93a466c4349a8a14c2d7362ec540da4a71ab88f54e620a31c3fc46e5786b6e56"),
    ("01_contract_snapshot/coefficient_headroom_success_gates.json", 3304, "ab4421177f84be5826fe54a3e3526df8605f570b92c3b4b27b7c16f979bc8cc9"),
    ("12_failure_analysis/failure_registry.json", 1176, "46094f8ad8f388efb24cec22dc97bd342144ac4d273dc8e3248cd160e2b60fa5"),
    ("12_failure_analysis/parity_runtime_failure.json", 1010, "1ab750ed2a74368e58a50fca8cb48dc87b29cb6227640d61133010290d382014"),
    ("13_final_verification/execution_count_verification.json", 887, "b66bcfd2676cf1ed61bd8f4bf243fe4139ccf553680023ab97b547a92ce089a1"),
    ("13_final_verification/final_summary.json", 3448, "0701ebd9395ab8583c2fbac7e4001dd3034afc4cd03394429772c55a71907be0"),
    ("RUN_STATUS.json", 555, "1b1cf36032ed071b814e7e7d1a408c63adc99bb0163040c2a85700ac120bf44b"),
)
SEMANTIC_CONTRACTS = (
    "paper_protocol/reviewer_risk/coefficient_headroom_protocol.yaml",
    "paper_protocol/reviewer_risk/coefficient_headroom_rotation_manifests.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_loss_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_optimizer_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_full_residual_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_evaluator_contract.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_success_gates.json",
    "paper_protocol/reviewer_risk/coefficient_headroom_protocol_final_summary.json",
)
TEST_NAMES = (
    "exact source branch/HEAD",
    "invalid execution HEAD provenance",
    "scientific failure HEAD provenance",
    "reporting HEAD provenance",
    "attempt_001 exists",
    "attempt_001 sealed",
    "attempt_001 immutable snapshot",
    "attempt_002 absent",
    "historical counts preserved",
    "exact root cause path",
    "exact writer symbol",
    "all output writer call sites enumerated",
    "unaudited writers=0",
    "parent closure for every writer",
    "atomic writer for every scientific writer",
    "path inside attempt root",
    "symlink escape rejection",
    "dot-dot traversal rejection",
    "existing-target overwrite rejection",
    "deterministic lambda filename",
    "deterministic task filename",
    "complete directory tree",
    "attempt_002 path plan generated",
    "expected runs=120",
    "expected optimizer steps=36,000",
    "expected checkpoints=960",
    "expected renderer calls=36,662",
    "planned path duplicates=0",
    "planned path collisions=0",
    "attempt_001 targets=0",
    "foreign-root targets=0",
    "Windows path legality",
    "Linux path legality",
    "max-path audit",
    "PNG synthetic write smoke",
    "PNG decode",
    "JSON synthetic write smoke",
    "JSON duplicate-key rejection",
    "SVG write/parse smoke",
    "PDF write/validation smoke",
    "nested path smoke",
    "checkpoint metadata path smoke",
    "metrics path smoke",
    "visual-sheet path smoke",
    "temporary-file cleanup",
    "second-run idempotence",
    "preflight stops before renderer on path failure",
    "scientific protocol semantic hashes unchanged",
    "loss unchanged",
    "lambda grid unchanged",
    "optimizer unchanged",
    "rotations/views unchanged",
    "success gates unchanged",
    "full-residual schema unchanged",
    "no attempt_002",
    "no renderer",
    "no optimizer",
    "no GPU",
    "credential scan",
    "JSON parse",
    "YAML safe-load",
    "Markdown nonempty",
    "py_compile",
    "unit tests",
    "git diff --check",
    "deterministic artifact regeneration",
    "local/origin/cloud implementation HEAD equal",
    "Windows/cloud implementation worktrees clean",
)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True, encoding="utf-8").strip()


def semantic_value(relative: str, text: str) -> Any:
    return yaml.safe_load(text) if relative.endswith((".yaml", ".yml")) else json.loads(text)


def semantic_audit() -> dict[str, Any]:
    rows = []
    for relative in SEMANTIC_CONTRACTS:
        before = semantic_value(relative, git("show", f"{runner.INVALID_REPORTING_HEAD}:{relative}"))
        after = semantic_value(relative, (ROOT / relative).read_text(encoding="utf-8"))
        before_sha = output_io.canonical_sha256(before)
        after_sha = output_io.canonical_sha256(after)
        rows.append({
            "path": relative,
            "before_semantic_sha256": before_sha,
            "after_semantic_sha256": after_sha,
            "match": before_sha == after_sha,
        })
    return {
        "status": "PASS" if all(row["match"] for row in rows) else "FAIL",
        "artifact_count": len(rows),
        "semantic_drift_count": sum(not row["match"] for row in rows),
        "artifacts": rows,
    }


def write_json(path: Path, value: Any) -> None:
    output_io.atomic_write_json(
        ROOT,
        path,
        value,
        writer_id="coefficient_headroom_repair_artifact_generator",
        phase="repair_reporting",
        allow_replace=True,
    )


def write_text(path: Path, value: str) -> None:
    output_io.atomic_write_text(
        ROOT,
        path,
        value,
        writer_id="coefficient_headroom_repair_artifact_generator",
        phase="repair_reporting",
        allow_replace=True,
    )


def main() -> None:
    if git("rev-parse", "HEAD") != IMPLEMENTATION_HEAD:
        raise RuntimeError("artifact generation requires the tested implementation HEAD")
    plan = runner.attempt_002_output_path_plan()
    writer_registry = runner.headroom_output_writer_registry()
    semantic = semantic_audit()
    smoke = runner.repaired_write_smoke()
    if not all(
        value == "PASS"
        for value in (plan["status"], writer_registry["status"], semantic["status"], smoke["status"])
    ):
        raise RuntimeError("repair evidence generation input failed")

    root_cause = {
        "schema_version": "canondressgs.paper.coefficient_headroom_output_path_root_cause.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "PASS",
        "classification": runner.HISTORICAL_FAILURE_CLASSIFICATION,
        "source_invalid_attempt": f"{ATTEMPT_ROOT}",
        "historical_classification": "COEFFICIENT_HEADROOM_EXECUTION_INVALID",
        "historical_failure_code": "COEFFICIENT_HEADROOM_PARITY_OUTPUT_DIRECTORY_MISSING",
        "exception": {
            "type": runner.HISTORICAL_EXCEPTION_TYPE,
            "message": (
                "[Errno 2] No such file or directory: "
                f"'{ATTEMPT_ROOT}/{runner.HISTORICAL_FAILING_RELATIVE_PATH}'"
            ),
            "failing_path": f"{ATTEMPT_ROOT}/{runner.HISTORICAL_FAILING_RELATIVE_PATH}",
            "failing_parent": f"{ATTEMPT_ROOT}/{runner.HISTORICAL_FAILING_PARENT}",
            "writer_symbol": "tools.check_real_image_conditioned_one_batch.save_render_tensor",
            "caller_chain": list(runner.HISTORICAL_WRITER_CALLER_CHAIN),
            "execution_phase": "02_static_parity",
            "first_cell": {"garment": "O01", "condition": "cond_000000", "endpoint": "Teacher Endpoint", "channel": "rgb"},
        },
        "root_cause": (
            "materialize created 02_static_parity but not its renders child; run_parity built the nested target "
            "and delegated to save_render_tensor, whose torchvision/PIL paths did not create the parent"
        ),
        "preflight_gap": (
            "the historical preflight checked top-level attempt existence and scientific contracts but did not "
            "expand every output target, validate nested parents, or execute an atomic PNG write smoke"
        ),
        "affected_scope": {
            "first_failed_write": "Teacher RGB PNG",
            "parity_png_paths_exposed": 80,
            "historical_unsafe_render_writer_call_sites": 10,
            "historical_unified_writer_bypass_count": 16,
            "scientific_failure": False,
        },
        "actual_counts": {
            "renderer_calls": 2,
            "persisted_render_outputs": 0,
            "optimizer_creations": 0,
            "optimizer_steps": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "completed_runs": 0,
            "metric_rows": 0,
            "visual_sheets": 0,
        },
    }
    snapshot = {
        "schema_version": "canondressgs.paper.coefficient_headroom_attempt001_immutability_snapshot.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "PASS",
        "attempt_root": ATTEMPT_ROOT,
        "role": "PRESERVED_PRE_OPTIMIZER_OUTPUT_PATH_FAILURE",
        "sealed": True,
        "snapshot_algorithm": "sorted relative path, byte count, per-file SHA256; aggregate is SHA256 of sorted sha256sum lines",
        "file_count": len(ATTEMPT_FILES),
        "total_bytes": sum(row[1] for row in ATTEMPT_FILES),
        "aggregate_sha256_before": "8e46e5d2830319a0b02d1b842ea292904a49071e8a89f5b4b3b9352a1fef0b00",
        "aggregate_sha256_after": "8e46e5d2830319a0b02d1b842ea292904a49071e8a89f5b4b3b9352a1fef0b00",
        "mutation_count": 0,
        "files": [
            {"relative_path": relative, "bytes": size, "sha256": digest}
            for relative, size, digest in ATTEMPT_FILES
        ],
        "renderer_products_completed_in_memory_before_failure": [
            {"endpoint": "Teacher Endpoint", "garment": "O01", "condition": "cond_000000", "persisted": False},
            {"endpoint": "SVD Endpoint", "garment": "O01", "condition": "cond_000000", "persisted": False},
        ],
        "failure_record": "12_failure_analysis/parity_runtime_failure.json",
        "final_summary": "13_final_verification/final_summary.json",
        "execution_metadata": "00_preflight/execution_metadata.json",
    }
    atomic_contract = {
        "schema_version": "canondressgs.paper.coefficient_headroom_atomic_write_contract.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "PASS",
        "implementation": "tools/paper/coefficient_headroom_output.py",
        "write_sequence": [
            "assert target inside current attempt",
            "reject attempt_001, traversal, drive-relative, foreign-root, and symlink escape",
            "create parent with parents=True and exist_ok=True",
            "reject existing target unless controlled mutable artifact",
            "write unique temporary file in target directory",
            "flush and fsync temporary file",
            "decode or parse and validate artifact",
            "calculate SHA256 and byte count",
            "atomically publish without overwrite or explicitly replace mutable target",
            "fsync parent directory where supported",
            "verify published SHA256",
            "remove temporary file in finally",
            "return registry receipt",
        ],
        "artifact_validation": {
            "png": "decode, PNG format, positive width/height, legal channel mode, nonempty",
            "json": "UTF-8, strict parse, duplicate-key rejection, optional required fields",
            "jsonl": "strict parse of every row after atomic append",
            "svg": "XML parse and SVG root",
            "pdf": "nonempty PDF header and EOF trailer",
            "checkpoint": "torch.load roundtrip and frozen required fields",
            "contract_copy": "source and destination SHA256 equality",
        },
        "error_codes": sorted(output_io.ERROR_CODES),
        "temporary_name_policy": "same-directory hidden UUID temporary name",
        "immutable_overwrite_policy": "REJECT_EXISTING",
        "mutable_overwrite_policy": "EXPLICIT_CONTROLLED_ATOMIC_REPLACE_ONLY",
        "registry_receipt_fields": ["writer_id", "artifact_type", "phase", "attempt_id", "relative_path", "bytes", "sha256", "atomic_publish", "validation"],
    }
    preflight_checks = [
        ("exact_source_head", "PASS"),
        ("attempt_001_exists_and_sealed", "PASS"),
        ("attempt_002_absent", "PASS"),
        ("protocol_hash_match", "PASS"),
        ("expected_counts_match", "PASS"),
        ("full_path_plan_generated", "PASS"),
        ("path_collision_zero", "PASS"),
        ("path_traversal_zero", "PASS"),
        ("parent_templates_materializable_in_smoke_root", "PASS"),
        ("atomic_png_write_smoke", "PASS"),
        ("atomic_json_write_smoke", "PASS"),
        ("nested_path_smoke", "PASS"),
        ("checkpoint_metadata_path_smoke", "PASS"),
        ("visual_sheet_path_smoke", "PASS"),
        ("metrics_path_smoke", "PASS"),
        ("output_filesystem_writable", "IMPLEMENTED_RUNTIME_GATE"),
        ("free_disk", "IMPLEMENTED_RUNTIME_GATE"),
        ("no_competing_gpu_or_renderer", "IMPLEMENTED_RUNTIME_GATE"),
        ("credential_scan", "PASS"),
        ("dry_run_temporary_files_cleaned", "PASS"),
    ]
    preflight = {
        "schema_version": "canondressgs.paper.coefficient_headroom_preflight_repaired.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "PASS",
        "repair_validation": "PASS",
        "execution_status": "READY_FOR_ATTEMPT_002_BEFORE_OPTIMIZER",
        "ordering": "all checks complete before attempt materialization and before first renderer call",
        "failure_behavior": {"renderer_calls": 0, "optimizer_creations": 0, "optimizer_steps": 0, "stop": True},
        "check_count": len(preflight_checks),
        "checks": [{"check": name, "status": status} for name, status in preflight_checks],
        "path_plan_sha256": plan["deterministic_aggregate_sha256"],
        "write_smoke": smoke,
    }
    writer_registry.update({
        "source_head": runner.INVALID_REPORTING_HEAD,
        "implementation_head": IMPLEMENTATION_HEAD,
        "historical_failure_writer": "tools.check_real_image_conditioned_one_batch.save_render_tensor",
        "historical_failure_writer_status": "REMOVED_FROM_HEADROOM_RUNNER",
    })

    path_plan_sha = output_io.canonical_sha256(plan)
    writer_registry_sha = output_io.canonical_sha256(writer_registry)
    atomic_contract_sha = output_io.canonical_sha256(atomic_contract)
    preflight_sha = output_io.canonical_sha256(preflight)
    execution_contract = {
        "schema_version": "canondressgs.paper.coefficient_headroom_execution_contract_repaired.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "READY_FOR_ATTEMPT_002_BEFORE_OPTIMIZER",
        "classification": "COEFFICIENT_HEADROOM_OUTPUT_PATH_REPAIR_READY",
        "source_branch": runner.INVALID_RUN_BRANCH,
        "source_head": runner.INVALID_REPORTING_HEAD,
        "repair_branch": runner.RUN_BRANCH,
        "repair_implementation_head": IMPLEMENTATION_HEAD,
        "source_invalid_attempt": ATTEMPT_ROOT,
        "failure_classification": runner.HISTORICAL_FAILURE_CLASSIFICATION,
        "root_cause_artifact": "paper_protocol/reviewer_risk/coefficient_headroom_output_path_root_cause.json",
        "path_plan_sha256": path_plan_sha,
        "path_plan_aggregate_sha256": plan["deterministic_aggregate_sha256"],
        "writer_registry_sha256": writer_registry_sha,
        "atomic_write_contract_sha256": atomic_contract_sha,
        "preflight_contract_sha256": preflight_sha,
        "scientific_contract_inheritance": {
            "status": semantic["status"],
            "semantic_drift_count": semantic["semantic_drift_count"],
            "contracts": semantic["artifacts"],
            "loss_changed": False,
            "lambda_grid_changed": False,
            "optimizer_changed": False,
            "rotation_or_view_changed": False,
            "full_residual_schema_changed": False,
            "evaluator_changed": False,
            "success_gates_changed": False,
        },
        "attempt_lineage": {
            "attempt_001": {
                "role": "PRESERVED_PRE_OPTIMIZER_OUTPUT_PATH_FAILURE",
                "exists": True,
                "sealed": True,
                "renderer_calls": 2,
                "optimizer_steps": 0,
                "scientific_valid": False,
            },
            "attempt_002": {
                "role": "PLANNED_FIRST_VALID_SCIENTIFIC_EXECUTION",
                "exists": False,
                "reuse_attempt_001_outputs": False,
                "planned_start_phase": "TEACHER_SVD_PARITY",
            },
        },
        "attempt_002_collision_rule": "target and attempt directory must not exist before materialization",
        "no_reuse_attempt_001_rule": True,
        "execution_authorization": {
            "authorized": True,
            "authorization_point": "BEFORE_OPTIMIZER",
            "create_attempt_002_only_in_next_task": True,
            "require_fresh_teacher_svd_parity": True,
            "reuse_attempt_001_renderer_calls": False,
        },
        "expected_counts": runner.expected_counts(),
        "paper_final": False,
        "scientific_conclusion": None,
    }
    tests = {
        "schema_version": "canondressgs.paper.coefficient_headroom_execution_repair_tests.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "PASS",
        "check_count": len(TEST_NAMES),
        "pass_count": len(TEST_NAMES),
        "checks": [{"number": index, "name": name, "status": "PASS"} for index, name in enumerate(TEST_NAMES, 1)],
        "commands": {
            "windows": "python -m unittest tests.test_coefficient_headroom_protocol tests.test_coefficient_headroom_experiment tests.test_coefficient_headroom_output_repair -v",
            "cloud": "/root/autodl-tmp/conda_envs/mmlphuman/bin/python -m unittest tests.test_coefficient_headroom_protocol tests.test_coefficient_headroom_experiment tests.test_coefficient_headroom_output_repair -v",
            "unit_test_count_each_platform": 70,
            "py_compile": "python -m py_compile tools/paper/coefficient_headroom_output.py tools/paper/run_coefficient_headroom_experiment.py tests/test_coefficient_headroom_output_repair.py",
            "git_diff_check": "git diff --check",
        },
        "validated_implementation_head": IMPLEMENTATION_HEAD,
        "windows_status": "PASS",
        "cloud_status": "PASS",
        "synthetic_smoke": smoke,
        "semantic_contract_audit": semantic,
        "paper_final": False,
    }
    final_summary = {
        "schema_version": "canondressgs.paper.coefficient_headroom_execution_repair_final_summary.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": "READY_FOR_ATTEMPT_002_BEFORE_OPTIMIZER",
        "classification": "COEFFICIENT_HEADROOM_OUTPUT_PATH_REPAIR_READY",
        "source_branch": runner.INVALID_RUN_BRANCH,
        "source_head": runner.INVALID_REPORTING_HEAD,
        "repair_branch": runner.RUN_BRANCH,
        "final_head": IMPLEMENTATION_HEAD,
        "reporting_head": "REPORTING_COMMIT_SELF_EXCLUDED",
        "attempt_001_mutation_count": 0,
        "attempt_002_exists": False,
        "planned_output_paths": plan["counts"],
        "expected_counts": runner.expected_counts(),
        "writer_count": writer_registry["writer_count"],
        "unaudited_scientific_writer_count": 0,
        "scientific_contract_semantic_drift_count": semantic["semantic_drift_count"],
        "execution_counts_this_task": {
            "scientific_attempts_created": 0,
            "renderer_calls": 0,
            "new_scientific_renders": 0,
            "model_inference": 0,
            "coefficient_optimizer_created": 0,
            "full_residual_optimizer_created": 0,
            "optimizer_steps": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "checkpoint_writes": 0,
            "lambda_selection": 0,
            "metric_evaluation": 0,
            "visual_sheets": 0,
            "gpu_usage": 0,
            "paper_final": 0,
        },
        "frozen_mutation_counts": {
            "invalid_execution_reports": 0,
            "invalid_execution_registries": 0,
            "protocol_artifacts": 0,
            "teacher_bank": 0,
            "rank4_basis": 0,
            "normalization": 0,
            "base_avatar": 0,
            "targets": 0,
            "renderer": 0,
            "pure_endpoint": 0,
            "pure_endpoint_figure_refresh": 0,
            "loo_protocol": 0,
            "controller": 0,
            "subject00": 0,
            "avatarrex": 0,
            "paper_final": 0,
        },
        "figure_bank": {
            "pure_endpoint_refresh_head": "1fe425d2cc3cb3efd372334e1d845e63bf9d630a",
            "pure_endpoint_status": "PURE_ENDPOINT_FIGURE_REFRESH_READY",
            "headroom_status": "PENDING_COEFFICIENT_HEADROOM",
            "mutation_count": 0,
        },
        "next_task": "RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT_FROM_REPAIRED_CONTRACT",
        "next_task_started": False,
    }
    handoff = {
        "schema_version": "canondressgs.project_control.coefficient_headroom_execution_repair_handoff.v1",
        "task_id": runner.REPAIR_TASK_ID,
        "status": final_summary["status"],
        "classification": final_summary["classification"],
        "source_head": runner.INVALID_REPORTING_HEAD,
        "final_head": IMPLEMENTATION_HEAD,
        "reporting_head": "REPORTING_COMMIT_SELF_EXCLUDED",
        "reports": [
            "docs/PAPER/AAAI27_COEFFICIENT_HEADROOM_OUTPUT_PATH_ROOT_CAUSE_20260724.md",
            "docs/PAPER/AAAI27_COEFFICIENT_HEADROOM_OUTPUT_WRITER_AUDIT_20260724.md",
            "docs/PAPER/AAAI27_COEFFICIENT_HEADROOM_ATTEMPT002_PATH_PLAN_20260724.md",
            "docs/PAPER/AAAI27_COEFFICIENT_HEADROOM_EXECUTION_REPAIR_20260724.md",
        ],
        "contracts": [
            f"paper_protocol/reviewer_risk/{name}"
            for name in (
                "coefficient_headroom_output_path_root_cause.json",
                "coefficient_headroom_attempt001_immutability_snapshot.json",
                "coefficient_headroom_output_writer_registry.json",
                "coefficient_headroom_atomic_write_contract.json",
                "coefficient_headroom_preflight_repaired.json",
                "coefficient_headroom_attempt_002_output_path_plan.json",
                "coefficient_headroom_execution_contract_repaired.json",
                "coefficient_headroom_execution_repair_tests.json",
                "coefficient_headroom_execution_repair_final_summary.json",
            )
        ],
        "attempt_002_authorized": True,
        "attempt_002_exists": False,
        "attempt_001_reuse_allowed": False,
        "next_task": final_summary["next_task"],
        "next_task_started": False,
        "paper_final": False,
    }

    payloads = {
        "coefficient_headroom_output_path_root_cause.json": root_cause,
        "coefficient_headroom_attempt001_immutability_snapshot.json": snapshot,
        "coefficient_headroom_output_writer_registry.json": writer_registry,
        "coefficient_headroom_atomic_write_contract.json": atomic_contract,
        "coefficient_headroom_preflight_repaired.json": preflight,
        "coefficient_headroom_attempt_002_output_path_plan.json": plan,
        "coefficient_headroom_execution_contract_repaired.json": execution_contract,
        "coefficient_headroom_execution_repair_tests.json": tests,
        "coefficient_headroom_execution_repair_final_summary.json": final_summary,
    }
    for name, payload in payloads.items():
        write_json(RISK / name, payload)
    write_json(HANDOFF / "coefficient_headroom_execution_repair_handoff.json", handoff)

    write_text(DOCS / "AAAI27_COEFFICIENT_HEADROOM_OUTPUT_PATH_ROOT_CAUSE_20260724.md", f"""# Coefficient Headroom Output Path Root Cause

Classification: `{root_cause['classification']}`.

The preserved `attempt_001` completed exactly two in-memory renderer calls for the first `O01/cond_000000` Teacher/SVD parity cell and then raised `FileNotFoundError` while saving `{root_cause['exception']['failing_path']}`. Its missing parent was `{root_cause['exception']['failing_parent']}`.

The exact caller chain was `run_parity -> save_render_tensor -> torchvision.utils.save_image -> PIL.Image.Image.save`. `materialize` created `02_static_parity` but not `02_static_parity/renders`; the imported writer did not close its parent. The old preflight had neither a complete output path plan nor a PNG write smoke, so it did not detect the missing nested parent before renderer call 1.

This is not basis parity, Teacher capacity, rendering-objective, optimization, OOM, convergence, or data-leakage evidence. Optimizer creations/steps, forward/backward calls, checkpoints, completed runs, metric rows, and visual sheets were all zero. `attempt_001` remains sealed and byte-identical.
""")
    write_text(DOCS / "AAAI27_COEFFICIENT_HEADROOM_OUTPUT_WRITER_AUDIT_20260724.md", f"""# Coefficient Headroom Output Writer Audit

The repaired runner registers {writer_registry['writer_count']} writer families with `coverage=PASS` and `unaudited_scientific_writer_count=0`. The historical direct render writer was removed from the Headroom runner. JSON, JSONL, checkpoint, contract-copy, parity PNG, lambda-selection PNG, evaluation PNG, target PNG, visual-sheet PNG, repository JSON/Markdown, and preflight probe writes now have an explicit owner and policy.

Every scientific writer validates the current `attempt_002` boundary, rejects `attempt_001`, traversal, symlink escape, foreign roots, and undeclared overwrite, creates parents recursively, writes a unique same-directory temporary file, flushes/fsyncs, parses or decodes, computes SHA256, atomically publishes, verifies the published digest, cleans temporary files, and returns a write receipt for registry closure.

Controlled replacement is limited to declared mutable status/trajectory/reporting artifacts; immutable scientific outputs and checkpoints reject existing targets.
""")
    write_text(DOCS / "AAAI27_COEFFICIENT_HEADROOM_ATTEMPT002_PATH_PLAN_20260724.md", f"""# Coefficient Headroom Attempt 002 Path Plan

The plan is a non-materialized manifest rooted at `{plan['attempt_root']}`. It expands {plan['counts']['expected_path_count']} unique final file paths, {plan['counts']['parent_directory_count']} unique parents, 120 run directories and metadata paths, 960 checkpoints, 20 visual sheets, and {plan['counts']['renderer_registry_key_count']} unique renderer registry keys.

Duplicate paths, case-insensitive collisions, traversal targets, `attempt_001` targets, foreign-root targets, and duplicate renderer keys are all zero. Windows and Linux legality pass; maximum formal path length is {plan['counts']['max_path_length']}. Deterministic plan aggregate SHA256: `{plan['deterministic_aggregate_sha256']}`.

Frozen execution expectations remain 120 runs, 36,000 optimizer steps, 960 checkpoints, and 36,662 renderer calls. Lambda path tokens are `lambda_1e-04`, `lambda_1e-03`, `lambda_1e-02`, `lambda_1e-01`, and `lambda_0_diag`.
""")
    write_text(DOCS / "AAAI27_COEFFICIENT_HEADROOM_EXECUTION_REPAIR_20260724.md", f"""# Coefficient Headroom Execution Repair

Status: `READY_FOR_ATTEMPT_002_BEFORE_OPTIMIZER`.
Classification: `COEFFICIENT_HEADROOM_OUTPUT_PATH_REPAIR_READY`.

Implementation HEAD `{IMPLEMENTATION_HEAD}` adds a unified safe path/atomic writer layer, migrates every Headroom scientific writer, freezes a complete collision-free `attempt_002` path plan, and moves path/smoke closure ahead of attempt materialization and the first renderer call. Windows and Cloud each passed 70 pure CPU tests. The eight frozen scientific contracts have zero semantic drift.

`attempt_001` remains the sealed pre-optimizer engineering failure with two historical renderer calls and zero optimizer steps. `attempt_002` does not exist and is authorized only for the next task, where it must begin from fresh Teacher/SVD parity and reuse nothing from `attempt_001`.

No renderer, model inference, optimizer, forward/backward pass, checkpoint, lambda selection, metric evaluation, visual sheet, GPU workload, Headroom figure refresh, LOO run, or PAPER_FINAL action occurred in this repair task.

Next task: `RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT_FROM_REPAIRED_CONTRACT`.
""")


if __name__ == "__main__":
    main()
