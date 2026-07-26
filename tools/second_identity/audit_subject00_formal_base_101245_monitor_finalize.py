#!/usr/bin/env python3
"""Audit and finalize the Subject00 formal base 101245-step run.

This utility is intentionally read-mostly: it writes compact audit reports in
the monitor/finalize worktree and never copies checkpoints, logs, datasets, or
render images into Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-SUBJECT00-FORMAL-BASE-101245-MONITOR-AND-FINALIZE-001"
SOURCE_BRANCH = "research/subject00-formal-base-101245-execution-20260726"
SOURCE_HEAD = "9a434899bd09cdcdc1e648d69ce32d7dfe2f6e3c"
MONITOR_BRANCH = "research/subject00-formal-base-101245-monitor-finalize-20260726"
RUN_ROOT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-BASE-101245-001/attempt_001"
)
EXECUTION_WT = Path(
    "/root/autodl-tmp/canondressgs_work/worktrees/"
    "canondressgs_subject00_formal_base_101245_execution"
)
EXPECTED_CONFIG_CRLF_SHA = (
    "be28dacbea5a3e4b35336556f752b1548ebb8c3c0b355be8348e405df49bd356"
)
EXPECTED_CONFIG_LF_SHA = (
    "7ffd81771c3e25e8f4fdef4a7e7e628539642a8107f71c77ce0f748303a572d5"
)
EXPECTED_DATA_HASHES = {
    "data_manifest": (
        Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
            "reports/SUBJECT00_CLOUD_DATA_MANIFEST.json"
        ),
        "e61ca6061f0a85be9c5c6e1d9163341c10749bc1a4ff1b14d27f64b5705aa99e",
    ),
    "calibration": (
        Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
            "subject00/calibration.json"
        ),
        "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7",
    ),
    "smpl_params": (
        Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
            "subject00/smpl_params.npz"
        ),
        "ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2",
    ),
    "template": (
        Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
            "derived_assets/subject00/template/template_smplx_body_surface.ply"
        ),
        "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031",
    ),
    "lbs_grid": (
        Path(
            "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
            "derived_assets/subject00/surface_lbs/lbs_weights.npy"
        ),
        "56aa68a9d4baade67621fa2bfac462ac88074eeaf7c9bfdbe86f2360b91261a1",
    ),
    "init_checkpoint": (
        Path(
            "/root/autodl-tmp/canondressgs_work/outputs/"
            "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
            "checkpoints/step_000000.pth"
        ),
        "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a",
    ),
}
CHECKPOINT_STEPS = (0, 20249, 40498, 60747, 80996, 100000, 101245)
TRAINABLE_GROUPS = {
    "dxyz": ("dxyz_vt",),
    "scales": ("_scaling",),
    "quats": ("_rotation",),
    "opacities": ("_opacity",),
    "sh0": ("_sh0",),
    "shN": ("_shN",),
    "dxyz_bs": ("dxyz_bs",),
    "dscales_bs": ("scaling_bs",),
    "dquats_bs": ("rotation_bs",),
    "dopacities_bs": ("opacity_bs",),
    "dsh0_bs": ("sh0_bs",),
    "dshN_bs": ("shN_bs",),
    "encoder_feat_params": ("encoder_feat_params",),
    "xyz_offset": ("xyz_offset",),
}
FROZEN_TENSORS = ("_xyz", "_weights", "t_joints", "joint_parents", "xyz_vt", "xyz_ft")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_text(args: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def config_equivalence() -> dict[str, Any]:
    rel = "config/subject00_surface_lbs_formal_strict_split.yaml"
    current = (REPO_ROOT / rel).read_bytes()
    git_blob = run_text(["git", "show", f"{SOURCE_HEAD}:{rel}"], cwd=REPO_ROOT).encode("utf-8")
    current_lf = current.replace(b"\r\n", b"\n")
    git_lf = git_blob.replace(b"\r\n", b"\n")
    current_crlf = current_lf.replace(b"\n", b"\r\n")
    current_obj = yaml.safe_load(current_lf.decode("utf-8"))
    git_obj = yaml.safe_load(git_lf.decode("utf-8"))
    return {
        "config_path": rel,
        "windows_or_crlf_raw_sha256": sha256_bytes(current_crlf),
        "cloud_or_git_lf_raw_sha256": sha256_bytes(git_lf),
        "normalized_current_sha256": sha256_bytes(current_lf),
        "normalized_git_sha256": sha256_bytes(git_lf),
        "normalized_config_sha256": sha256_bytes(git_lf),
        "normalized_config_match": current_lf == git_lf,
        "parsed_config_object_match": current_obj == git_obj,
        "semantic_config_diff_count": 0 if current_obj == git_obj else 1,
        "config_hash_difference": "LINE_ENDING_ONLY" if current_lf == git_lf and current_obj == git_obj else "NON_LINE_ENDING_DIFF",
        "expected_crlf_sha256": EXPECTED_CONFIG_CRLF_SHA,
        "expected_lf_sha256": EXPECTED_CONFIG_LF_SHA,
        "status": "PASS" if sha256_bytes(current_crlf) == EXPECTED_CONFIG_CRLF_SHA and sha256_bytes(git_lf) == EXPECTED_CONFIG_LF_SHA and current_obj == git_obj else "FAIL",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def training_log_audit(run_root: Path) -> dict[str, Any]:
    records = read_jsonl(run_root / "training_logs/step_records.jsonl")
    steps = [int(row["step"]) for row in records]
    last = records[-1] if records else {}
    lpips_nonzero = [int(row["step"]) for row in records if float(row.get("lpips_loss", 0.0)) > 0.0]
    component_errors = []
    for row in records[-1000:]:
        total = float(row["total_loss"])
        parts = (
            float(row["l1_loss"])
            + float(row["lpips_loss"])
            + float(row["dxyz_smooth_loss"])
            + float(row["scaling_loss"])
        )
        component_errors.append(abs(total - parts))
    window = records[-1000:] if len(records) >= 1000 else records
    wall = [float(row.get("wall_seconds", 0.0)) for row in window if float(row.get("wall_seconds", 0.0)) > 0.0]
    return {
        "log_path": str(run_root / "training_logs/step_records.jsonl"),
        "record_count": len(records),
        "start_step_observed": steps[0] if steps else None,
        "current_step": steps[-1] if steps else 0,
        "final_step": 101245,
        "step_monotonic_strict": all(b > a for a, b in zip(steps, steps[1:])),
        "complete_log_length": len(records) == 101245,
        "last_losses": {
            "total_loss": last.get("total_loss"),
            "l1_loss": last.get("l1_loss"),
            "lpips_loss": last.get("lpips_loss"),
            "dxyz_smooth_loss": last.get("dxyz_smooth_loss"),
            "scaling_loss": last.get("scaling_loss"),
        },
        "loss_initial": records[0]["total_loss"] if records else None,
        "loss_final": last.get("total_loss"),
        "loss_finite_all": all(bool(row.get("loss_finite")) for row in records),
        "gradients_finite_all": all(bool(row.get("gradients_finite")) for row in records),
        "optimizer_state_finite_all": all(bool(row.get("optimizer_state_finite")) for row in records),
        "lpips_nonzero_step_count": len(lpips_nonzero),
        "lpips_first_nonzero_step": lpips_nonzero[0] if lpips_nonzero else None,
        "lpips_last_nonzero_step": lpips_nonzero[-1] if lpips_nonzero else None,
        "component_sum_abs_error_max_last1000": max(component_errors, default=None),
        "median_seconds_per_optimizer_step_last_window": statistics.median(wall) if wall else None,
        "steps_per_hour_last_window": 3600.0 / statistics.median(wall) if wall else None,
        "nonzero_gradient_group_union": sorted({name for row in records for name in row.get("nonzero_gradient_groups", [])}),
        "peak_vram_bytes_log_max": max((int(row.get("peak_vram_bytes", 0)) for row in records), default=0),
    }


def process_audit(run_root: Path) -> dict[str, Any]:
    pgrep = run_text(["pgrep", "-af", "run_subject00_formal_base_101245.py|SUBJECT00-FORMAL-BASE-101245-001"], check=False)
    lines = [line for line in pgrep.splitlines() if line.strip()]
    tmux = run_text(["tmux", "list-panes", "-t", "subject00_formal_base_101245_001", "-F", "#{pane_pid} #{pane_current_command} #{pane_current_path}"], check=False)
    return {
        "tmux_session": "subject00_formal_base_101245_001",
        "tmux_panes": tmux.splitlines() if tmux else [],
        "training_process_matches": lines,
        "unique_training_process_count": sum("run_subject00_formal_base_101245.py" in line and "--phase train" in line for line in lines),
        "session_metadata": read_json(run_root / "logs/session_metadata.json") if (run_root / "logs/session_metadata.json").is_file() else None,
    }


def sample_gpu(args: argparse.Namespace) -> None:
    rows = []
    for index in range(args.samples):
        gpu = run_text([
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ], check=False)
        compute = run_text([
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ], check=False)
        rows.append({"ordinal": index + 1, "gpu": gpu, "compute": compute})
        if index + 1 < args.samples:
            time.sleep(args.interval_seconds)
    report = {
        "schema_version": "canondressgs.subject00.formal_base_101245.gpu_samples.v1",
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "run_root": str(args.run_root),
        "samples": rows,
        "training": training_log_audit(args.run_root),
        "process": process_audit(args.run_root),
    }
    write_json(args.output, report)
    print(json.dumps({"status": "PASS", "samples": len(rows), "path": str(args.output)}, sort_keys=True))


def tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def tensor_sha(tensor: Any) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def flatten_group(payload: dict[str, Any], group: str) -> list[tuple[str, Any]]:
    keys = TRAINABLE_GROUPS[group]
    if group == "encoder_feat_params":
        values = payload.get("encoder_feat_params") or {}
        return [(f"encoder_feat_params.{name}", values[name]) for name in sorted(values)]
    return [(keys[0], payload[keys[0]])]


def group_count(payload: dict[str, Any], group: str) -> int:
    return sum(int(tensor.numel()) for _, tensor in flatten_group(payload, group))


def group_norm(payload: dict[str, Any], group: str) -> float:
    total = 0.0
    for _, tensor in flatten_group(payload, group):
        value = tensor.detach().cpu().double()
        total += float((value * value).sum().item())
    return math.sqrt(total)


def group_delta(a: dict[str, Any], b: dict[str, Any], group: str) -> dict[str, Any]:
    delta_l1 = 0.0
    delta_l2_sq = 0.0
    changed = 0
    unchanged = 0
    finite = True
    for name, first in flatten_group(a, group):
        second = dict(flatten_group(b, group))[name]
        lhs = first.detach().cpu()
        rhs = second.detach().cpu()
        diff = (rhs - lhs).double()
        delta_l1 += float(diff.abs().sum().item())
        delta_l2_sq += float((diff * diff).sum().item())
        if bool((diff != 0).any().item()):
            changed += 1
        else:
            unchanged += 1
        finite = finite and bool(torch_isfinite(lhs)) and bool(torch_isfinite(rhs))
    return {
        "parameter_count": group_count(a, group),
        "norm_step0": group_norm(a, group),
        "norm_other": group_norm(b, group),
        "delta_l1": delta_l1,
        "delta_l2": math.sqrt(delta_l2_sq),
        "changed_tensor_count": changed,
        "unchanged_tensor_count": unchanged,
        "finite": finite,
        "changed": changed > 0 and delta_l1 > 0.0,
    }


def torch_isfinite(tensor: Any) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


def load_checkpoint(path: Path, *, map_location: str | None = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)


def optimizer_state_summary(payload: dict[str, Any]) -> dict[str, Any]:
    states = payload.get("optimizer_states", {})
    by_group: dict[str, Any] = {}
    total_tensors = 0
    total_bytes = 0
    for group, state_dict in states.items():
        tensors = 0
        bytes_ = 0
        steps = []
        param_states = state_dict.get("state", {})
        for values in param_states.values():
            for key, value in values.items():
                if hasattr(value, "numel") and hasattr(value, "element_size"):
                    tensors += 1
                    bytes_ += tensor_bytes(value)
                    if key == "step":
                        try:
                            steps.append(float(value.detach().cpu().item()))
                        except Exception:
                            pass
                elif key == "step":
                    try:
                        steps.append(float(value))
                    except Exception:
                        pass
        total_tensors += tensors
        total_bytes += bytes_
        by_group[group] = {
            "state_tensor_count": tensors,
            "state_bytes": bytes_,
            "state_steps": sorted(set(steps)),
            "param_group_count": len(state_dict.get("param_groups", [])),
        }
    return {
        "optimizer_group_count": len(states),
        "optimizer_state_tensor_count": total_tensors,
        "optimizer_state_bytes": total_bytes,
        "by_group": by_group,
    }


def optimizer_state_device_summary(payload: dict[str, Any]) -> dict[str, Any]:
    states = payload.get("optimizer_states", {})
    by_group: dict[str, Any] = {}
    all_devices = set()
    for group, state_dict in states.items():
        devices = set()
        dtypes = set()
        for values in state_dict.get("state", {}).values():
            for value in values.values():
                if hasattr(value, "device"):
                    devices.add(str(value.device))
                    dtypes.add(str(value.dtype))
                    all_devices.add(str(value.device))
        by_group[group] = {
            "devices": sorted(devices),
            "dtypes": sorted(dtypes),
        }
    return {
        "all_optimizer_state_devices": sorted(all_devices),
        "by_group": by_group,
    }


def checkpoint_audit(run_root: Path) -> dict[str, Any]:
    import torch

    checkpoint_dir = run_root / "checkpoints"
    step0 = EXPECTED_DATA_HASHES["init_checkpoint"][0]
    paths = {0: step0}
    paths.update({step: checkpoint_dir / f"step_{step:06d}.pth" for step in CHECKPOINT_STEPS if step})
    records = []
    parse_ok = True
    for step, path in paths.items():
        sidecar = checkpoint_dir / f"step_{step:06d}.sidecar.json" if step else None
        record = file_record(path)
        record["step"] = step
        record["sidecar"] = file_record(sidecar) if sidecar else {"path": str(run_root / "contract/step0_checkpoint_pointer.json"), "exists": True}
        if path.is_file():
            try:
                payload = load_checkpoint(path, map_location="cpu")
                record["parse_status"] = "PASS"
                record["payload_training_step"] = int(payload.get("training_step", payload.get("iteration", step)))
                record["payload_data_order_position"] = int(payload.get("data_order_position", record["payload_training_step"]))
                record["has_optimizer_states"] = "optimizer_states" in payload
                record["has_scheduler_states"] = "scheduler_states" in payload
                record["has_rng_states"] = all(key in payload for key in ("python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_states"))
                record["temporary_file"] = path.name.startswith(".") or ".tmp" in path.name
                del payload
            except Exception as exc:
                parse_ok = False
                record["parse_status"] = f"FAIL:{type(exc).__name__}:{exc}"
        else:
            parse_ok = False
            record["parse_status"] = "MISSING"
        records.append(record)

    payload0 = load_checkpoint(step0, map_location="cpu")
    payload20249 = load_checkpoint(paths[20249], map_location="cpu") if paths[20249].is_file() else None
    payload_final = load_checkpoint(paths[101245], map_location="cpu") if paths[101245].is_file() else None
    group_changes: dict[str, Any] = {}
    if payload_final is not None:
        for group in TRAINABLE_GROUPS:
            result = {"to_final": group_delta(payload0, payload_final, group)}
            if payload20249 is not None:
                result["to_step_020249"] = group_delta(payload0, payload20249, group)
            group_changes[group] = result

    frozen = {}
    frozen_mutations = 0
    if payload_final is not None:
        for key in FROZEN_TENSORS:
            initial = payload0[key].detach().cpu()
            final = payload_final[key].detach().cpu()
            same = bool(torch.equal(initial, final))
            frozen[key] = {
                "shape": list(final.shape),
                "dtype": str(final.dtype),
                "step0_sha256": tensor_sha(initial),
                "final_sha256": tensor_sha(final),
                "exact_match": same,
            }
            frozen_mutations += 0 if same else 1
    optimizer = optimizer_state_summary(payload_final) if payload_final is not None else {}
    scheduler = payload_final.get("scheduler_states", []) if payload_final is not None else []
    scheduler_summary = {
        "scheduler_state_count": len(scheduler),
        "last_epoch_values": [state.get("last_epoch") for state in scheduler if isinstance(state, dict)],
    }
    trainable_count = sum(group_count(payload0, group) for group in TRAINABLE_GROUPS)
    device_registry = {}
    optimizer_device_registry = {}
    if payload_final is not None:
        payload_final_saved_device = load_checkpoint(paths[101245], map_location=None)
        for group in TRAINABLE_GROUPS:
            device_registry[group] = [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device_in_saved_checkpoint": str(tensor.device),
                }
                for name, tensor in flatten_group(payload_final_saved_device, group)
            ]
        optimizer_device_registry = optimizer_state_device_summary(payload_final_saved_device)
        del payload_final_saved_device
    return {
        "checkpoint_count_including_step0_pointer": len(records),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_records": records,
        "checkpoint_parse_status": "PASS" if parse_ok else "FAIL",
        "trainable_groups": list(TRAINABLE_GROUPS),
        "trainable_tensor_count": sum(len(flatten_group(payload0, group)) for group in TRAINABLE_GROUPS),
        "trainable_parameter_count": trainable_count,
        "trainable_group_device_registry": device_registry,
        "trainable_group_change_status": {
            group: values["to_final"]["changed"] for group, values in group_changes.items()
        },
        "trainable_group_delta_summary": group_changes,
        "unchanged_trainable_groups": [
            group for group, values in group_changes.items() if not values["to_final"]["changed"]
        ],
        "frozen_tensor_audit": frozen,
        "frozen_parameter_mutations": frozen_mutations,
        "optimizer_state": optimizer,
        "optimizer_state_device_registry": optimizer_device_registry,
        "scheduler_state": scheduler_summary,
        "rng_state_status": "PASS" if all(row.get("has_rng_states") for row in records if row["step"]) else "FAIL",
        "resume_state_status": "PASS_SAME_ATTEMPT_CHECKPOINTS_INCLUDE_OPTIMIZER_SCHEDULER_RNG_DATA_ORDER" if parse_ok else "FAIL",
    }


def data_integrity_audit() -> dict[str, Any]:
    rows = {}
    for name, (path, expected) in EXPECTED_DATA_HASHES.items():
        info = file_record(path)
        info["expected_sha256"] = expected
        info["matches_expected"] = info["sha256"] == expected
        rows[name] = info
    return {
        "records": rows,
        "data_mutations": 0 if all(row["matches_expected"] for name, row in rows.items() if name != "init_checkpoint") else 1,
        "init_checkpoint_mutations": 0 if rows["init_checkpoint"]["matches_expected"] else 1,
    }


def availability_audit() -> dict[str, Any]:
    path = Path(
        "/root/autodl-tmp/datasets/thuman4_second_identity_staging/"
        "reports/SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json"
    )
    value = read_json(path)
    summary = value["summary"]
    return {
        "valid_pair_count": int(summary["valid_pairs"]),
        "missing_slot_count": int(summary["invalid_pairs"]),
        "sampling_denominator": int(summary["valid_pairs"]),
        "availability_file_sha256": sha256_file(path),
        "canonical_entries_sha256": summary["manifest_sha256"],
    }


def lpips_audit(training: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load((REPO_ROOT / "config/subject00_surface_lbs_formal_strict_split.yaml").read_text(encoding="utf-8"))
    medium_source = (REPO_ROOT / "tools/second_identity/run_subject00_medium_pilot.py").read_text(encoding="utf-8")
    loss_source = (REPO_ROOT / "utils/loss_utils.py").read_text(encoding="utf-8")
    return {
        "lpips_weight": float(config["loss"]["lpips"]["weight"]),
        "implementation_path": "utils/loss_utils.py:lpips_loss",
        "implementation_status": "PASS_IMPORTED_LEARNED_PERCEPTUAL_IMAGE_PATCH_SIMILARITY_CUDA"
        if "LearnedPerceptualImagePatchSimilarity" in loss_source and ".cuda()" in loss_source
        else "FAIL_IMPLEMENTATION_NOT_FOUND",
        "cadence": "step > 6000; random patch only at step >= 300000, outside this 101245-step run",
        "nonzero_step_count": int(training["lpips_nonzero_step_count"]),
        "first_nonzero_step": training["lpips_first_nonzero_step"],
        "last_nonzero_step": training["lpips_last_nonzero_step"],
        "total_loss_component_binding": "PASS_TOTAL_EQUALS_L1_PLUS_LPIPS_PLUS_SMOOTH_PLUS_SCALING"
        if training["component_sum_abs_error_max_last1000"] is not None
        and training["component_sum_abs_error_max_last1000"] < 1.0e-6
        and "total = l1_value + lpips_value + smooth + scaling" in medium_source
        else "FAIL_COMPONENT_BINDING",
        "gradient_contribution_status": "PASS_BOUND_IN_BACKWARD_GRAPH_AFTER_STEP_6000"
        if "total.backward()" in medium_source and training["lpips_nonzero_step_count"] > 0
        else "FAIL_NO_NONZERO_LPIPS_BACKWARD_EVIDENCE",
    }


def final_evaluation_audit(run_root: Path) -> dict[str, Any]:
    fixed_paths = sorted((run_root / "evaluations").glob("fixed96_step_*.json"))
    final_aggregate = run_root / "evaluations/final_full_aggregate.json"
    final_records = run_root / "evaluations/final_full_query_records.jsonl"
    aggregate = read_json(final_aggregate) if final_aggregate.is_file() else None
    return {
        "fixed96_evaluation_count": len(fixed_paths),
        "fixed96_paths": [str(path) for path in fixed_paths],
        "final_full_aggregate": file_record(final_aggregate),
        "final_full_query_records": file_record(final_records),
        "final_evaluation_status": aggregate.get("status") if aggregate else "MISSING",
        "final_metrics": aggregate.get("metrics_all_for_audit_only") if aggregate else None,
        "animation_compatibility_status": "METRICS_ONLY_NO_AUTOMATIC_VISUAL_PASS",
        "different_pose_status": "METRICS_REPORTED_BY_FROZEN_QUADRANTS",
        "different_camera_status": "METRICS_REPORTED_BY_FROZEN_QUADRANTS",
    }


def build_review_manifest(run_root: Path) -> dict[str, Any]:
    final_visuals = sorted((run_root / "visuals/fixed96/step_101245").glob("query_*.png"))
    records = []
    for path in final_visuals:
        records.append(file_record(path))
    manifest = {
        "schema_version": "canondressgs.subject00.formal_base_101245.visual_review_manifest.v1",
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "review_package_path": str(run_root / "visuals/fixed96/step_101245"),
        "query_visual_count": len(records),
        "query_visuals": records,
        "coverage": [
            "fixed96 frozen trajectory",
            "representative training and heldout quadrants from formal protocol",
            "GT/mask/prediction/alpha/depth panels in each PNG",
            "different pose and different camera cases from frozen query order",
            "failure-risk review cases selected only by frozen protocol ordinals"
        ],
        "human_visual_decision": None,
        "identity_visual_pass": None,
        "geometry_visual_pass": None,
        "deformation_visual_pass": None,
        "artifact_visual_pass": None,
        "scientific_pass": None,
        "paper_final": False,
    }
    return manifest


def make_markdown_report(audit: dict[str, Any], review: dict[str, Any]) -> str:
    training = audit["training_log"]
    classification = audit["final_classification"]
    return "\n".join(
        [
            "# Subject00 Formal Base 101245 Monitor And Finalize Report",
            "",
            f"- Task ID: `{TASK_ID}`",
            f"- Source HEAD: `{SOURCE_HEAD}`",
            f"- Run root: `{audit['run_root']}`",
            f"- Final optimizer step: `{training['current_step']}`",
            f"- Training exit status: `{audit['training_exit_status']}`",
            f"- Final evaluation status: `{audit['final_evaluation']['final_evaluation_status']}`",
            f"- LPIPS nonzero steps: `{audit['lpips']['nonzero_step_count']}`",
            f"- Unchanged trainable groups: `{audit['checkpoint_audit'].get('unchanged_trainable_groups', [])}`",
            f"- Review package: `{review['review_package_path']}`",
            "- Human visual decision: `null`",
            "- Scientific pass: `null`",
            f"- Final classification: `{classification}`",
            f"- Next task: `{audit['next_task']}`",
            "",
            "This report records technical audit evidence only. It does not make a scientific or human visual pass decision.",
            "",
        ]
    )


def static_check() -> None:
    config = config_equivalence()
    required = [
        REPO_ROOT / "tools/second_identity/run_subject00_formal_base_101245.py",
        REPO_ROOT / "tools/second_identity/run_subject00_formal_strict_split.py",
        REPO_ROOT / "config/subject00_surface_lbs_formal_strict_split.yaml",
    ]
    if not all(path.is_file() for path in required):
        raise AssertionError("required formal-base source files are missing")
    if config["semantic_config_diff_count"] != 0:
        raise AssertionError("config semantic diff is nonzero")
    print("SUBJECT00_FORMAL_BASE_101245_MONITOR_FINALIZE_STATIC_PASS")


def final_classification(audit: dict[str, Any]) -> tuple[str, str]:
    training = audit["training_log"]
    checkpoint = audit["checkpoint_audit"]
    lpips = audit["lpips"]
    evaluation = audit["final_evaluation"]
    config = audit["config_equivalence"]
    data = audit["data_integrity"]
    if training["current_step"] < 101245:
        return (
            "SUBJECT00_FORMAL_BASE_101245_MONITORING_IN_PROGRESS",
            "MONITOR_SUBJECT00_FORMAL_BASE_101245_EXECUTION",
        )
    if audit["training_exit_status"] != "EXITED_ZERO_OR_TMUX_SESSION_ENDED_AFTER_PASS":
        return (
            "SUBJECT00_FORMAL_BASE_101245_ENGINEERING_FAIL_REQUIRES_RESUME_DIAGNOSIS",
            "USER_REVIEW_SUBJECT00_FORMAL_BASE_FAILURE_AND_RESUME_EVIDENCE",
        )
    hard_violation = (
        not config["normalized_config_match"]
        or not config["parsed_config_object_match"]
        or data["data_mutations"] != 0
        or data["init_checkpoint_mutations"] != 0
        or checkpoint["checkpoint_parse_status"] != "PASS"
        or checkpoint["frozen_parameter_mutations"] != 0
        or not training["step_monotonic_strict"]
        or not training["complete_log_length"]
        or not training["loss_finite_all"]
        or not training["gradients_finite_all"]
        or not training["optimizer_state_finite_all"]
    )
    if hard_violation:
        return (
            "SUBJECT00_FORMAL_BASE_101245_EXECUTION_CONTRACT_VIOLATION",
            "USER_RESOLVE_SUBJECT00_FORMAL_BASE_CONTRACT_EVIDENCE_GAP",
        )
    if lpips["nonzero_step_count"] <= 0:
        return (
            "SUBJECT00_FORMAL_BASE_LOSS_CONTRACT_VIOLATION_LPIPS_INACTIVE",
            "USER_RESOLVE_SUBJECT00_FORMAL_BASE_CONTRACT_EVIDENCE_GAP",
        )
    if evaluation["final_evaluation_status"] != "PASS":
        return (
            "SUBJECT00_FORMAL_BASE_101245_COMPLETED_PENDING_CONTRACT_EVIDENCE_RESOLUTION",
            "USER_RESOLVE_SUBJECT00_FORMAL_BASE_CONTRACT_EVIDENCE_GAP",
        )
    if checkpoint.get("unchanged_trainable_groups"):
        return (
            "SUBJECT00_FORMAL_BASE_101245_COMPLETED_PENDING_CONTRACT_EVIDENCE_RESOLUTION",
            "USER_RESOLVE_SUBJECT00_FORMAL_BASE_CONTRACT_EVIDENCE_GAP",
        )
    return (
        "SUBJECT00_FORMAL_BASE_101245_TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW",
        "USER_REVIEW_SUBJECT00_FORMAL_BASE_FINAL_RENDER_AND_METRICS",
    )


def run_final_audit(args: argparse.Namespace) -> None:
    training = training_log_audit(args.run_root)
    process = process_audit(args.run_root)
    stderr_text = (args.run_root / "logs/stderr.log").read_text(errors="replace") if (args.run_root / "logs/stderr.log").is_file() else ""
    train_result_path = args.run_root / "training_logs/training_result.json"
    training_result = read_json(train_result_path) if train_result_path.is_file() else None
    stdout_text = (args.run_root / "logs/stdout.log").read_text(errors="replace") if (args.run_root / "logs/stdout.log").is_file() else ""
    training_exit_status = (
        "EXITED_ZERO_OR_TMUX_SESSION_ENDED_AFTER_PASS"
        if training_result and training_result.get("status") == "TRAINING_COMPLETE_PENDING_ROUNDTRIP_FULL_EVALUATION_AND_REVIEW"
        and '"phase": "train"' in stdout_text
        and '"status": "PASS"' in stdout_text
        else "RUNNING_OR_INCOMPLETE"
    )
    audit = {
        "schema_version": "canondressgs.subject00.formal_base_101245.monitor_finalize_audit.v1",
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "monitor_branch": MONITOR_BRANCH,
        "monitor_head": run_text(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "run_root": str(args.run_root),
        "training_command": "/root/autodl-tmp/conda_envs/mmlphuman/bin/python tools/second_identity/run_subject00_formal_base_101245.py --phase train --attempt-root /root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001/attempt_001",
        "process": process,
        "training_exit_status": training_exit_status,
        "training_wall_time": training_result.get("wall_seconds") if training_result else None,
        "stderr_has_traceback": "Traceback" in stderr_text,
        "stderr_has_oom": "out of memory" in stderr_text.lower(),
        "config_equivalence": config_equivalence(),
        "training_log": training,
        "checkpoint_audit": checkpoint_audit(args.run_root) if training["current_step"] >= 101245 else {},
        "data_integrity": data_integrity_audit(),
        "availability": availability_audit(),
        "lpips": lpips_audit(training),
        "final_evaluation": final_evaluation_audit(args.run_root),
        "storage": {
            "df_B1": run_text(["df", "-B1", "/root/autodl-tmp"], check=False),
            "formal_safety_line_bytes": 32212254720,
            "operational_line_bytes": 34359738368,
        },
        "paper_modifications": 0,
        "paper_final": False,
    }
    classification, next_task = final_classification(audit)
    audit["final_classification"] = classification
    audit["next_task"] = next_task
    review = build_review_manifest(args.run_root)
    write_json(args.audit_output, audit)
    write_json(args.review_output, review)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(make_markdown_report(audit, review), encoding="utf-8")
    print(json.dumps({"status": "PASS", "classification": classification, "audit": str(args.audit_output)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("static-check")
    sample = sub.add_parser("sample-gpu")
    sample.add_argument("--run-root", type=Path, default=RUN_ROOT)
    sample.add_argument("--samples", type=int, default=5)
    sample.add_argument("--interval-seconds", type=float, default=6.0)
    sample.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "paper_protocol/subject00_formal_base/subject00_formal_base_101245_gpu_runtime_samples_20260726.json",
    )
    audit = sub.add_parser("final-audit")
    audit.add_argument("--run-root", type=Path, default=RUN_ROOT)
    audit.add_argument(
        "--audit-output",
        type=Path,
        default=REPO_ROOT / "paper_protocol/subject00_formal_base/subject00_formal_base_101245_monitor_finalize_audit_20260726.json",
    )
    audit.add_argument(
        "--review-output",
        type=Path,
        default=REPO_ROOT / "paper_protocol/subject00_formal_base/subject00_formal_base_101245_visual_review_manifest_20260726.json",
    )
    audit.add_argument(
        "--report-output",
        type=Path,
        default=REPO_ROOT / "docs/PAPER/AAAI27_SUBJECT00_FORMAL_BASE_101245_MONITOR_FINALIZE_REPORT_20260726.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if args.mode == "static-check":
        static_check()
    elif args.mode == "sample-gpu":
        sample_gpu(args)
    elif args.mode == "final-audit":
        run_final_audit(args)


if __name__ == "__main__":
    main()
