from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    CONDITIONS,
    KINDS,
    OUTFITS,
    _base_gradient_count,
    _base_named_tensors,
    _json_fingerprint,
    _load_samples,
    _loss,
    _optimizer,
    _regularization,
    _render,
    _sha256,
    _tensor_state_fingerprint,
    _transition_targets,
    _write_csv,
    _write_json,
    _write_text,
    build_oracle,
    load_contract,
)
from utils.full_training_checkpoint_utils import load_full_training_checkpoint  # noqa: E402


FILE_FIELDS = (
    "rgb", "foreground_mask", "clothing_mask", "target_edit_rgb", "target_base_rgb",
    "target_edit_mask", "target_edit_core_mask", "target_preserve_mask",
    "target_transition_mask", "target_protected_mask", "target_foreground_mask",
    "target_clothing_mask", "target_clothing_mask_raw", "target_base_foreground_mask",
    "target_old_clothing_mask", "target_revealed_skin_mask",
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _tensor_sha(name: str, value: torch.Tensor) -> str:
    return _tensor_state_fingerprint([(name, value)])


def _full_input_fingerprint(
    manifest_path: Path,
    config_path: Path,
    pipeline_path: Path,
    v5_output: Path,
    base: Any,
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    condition_map = {item["condition_id"]: item for item in manifest["conditions"]}
    samples = []
    for outfit in manifest["outfits"]:
        if outfit["outfit_id"] not in OUTFITS:
            continue
        for observation in outfit["observations"]:
            condition = observation["condition_id"]
            files = {}
            for field in FILE_FIELDS:
                path = _resolve(root, observation[field])
                files[field] = {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}
            geometry = condition_map[condition]
            samples.append({
                "outfit_id": outfit["outfit_id"],
                "condition_id": condition,
                "view": {"cond_000000": "front", "cond_000318": "back", "cond_000017": "left", "cond_000347": "right"}[condition],
                "files": files,
                "manifest_checksums": observation.get("checksums", {}),
                "pose_camera_source_checksums": geometry.get("source_checksum", {}),
                "pose_length": len(geometry["pose"]),
                "camera_shapes": {"K": [3, 3], "w2c": [4, 4]},
            })
    if len(samples) != 12:
        raise ValueError(f"expected 12 Module 4B samples, found {len(samples)}")
    graph = training.build_anchor_knn_edges(base.xyz_vt.to(base._xyz.device), 4)
    weights = base.nbr_gs_invdist.to(device=base._xyz.device, dtype=base._xyz.dtype)
    weights = weights / weights.sum(1, keepdim=True)
    checkpoint = Path(pipeline["base"]["model_dir"]) / pipeline["base"]["checkpoint_path"]
    lbs = Path(pipeline["base"]["lbs_grid_path"])
    v5_status = v5_output / "run_partial.json"
    value = {
        "schema_version": "canondressgs.module4b.input_fingerprint.v1",
        "supervision_mode": manifest["supervision_mode"],
        "sample_count": len(samples),
        "samples": samples,
        "fixture_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "module4b_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "v5_3_loss_config": {
            "path": str((PROJECT_ROOT / "configs/canon_dress_gs_dual_target_v5_3_boundary_alpha.yaml").resolve()),
            "sha256": _sha256(PROJECT_ROOT / "configs/canon_dress_gs_dual_target_v5_3_boundary_alpha.yaml"),
        },
        "pipeline_config": {"path": str(pipeline_path), "sha256": _sha256(pipeline_path)},
        "base_checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "lbs_grid": {"path": str(lbs), "sha256": _sha256(lbs)},
        "v5_3_sealed_status": {"path": str(v5_status), "sha256": _sha256(v5_status)},
        "anchor_graph": {"shape": list(graph.shape), "sha256": _tensor_sha("anchor_graph", graph)},
        "anchor_to_gaussian_mapping": {
            "indices_shape": list(base.nbr_gs.shape),
            "weights_shape": list(weights.shape),
            "indices_sha256": _tensor_sha("indices", base.nbr_gs),
            "normalized_weights_sha256": _tensor_sha("weights", weights),
            "row_sum_max_abs_error": float((weights.sum(1) - 1).abs().max()),
        },
    }
    value["aggregate_sha256"] = _json_fingerprint(value)
    return value


def _gradient_audit(
    root: Path,
    manifest: Path,
    config: dict[str, Any],
    pipeline: dict[str, Any],
    base: Any,
    device: torch.device,
) -> dict[str, Any]:
    background = torch.tensor(pipeline["render"]["background"], device=device, dtype=base._xyz.dtype)
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    results: dict[str, Any] = {}
    for outfit in OUTFITS:
        samples = _load_samples(manifest, outfit)
        transitions = _transition_targets(samples, device)
        results[outfit] = {}
        for kind in KINDS:
            run_dir = root / outfit / f"{kind}_oracle"
            cap = json.loads((run_dir / "transition_gradient_cap.json").read_text(encoding="utf-8"))
            coefficient = float(cap["final_frozen_coefficient"])
            oracle = build_oracle(kind, base, config, device).to(device)
            oracle.configure_stage(3)
            optimizer, group_names = _optimizer(oracle, config)
            load_full_training_checkpoint(
                run_dir / "checkpoints/step_000480.pth",
                model=oracle,
                optimizer=optimizer,
                optimizer_group_names=group_names,
                scheduler=None,
                scaler=None,
                expected_data_state={
                    "manifest": str(manifest.resolve()),
                    "manifest_sha256": _sha256(manifest),
                    "outfit_id": outfit,
                    "conditions": list(CONDITIONS),
                },
                expected_method_state={
                    "schema_version": "canondressgs.module4b.canonical_capacity.v1",
                    "oracle_kind": kind,
                    "config_sha256": _json_fingerprint(config),
                    "transition_coefficient": coefficient,
                    "shared_canonical_field": True,
                },
            )
            condition_rows = []
            for condition in CONDITIONS:
                oracle.zero_grad(set_to_none=True)
                output = oracle(base)
                rgb, alpha = _render(base, samples[condition], output.canonical_overrides, background)
                parts = _loss(rgb, alpha, samples[condition], transitions[condition], device, config, coefficient)
                regularizer, _ = _regularization(output, config)
                objective = parts["total"] + regularizer
                objective.backward()
                parameter_gradients = {}
                squared = torch.zeros((), device=device, dtype=torch.float64)
                nonzero_count = 0
                for name, parameter in oracle.named_parameters():
                    if parameter.grad is None:
                        parameter_gradients[name] = {"norm": 0.0, "max_abs": 0.0, "finite": True, "nonzero": False}
                        continue
                    gradient = parameter.grad.detach()
                    norm = float(torch.linalg.vector_norm(gradient.double()))
                    nonzero = bool(torch.count_nonzero(gradient).item())
                    parameter_gradients[name] = {
                        "norm": norm,
                        "max_abs": float(gradient.abs().max()),
                        "finite": bool(torch.isfinite(gradient).all()),
                        "nonzero": nonzero,
                    }
                    squared += gradient.double().square().sum()
                    nonzero_count += int(nonzero)
                condition_rows.append({
                    "condition_id": condition,
                    "objective": float(objective.detach()),
                    "gradient_norm": float(torch.sqrt(squared)),
                    "nonzero_gradient_parameter_count": nonzero_count,
                    "parameter_gradients": parameter_gradients,
                    "finite": bool(torch.isfinite(objective) and all(item["finite"] for item in parameter_gradients.values())),
                })
            parameter_norm = float(torch.sqrt(sum(parameter.detach().double().square().sum() for parameter in oracle.parameters())))
            result = {
                "evaluation_only": True,
                "optimizer_step_executed": False,
                "checkpoint": str(run_dir / "checkpoints/step_000480.pth"),
                "checkpoint_sha256": _sha256(run_dir / "checkpoints/step_000480.pth"),
                "parameter_norm": parameter_norm,
                "conditions": condition_rows,
                "all_finite": all(row["finite"] for row in condition_rows),
                "base_gradient_count": _base_gradient_count(base),
            }
            _write_json(run_dir / "final_gradient_audit.json", result)
            results[outfit][kind] = result
            del oracle, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    base_after = _tensor_state_fingerprint(_base_named_tensors(base))
    return {
        "runs": results,
        "base_fingerprint_before": base_before,
        "base_fingerprint_after": base_after,
        "base_bitwise_exact": base_before == base_after,
        "base_gradient_count": _base_gradient_count(base),
        "optimizer_steps_executed": 0,
    }


def _flatten_per_view(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step, item in sorted(value.items(), key=lambda pair: int(pair[0])):
        for row in item["conditions"]:
            rows.append({"step": int(step), **row})
    return rows


def _copy_required_artifacts(root: Path) -> None:
    (root / "contract").mkdir(exist_ok=True)
    shutil.copy2(root / "input_fingerprint_manifest.json", root / "input_fingerprint_manifest_preflight.json")
    shutil.copy2(root / "module4b_input_fingerprint.json", root / "contract/module4b_input_fingerprint.json")
    shutil.copy2(root / "method_contract.json", root / "module4b_oracle_contract.json")
    shutil.copy2(root / "module4b_oracle_contract.json", root / "contract/module4b_oracle_contract.json")
    shutil.copy2(root / "preflight/IMPLEMENTATION_AUDIT.md", root / "MODULE4B_ORACLE_IMPLEMENTATION_AUDIT.md")
    for outfit in OUTFITS:
        for kind in KINDS:
            run_dir = root / outfit / f"{kind}_oracle"
            resolved = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
            _write_json(run_dir / "config.json", resolved)
            shutil.copy2(run_dir / "run_manifest.json", run_dir / "manifest.json")
            shutil.copy2(run_dir / "metrics_history.csv", run_dir / "metrics.csv")
            per_view = json.loads((run_dir / "per_view_metrics.json").read_text(encoding="utf-8"))
            _write_csv(run_dir / "per_view_metrics.csv", _flatten_per_view(per_view))
            diagnostic = run_dir / "diagnostic_panels"
            diagnostic.mkdir(exist_ok=True)
            for source in sorted((run_dir / "visuals").glob("*.png")):
                shutil.copy2(source, diagnostic / source.name)
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            _write_json(run_dir / "FINAL_STATUS.json", status)
            _write_text(run_dir / "FINAL_STATUS.md", "\n".join([
                f"# {outfit} {kind.title()} Oracle Final Status",
                "",
                f"- status: `{status['final_status']}`",
                f"- fit: `{status['fit_classification']}`",
                f"- visual: `{status['visual_acceptance_status']}`",
                "- optimizer steps: `480`",
                "- base/backbone: frozen and exact",
            ]))
            _write_text(run_dir / "execution_summary.log", json.dumps(status, sort_keys=True))
    comparison = root / "comparison"
    comparisons = root / "comparisons"
    comparisons.mkdir(exist_ok=True)
    rows = []
    with (comparison / "capacity_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _write_csv(root / "module4b_gaussian_oracle_summary.csv", [row for row in rows if row["oracle_kind"] == "gaussian"])
    _write_csv(root / "module4b_anchor_oracle_summary.csv", [row for row in rows if row["oracle_kind"] == "anchor"])
    adjudication = json.loads((root / "final_adjudication.json").read_text(encoding="utf-8"))
    retention_rows = [{"outfit_id": outfit, **values} for outfit, values in adjudication["capacity_retention"].items()]
    _write_csv(root / "module4b_capacity_retention.csv", retention_rows)
    _write_json(root / "module4b_outfit_decision_matrix.json", adjudication)
    shutil.copy2(comparison / "outfit_capacity_contact_sheet.png", root / "module4b_gaussian_oracle_contact_sheet.png")
    shutil.copy2(comparison / "outfit_capacity_contact_sheet.png", root / "module4b_anchor_oracle_contact_sheet.png")
    shutil.copy2(comparison / "gaussian_vs_anchor_contact_sheet.png", root / "module4b_oracle_comparison_contact_sheet.png")
    for name in (
        "module4b_gaussian_oracle_summary.csv", "module4b_anchor_oracle_summary.csv",
        "module4b_capacity_retention.csv", "module4b_outfit_decision_matrix.json",
        "module4b_gaussian_oracle_contact_sheet.png", "module4b_anchor_oracle_contact_sheet.png",
        "module4b_oracle_comparison_contact_sheet.png",
    ):
        shutil.copy2(root / name, comparisons / name)
    shutil.copy2(root / "VISUAL_ACCEPTANCE.md", root / "MODULE4B_VISUAL_ACCEPTANCE.md")
    shutil.copy2(root / "FINAL_ADJUDICATION.md", root / "MODULE4B_FINAL_ADJUDICATION.md")
    shutil.copy2(root / "run_status.json", root / "MODULE4B_FINAL_STATUS.json")
    final_dir = root / "final_adjudication"
    final_dir.mkdir(exist_ok=True)
    for name in ("MODULE4B_VISUAL_ACCEPTANCE.md", "MODULE4B_FINAL_ADJUDICATION.md", "MODULE4B_FINAL_STATUS.json"):
        shutil.copy2(root / name, final_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal Module 4B evidence without optimizer updates")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/oracle/module4b_canonical_capacity_v1.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=PROJECT_ROOT / "configs/canon_dress_gs_mvp_real.yaml")
    parser.add_argument("--v5-3-output", type=Path, default=Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.output.resolve()
    final = json.loads((root / "final_adjudication.json").read_text(encoding="utf-8"))
    if final.get("final_status") not in {"PASS", "PARTIAL", "FAIL"}:
        raise RuntimeError("Module 4B final adjudication must exist before sealing")
    config = load_contract(args.config)
    pipeline = training.load_config(args.pipeline_config)
    device = torch.device(args.device)
    base = training.load_frozen_mmlphuman_base(
        pipeline["base"]["model_dir"], pipeline["base"]["checkpoint_path"], device,
    )
    fingerprint = _full_input_fingerprint(
        args.manifest.resolve(), args.config.resolve(), args.pipeline_config.resolve(),
        args.v5_3_output.resolve(), base, pipeline,
    )
    _write_json(root / "module4b_input_fingerprint.json", fingerprint)
    preflight = json.loads((root / "input_fingerprint_manifest.json").read_text(encoding="utf-8"))
    integrity = {
        "postrun_full_fingerprint": fingerprint["aggregate_sha256"],
        "top_level_checks": {},
    }
    mapping = {
        "config": "module4b_config", "pipeline_config": "pipeline_config",
        "fixture_manifest": "fixture_manifest", "base_checkpoint": "base_checkpoint",
        "lbs_grid": "lbs_grid",
    }
    for old, new in mapping.items():
        integrity["top_level_checks"][old] = preflight[old]["sha256"] == fingerprint[new]["sha256"]
    integrity["all_inputs_unchanged"] = all(integrity["top_level_checks"].values())
    if not integrity["all_inputs_unchanged"]:
        raise RuntimeError(f"post-run input integrity failed: {integrity}")
    _write_json(root / "input_integrity_postrun.json", integrity)
    gradient = _gradient_audit(root, args.manifest.resolve(), config, pipeline, base, device)
    if not gradient["base_bitwise_exact"] or gradient["base_gradient_count"] != 0:
        raise RuntimeError("evaluation-only gradient audit mutated the frozen base")
    _write_json(root / "final_gradient_audit.json", gradient)
    _copy_required_artifacts(root)
    _write_json(root / "seal_status.json", {
        "status": "COMPLETE",
        "optimizer_steps_executed": 0,
        "inputs_unchanged": True,
        "base_exact_during_gradient_audit": True,
        "required_aliases_written": True,
    })


if __name__ == "__main__":
    main()
