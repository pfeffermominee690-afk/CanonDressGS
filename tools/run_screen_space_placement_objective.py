from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import CanonicalGaussianOverrides  # noqa: E402
from scene.screen_space_placement_objective import (  # noqa: E402
    OBJECTIVE_NAME,
    SUPPORT_RENDER_NAME,
    build_editable_gaussian_pool,
    build_placement_masks,
    fixed_attribute_support_render,
    qualify_support_proxy,
    validate_pool_source_names,
)
from tools.run_alpha_raster_audit import (  # noqa: E402
    _checkpoint_path,
    load_runtime,
    load_state_model,
    render_explicit,
    sha256,
    target_free_state,
    trusted_regions,
)
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _environment,
    _git_state,
    _load_samples,
    _tensor_state_fingerprint,
)
from tools.run_new_silhouette_semantics import attach_artifact_masks  # noqa: E402
from utils.dressable_camera_utils import build_mmlphuman_camera  # noqa: E402
from utils.mmlphuman_state_utils import mmlphuman_state_transaction  # noqa: E402


SCHEMA = "canondressgs.screen_space_placement_objective.v1"
STATES = ("P1", "P2", "P3")
FOCUS = ("cond_000318", "cond_000347")
OUTPUT_DIRS = (
    "contract", "input_audit", "editable_gaussian_pool",
    "support_render_validation", "placement_loss", "gradient_direction_audit",
    "gradient_calibration", "G0_historical", "G1_placement_objective",
    "G2_placement_warmup_if_eligible", "comparisons", "instrumentation",
    "visual_acceptance", "final_adjudication",
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/research/subject02_screen_space_placement_objective_v1.yaml"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-SCREEN-SPACE-PLACEMENT-001/attempt_001"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def remote_ref(branch: str) -> str:
    output = git("ls-remote", "origin", f"refs/heads/{branch}")
    if not output:
        raise RuntimeError(f"missing frozen remote branch: {branch}")
    return output.split()[0]


def file_set_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected screen-space placement schema")
    if tuple(config["outfits"]) != ("O01", "O08"):
        raise ValueError("outfit protocol changed")
    if tuple(config["conditions"]) != (
        "cond_000000", "cond_000318", "cond_000017", "cond_000347",
    ):
        raise ValueError("condition protocol changed")
    if config["placement"]["objective_name"] != OBJECTIVE_NAME:
        raise ValueError("placement objective name changed")
    if config["support_render"]["name"] != SUPPORT_RENDER_NAME:
        raise ValueError("support renderer name changed")
    if float(config["support_render"]["opacity_center_value"]) != 0.05:
        raise ValueError("support opacity contract changed")
    if float(config["placement"]["coverage_threshold"]) != 0.65:
        raise ValueError("coverage threshold changed")
    if float(config["placement"]["spill_threshold"]) != 0.10:
        raise ValueError("spill threshold changed")
    if int(config["gradient_calibration"]["optimizer_steps"]) != 20:
        raise ValueError("calibration step contract changed")
    if int(config["G1"]["gate_step"]) != 400 or int(config["G1"]["steps"]) != 1000:
        raise ValueError("G1 schedule changed")
    validate_pool_source_names(config["editable_pool"]["source_names"])
    return config


def _p0(base: Any) -> CanonicalGaussianOverrides:
    return CanonicalGaussianOverrides(
        xyz=base._xyz, scaling=base._scaling, rotation=base._rotation,
        opacity=base._opacity, sh0=base._sh0, shN=base._shN,
    ).validate(base)


def _projected_center_hits(
    base: Any,
    samples: Mapping[str, Mapping[str, Any]],
    background: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    count = int(base._xyz.shape[0])
    hits = {
        name: torch.zeros(count, dtype=torch.int16, device=device)
        for name in ("visible", "old_clothing", "garment_envelope", "protected")
    }
    per_view: dict[str, Any] = {}
    for condition in config["conditions"]:
        state = target_free_state(samples[condition], device)
        _, base_alpha, _, tensors = render_explicit(base, state, _p0(base), background)
        camera = build_mmlphuman_camera(
            state["camera"], state["camera"]["height"], state["camera"]["width"], device,
        )
        homogeneous = torch.cat(
            (tensors["posed_xyz"], torch.ones_like(tensors["posed_xyz"][:, :1])), dim=1,
        ) @ camera["w2c"].T
        normalized = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1e-8)
        pixel = normalized @ camera["K"][:2, :2].T + camera["K"][:2, 2]
        x, y = pixel[:, 0].round().long(), pixel[:, 1].round().long()
        height, width = base_alpha.shape[-2:]
        valid = (
            (homogeneous[:, 2] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        )
        indices = torch.where(valid)[0]
        xx, yy = x[indices], y[indices]
        visible = base_alpha[0, yy, xx] >= float(config["editable_pool"]["base_visibility_alpha_min"])
        indices, xx, yy = indices[visible], xx[visible], yy[visible]
        base_foreground = samples[condition]["target_base_foreground_mask"].to(device)[0, yy, xx] >= 0.5
        old = samples[condition]["target_old_clothing_mask"].to(device)[0, yy, xx] >= 0.5
        protected = samples[condition]["target_protected_mask"].to(device)[0, yy, xx] >= 0.5
        hits["visible"][indices] += base_foreground.to(torch.int16)
        hits["old_clothing"][indices] += old.to(torch.int16)
        hits["garment_envelope"][indices] += (base_foreground & ~protected).to(torch.int16)
        hits["protected"][indices] += protected.to(torch.int16)
        per_view[condition] = {
            "base_foreground_center_hits": int(base_foreground.sum()),
            "old_clothing_center_hits": int(old.sum()),
            "nonprotected_envelope_center_hits": int((base_foreground & ~protected).sum()),
            "protected_center_hits": int(protected.sum()),
            "in_frame_alpha_visible": int(indices.numel()),
        }
    return hits, per_view


def _dominant_anchor(base: Any) -> torch.Tensor:
    strongest = base.nbr_gs_invdist.argmax(dim=1, keepdim=True)
    return base.nbr_gs.gather(1, strongest).squeeze(1).long()


def _source_outfit_parity(
    left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "target_base_foreground_mask", "target_old_clothing_mask", "target_protected_mask",
    )
    rows = []
    for condition in config["conditions"]:
        for field in fields:
            equal = bool(torch.equal(left[condition][field], right[condition][field]))
            rows.append({"condition": condition, "field": field, "bitwise_equal": equal})
    return {"rows": rows, "all_equal": all(row["bitwise_equal"] for row in rows)}


def _instrumented_n_pre(config: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    path = Path(config["source_instrumented_output"]) / "stage_comparison/pixel_support_aggregates.csv"
    lookup: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["state"] in STATES and row["condition"] in FOCUS:
                lookup[(row["state"], row["condition"])] = float(row["n_pre_mean"])
    if len(lookup) != 6:
        raise RuntimeError("instrumented N_pre evidence is incomplete")
    return lookup


def _save_support(path: Path, support: torch.Tensor) -> None:
    value = support.detach().cpu()[0].clamp(0, 1).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(value * 65535).astype(np.uint16), mode="I;16").save(path)


def _contact_sheet(path: Path, entries: list[tuple[str, Path]]) -> None:
    tile = (384, 576)
    canvas = Image.new("RGB", (tile[0] * 3, tile[1] * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image_path) in enumerate(entries):
        image = Image.open(image_path).convert("L").convert("RGB")
        image.thumbnail((tile[0], tile[1] - 32), Image.Resampling.LANCZOS)
        x = (index % 3) * tile[0] + (tile[0] - image.width) // 2
        y = (index // 3) * tile[1] + 28
        canvas.paste(image, (x, y))
        draw.text(((index % 3) * tile[0] + 8, (index // 3) * tile[1] + 6), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _historical_citation(config: Mapping[str, Any], n_pre: Mapping[tuple[str, str], float]) -> dict[str, Any]:
    return {
        "status": "CITATION_ONLY_NOT_RERUN",
        "candidate": "V6.1 S1 / P3",
        "source": config["source_instrumented_output"],
        "n_pre": {
            view: n_pre[("P3", condition)]
            for condition, view in (("cond_000318", "back"), ("cond_000347", "right"))
        },
        "optimizer_steps": 0,
        "historical_status_unchanged": True,
    }


def _artifact_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith(".tmp"):
            rows.append({
                "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    atomic_json(root / "contract/artifact_manifest.json", {"artifacts": rows})


def run_static_proxy(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError(f"append-only output already exists: {root}")
    for name in OUTPUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=False)
    status_path = root / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "RUNNING", "stage": "contract", "optimizer_steps": 0})
    stage = "contract"
    started = time.perf_counter()
    try:
        head = git("rev-parse", "HEAD")
        if git("branch", "--show-current") != config["research_branch"]:
            raise RuntimeError("formal run is not on the registered research branch")
        if git("status", "--short"):
            raise RuntimeError("formal run requires a clean cloud worktree")
        if git("rev-list", "-n", "1", config["source_tag"]) != config["source_head"]:
            raise RuntimeError("source annotated tag drift")
        frozen_before = {branch: remote_ref(branch) for branch in config["frozen_branches"]}
        if frozen_before != config["frozen_branches"]:
            raise RuntimeError("frozen branch drift before run")
        source_files = [
            Path(config["source_instrumented_output"]) / "final_adjudication/INSTRUMENTED_PROJECTION_ADMISSION_FINAL_STATUS.json",
            Path(config["source_instrumented_output"]) / "stage_comparison/pixel_support_aggregates.csv",
            Path(config["source_instrumented_output"]) / "stage_comparison/same_index_displacement_summary.json",
        ]
        source_hash_before = file_set_hash(source_files)
        checkpoints = []
        for state in STATES:
            for outfit in config["outfits"]:
                path = _checkpoint_path(config, state, outfit)
                actual = sha256(path)
                if actual != config["states"][state]["sha256"][outfit]:
                    raise RuntimeError(f"checkpoint drift: {state}/{outfit}")
                checkpoints.append({"state": state, "outfit": outfit, "path": str(path), "sha256": actual})
        atomic_json(root / "contract/config_resolved.json", config)
        atomic_json(root / "contract/run_manifest.json", {
            "schema_version": SCHEMA, "task_id": config["task_id"], "run_commit": head,
            "git": _git_state(), "environment": _environment(), "checkpoints": checkpoints,
            "optimizer_steps": 0, "optimizer_created": False, "permissions": config["permissions"],
            "source_evidence_hash_before": source_hash_before,
        })

        stage = "input_audit"
        atomic_json(status_path, {"status": "RUNNING", "stage": stage, "optimizer_steps": 0})
        base, o01, background, device = load_runtime(config, "O01", args.device)
        o08 = _load_samples(Path(config["source_manifest"]), "O08")
        attach_artifact_masks(o08)
        parity = _source_outfit_parity(o01, o08, config)
        if not parity["all_equal"]:
            raise RuntimeError("base-derived pool inputs differ across outfits")
        base_before = _tensor_state_fingerprint(_base_named_tensors(base))
        atomic_json(root / "input_audit/base_source_parity.json", parity)
        atomic_json(root / "input_audit/target_independence_contract.json", {
            "pool_source_names": config["editable_pool"]["source_names"],
            "target_rgb_used": False, "target_clothing_used": False,
            "target_silhouette_used": False, "outfit_id_used": False,
            "p1_parameters_used": False, "color_rule_used": False,
            "shared_across_outfits": parity["all_equal"],
        })

        stage = "editable_gaussian_pool"
        hits, per_view_hits = _projected_center_hits(base, o01, background, device, config)
        dominant = _dominant_anchor(base)
        pool = build_editable_gaussian_pool(
            old_clothing_hits=hits["old_clothing"],
            garment_envelope_hits=hits["garment_envelope"],
            protected_hits=hits["protected"], visible_hits=hits["visible"],
            dominant_anchor=dominant, anchor_neighbors=base.nbr_vt,
            protected_min_views=int(config["editable_pool"]["protected_min_views"]),
        )
        pool_indices = pool.indices.detach().cpu().numpy().astype(np.int64)
        pool_path = root / "editable_gaussian_pool/editable_gaussian_pool_indices.npy"
        np.save(pool_path, pool_indices, allow_pickle=False)
        pool_audit = {
            "status": "PASS", "gaussian_count": int(pool.indices.numel()),
            "total_base_gaussians": int(base._xyz.shape[0]),
            "seed_count": int(pool.seed_mask.sum()),
            "old_clothing_source_count": int(((hits["old_clothing"] > 0) & pool.seed_mask).sum()),
            "nonprotected_envelope_source_count": int(((hits["garment_envelope"] > 0) & pool.seed_mask).sum()),
            "graph_neighbor_added_count": int(pool.graph_added_mask.sum()),
            "protected_excluded_count": int(pool.protected_excluded_mask.sum()),
            "expanded_anchor_count": int(pool.expanded_anchor_mask.sum()),
            "indices_sha256": sha256(pool_path), "tensor_sha256": tensor_hash(pool.indices),
            "shared_across_outfits": parity["all_equal"],
            "target_independent": True, "definition": config["editable_pool"],
            "per_base_view": per_view_hits,
        }
        atomic_json(root / "editable_gaussian_pool/editable_gaussian_pool.json", pool_audit)
        atomic_text(root / "editable_gaussian_pool/EDITABLE_GAUSSIAN_POOL_AUDIT.md", "\n".join([
            "# Editable Gaussian Pool Audit", "",
            f"- total: `{pool_audit['gaussian_count']}` / `{pool_audit['total_base_gaussians']}`",
            f"- seed: `{pool_audit['seed_count']}`",
            f"- old-clothing source: `{pool_audit['old_clothing_source_count']}`",
            f"- nonprotected envelope source: `{pool_audit['nonprotected_envelope_source_count']}`",
            f"- graph-only additions: `{pool_audit['graph_neighbor_added_count']}`",
            f"- protected exclusions: `{pool_audit['protected_excluded_count']}`",
            f"- SHA256: `{pool_audit['indices_sha256']}`",
            "- O01/O08 share the same base-derived source tensors: `true`",
            "- target RGB/clothing/silhouette, outfit ID, P1 parameters, and color rules used: `false`",
        ]))

        stage = "support_render_validation"
        n_pre = _instrumented_n_pre(config)
        rows: list[dict[str, Any]] = []
        panels: list[tuple[str, Path]] = []
        for state_name in STATES:
            _, overrides = load_state_model(config, state_name, "O08", base)
            for condition in FOCUS:
                state = target_free_state(o08[condition], device)
                camera = build_mmlphuman_camera(
                    state["camera"], state["camera"]["height"], state["camera"]["width"], device,
                )
                with mmlphuman_state_transaction(base, state["pose"], state["Rh"], state["Th"]):
                    posed_xyz = base.compute_xyz(overrides.as_dict())
                    posed_covariance = base.get_covariance(canonical_overrides=overrides.as_dict())
                with torch.no_grad():
                    support, _ = fixed_attribute_support_render(
                        posed_xyz, posed_covariance, pool.indices,
                        camera, opacity=float(config["support_render"]["opacity_center_value"]),
                    )
                regions_with_batch = trusted_regions(o08[condition], config)
                regions = {name: value[0].to(device) for name, value in regions_with_batch.items()}
                target, forbidden = build_placement_masks(o08[condition], regions)
                target, forbidden = target.to(device) >= 0.5, forbidden.to(device) >= 0.5
                view = config["conditions"][condition]
                support_path = root / f"support_render_validation/{state_name}_{view}_support.png"
                _save_support(support_path, support)
                panels.append((f"{state_name} {view}", support_path))
                rows.append({
                    "state": state_name, "condition": condition, "view": view,
                    "mean_trusted_support": float(support[target].mean()) if target.any() else 0.0,
                    "low_support_hole_ratio": float((support[target] < 0.65).float().mean()) if target.any() else 0.0,
                    "spill_mean": float(support[forbidden].mean()) if forbidden.any() else 0.0,
                    "spill_active_fraction": float((support[forbidden] > 0.10).float().mean()) if forbidden.any() else 0.0,
                    "target_active_pixels": int(target.sum()), "forbidden_active_pixels": int(forbidden.sum()),
                    "instrumented_n_pre": n_pre[(state_name, condition)],
                    "opacity": 0.05, "attributes_with_gradient": ["xyz"],
                })
        write_csv(root / "support_render_validation/support_proxy_static_results.csv", rows)
        atomic_json(root / "support_render_validation/support_proxy_static_results.json", rows)
        _contact_sheet(root / "visual_acceptance/support_proxy_contact_sheet.png", panels)
        qualification = qualify_support_proxy(
            rows,
            minimum_p1_margin=float(config["proxy_qualification"]["p1_minus_p3_mean_support_min"]),
            minimum_spearman=float(config["proxy_qualification"]["spearman_vs_instrumented_n_pre_min"]),
            underfill_threshold=float(config["proxy_qualification"]["underfill_mean_support_max"]),
        )
        qualification.update({
            "threshold_tuning_used": False, "optimizer_steps": 0,
            "optimizer_created": False, "support_opacity": 0.05,
            "production_renderer_changed": False,
        })
        atomic_json(root / "support_render_validation/support_proxy_qualification.json", qualification)
        atomic_text(root / "support_render_validation/SUPPORT_PROXY_QUALIFICATION.md", "\n".join([
            "# Fixed-attribute support proxy qualification", "",
            *[f"- {row['state']} {row['view']}: mean={row['mean_trusted_support']:.9f}, hole={row['low_support_hole_ratio']:.9f}, spill={row['spill_active_fraction']:.9f}, N_pre={row['instrumented_n_pre']:.9f}" for row in rows],
            f"- P1-P3 back margin: `{qualification['p1_minus_p3_margin']['back']:.9f}` (required >=0.15)",
            f"- P1-P3 right margin: `{qualification['p1_minus_p3_margin']['right']:.9f}` (required >=0.15)",
            f"- Spearman vs instrumented N_pre: `{qualification['spearman']:.9f}` (required >=0.80)",
            f"- status: `{qualification['status']}`",
            "- threshold tuning: `false`",
        ]))

        historical = _historical_citation(config, n_pre)
        atomic_json(root / "G0_historical/P3_CITATION_ONLY.json", historical)
        atomic_text(root / "G0_historical/README.md", "# G0 historical baseline\n\nP3 is cited only. No historical run or result was modified.")
        base_after = _tensor_state_fingerprint(_base_named_tensors(base))
        frozen_after = {branch: remote_ref(branch) for branch in config["frozen_branches"]}
        source_hash_after = file_set_hash(source_files)
        safety = {
            "base_fingerprint_before": base_before, "base_fingerprint_after": base_after,
            "base_bitwise_exact": base_before == base_after,
            "base_gradient_count": _base_gradient_count(base),
            "frozen_branches_unchanged": frozen_before == frozen_after,
            "source_evidence_unchanged": source_hash_before == source_hash_after,
        }
        atomic_json(root / "input_audit/frozen_state_postcheck.json", safety)

        if qualification["status"] != "PASS":
            for directory in (
                "gradient_direction_audit", "gradient_calibration", "G1_placement_objective",
                "G2_placement_warmup_if_eligible", "instrumentation",
            ):
                atomic_json(root / directory / "NOT_RUN.json", {
                    "reason": "SUPPORT_PROXY_INVALID", "optimizer_steps": 0,
                    "required_next_task": config["proxy_qualification"]["failure_next_task"],
                })
            final = {
                "status": "SUPPORT_PROXY_INVALID", "task_id": config["task_id"],
                "run_commit": head, "optimizer_steps": 0, "optimizer_created": False,
                "qualification": qualification, "editable_pool": pool_audit,
                "safety": safety, "G1_status": "NOT_RUN", "G2_status": "NOT_RUN",
                "final_case_cp_cw_cg_cr_cf": "NOT_REACHED_PRETRAINING_PROXY_GATE",
                "v6_2_candidate": False, "seven_outfit_readjudication_allowed": False,
                "target_generation_allowed": False, "formal_image_conditioned_training_allowed": False,
                "next_task": config["proxy_qualification"]["failure_next_task"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(root / "final_adjudication/SCREEN_SPACE_PLACEMENT_FINAL_STATUS.json", final)
            atomic_text(root / "final_adjudication/GATE_ACCEPTANCE.md", "\n".join([
                "# Screen-space placement final adjudication", "",
                "**Status: SUPPORT_PROXY_INVALID**", "",
                "The preregistered fixed-attribute support proxy did not rank P1 above P3 by the required margin and did not preserve the required monotonic relation to instrumented N_pre. This is a proxy qualification failure, not a renderer, checkpoint, V6.1, or model-training failure.", "",
                "No optimizer was created and no gradient audit, calibration, G1, G2, seven-outfit rerun, target generation, or image-conditioned training was performed.", "",
                f"Next task: `{config['proxy_qualification']['failure_next_task']}`.",
            ]))
            atomic_json(root / "visual_acceptance/visual_acceptance.json", {
                "status": "STATIC_PROXY_CONTACT_SHEET_GENERATED_PENDING_OR_COMPLETED_SEPARATE_INSPECTION",
                "image": "support_proxy_contact_sheet.png", "optimizer_steps": 0,
            })
            atomic_json(status_path, final)
            _artifact_manifest(root)
            print(json.dumps(final, indent=2, default=str))
            return
        raise RuntimeError("proxy unexpectedly passed; this runner revision intentionally requires the next registered phase implementation")
    except Exception as error:
        atomic_json(status_path, {
            "status": "TOOL_FAILURE", "failure_stage": stage, "optimizer_steps": 0,
            "optimizer_created": False, "exception_type": type(error).__name__,
            "exception": str(error), "traceback": traceback.format_exc(), "failed_at": now(),
        })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prequalify the V6.2 screen-space placement objective")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("static-proxy",), default="static-proxy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    run_static_proxy(args, config)


if __name__ == "__main__":
    main()
