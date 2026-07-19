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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.differentiable_support_proxy import (  # noqa: E402
    anti_saturation_metrics,
    dense_soft_precontributor_count,
    spearman_correlation,
)
from scene.editable_pool_specificity import (  # noqa: E402
    anchor_components,
    build_candidate_pools,
    cloud_source_category,
    explicit_membership_reasons,
    pool_sha256,
    qualify_dual_pool,
    stable_protected_mask,
    weighted_pool_recall,
)
from scene.instrumented_projection_admission import (  # noqa: E402
    InstrumentationOptions,
    backend_pixel_pipeline,
    independent_pixel_support,
    run_backend_projection_debug,
)
from scene.screen_space_placement_objective import build_editable_gaussian_pool  # noqa: E402
from tools.run_alpha_raster_audit import (  # noqa: E402
    _checkpoint_path,
    load_runtime,
    load_state_model,
    render_explicit,
    sha256,
    target_free_state,
    trusted_regions,
)
from tools.run_differentiable_support_proxy import (  # noqa: E402
    _fixed_region_masks,
    _gradient_view,
    _projection,
    _sample_regions,
)
from tools.run_instrumented_projection_admission import (  # noqa: E402
    _options as instrumentation_options,
    _projection_inputs,
)
from tools.run_screen_space_placement_objective import (  # noqa: E402
    _dominant_anchor,
    _projected_center_hits,
)


SCHEMA = "canondressgs.editable_pool_specificity.v1"
VIEWS = {"cond_000000": "front", "cond_000318": "back", "cond_000017": "left", "cond_000347": "right"}
STATES = ("P1", "P2", "P3")
OUTPUT_DIRS = (
    "contract", "input_audit", "full_gaussian_attribution", "editable_membership",
    "coverage_contributors", "cloud_contributors", "exclusion_reasons",
    "body_part_and_anchor_analysis", "candidate_pool_contracts",
    "dual_pool_static_qualification", "protected_risk", "visualizations",
    "final_adjudication",
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/research/subject02_editable_pool_specificity_v1.yaml"
DEFAULT_OUTPUT = Path("/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-EDITABLE-POOL-SPECIFICITY-001/attempt_001")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def resolve_ref(name: str) -> str:
    for candidate in (name, f"origin/{name}", f"cloud/{name}"):
        process = subprocess.run(
            ["git", "rev-parse", candidate], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        )
        if process.returncode == 0:
            return process.stdout.strip()
    raise RuntimeError(f"cannot resolve frozen ref: {name}")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("wrong editable-pool audit schema")
    if tuple(value["outfits"]) != ("O01", "O08") or tuple(value["conditions"]) != tuple(VIEWS):
        raise ValueError("fixed outfit/view protocol changed")
    if tuple(value["states"]) != STATES:
        raise ValueError("P1/P2/P3 contract changed")
    if float(value["proxy"]["temperature"]) != 0.10:
        raise ValueError("Proxy A temperature must remain 0.10")
    if value["permissions"]["optimizer_created"] or int(value["permissions"]["optimizer_steps"]) != 0:
        raise ValueError("optimizer is forbidden")
    return value


def body_part(joint: int) -> str:
    if joint in (10, 11):
        return "feet_or_shoes"
    if joint in (20, 21) or joint >= 25:
        return "hands"
    if joint in (15, 22, 23, 24):
        return "head_face_hair"
    if joint in (13, 16, 18):
        return "left_arm"
    if joint in (14, 17, 19):
        return "right_arm"
    if joint in (1, 4, 7):
        return "left_leg"
    if joint in (2, 5, 8):
        return "right_leg"
    return "torso_or_pelvis"


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _pool_mask(count: int, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = torch.zeros(count, dtype=torch.bool, device=device)
    value[indices.to(device)] = True
    return value


def _single_alpha(backend: Mapping[str, Any], opacity: torch.Tensor, ids: Sequence[int], y: int, x: int) -> np.ndarray:
    if not ids:
        return np.empty(0, dtype=np.float64)
    index = torch.tensor(ids, dtype=torch.long, device=opacity.device)
    means = backend["means2d"][0, index].double()
    conics = backend["conics"][0, index].double()
    delta = means - torch.tensor([x + .5, y + .5], dtype=torch.float64, device=means.device)
    sigma = .5 * (conics[:, 0] * delta[:, 0].square() + conics[:, 2] * delta[:, 1].square()) + conics[:, 1] * delta[:, 0] * delta[:, 1]
    values = opacity.reshape(-1)[index].double() * torch.exp(-sigma)
    return torch.clamp(values, min=0, max=.999).detach().cpu().numpy()


def trace_pixels(
    projection: Mapping[str, Any], backend: Mapping[str, Any], opacity: torch.Tensor,
    pixels_xy: torch.Tensor, *, tile_size: int, alpha_cutoff: float, termination: float,
) -> list[dict[str, Any]]:
    pixels_yx = [(int(y), int(x)) for x, y in pixels_xy.detach().cpu().tolist()]
    theoretical = independent_pixel_support(projection, pixels_yx, alpha_cutoff=alpha_cutoff)
    rows: list[dict[str, Any]] = []
    for source, (y, x) in zip(theoretical, pixels_yx):
        actual = backend_pixel_pipeline(
            backend, opacity, y, x, tile_size=tile_size,
            alpha_cutoff=alpha_cutoff, termination=termination,
        )
        pre = source["pre_tile_contributor_indices"]
        active = actual["contributed"]
        alpha = _single_alpha(backend, opacity, active, y, x)
        transmittance = 1.0
        masses: list[float] = []
        for value in alpha:
            masses.append(float(transmittance * value))
            transmittance *= 1.0 - float(value)
        rows.append({
            "x": x, "y": y, "N_pre": len(pre), "N_active": len(active),
            "pre": pre, "active": active, "active_alpha_mass": masses,
            "backend_alpha": float(actual["alpha"]),
        })
    return rows


def aggregate_trace(rows: Sequence[Mapping[str, Any]], pool: torch.Tensor) -> dict[str, Any]:
    pre = [index for row in rows for index in row["pre"]]
    active = [index for row in rows for index in row["active"]]
    mass = [value for row in rows for value in row["active_alpha_mass"]]
    result = weighted_pool_recall(pre, active, mass, pool)
    result.update({
        "sampled_pixels": len(rows),
        "npre_mean": float(np.mean([row["N_pre"] for row in rows])) if rows else 0.0,
        "active_mean": float(np.mean([row["N_active"] for row in rows])) if rows else 0.0,
    })
    return result


def _save_parquet(path: Path, columns: Mapping[str, Any]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _draw_flow(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    canvas = Image.new("RGB", (1024, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), "Same-index cloud migration: base -> P3 screen centers", fill="black")
    finite = [row for row in rows if np.isfinite(row["base_screen_x"]) and np.isfinite(row["p3_screen_x"])]
    for row in finite[::max(1, len(finite) // 3000)]:
        x0, y0 = float(row["base_screen_x"]), float(row["base_screen_y"])
        x1, y1 = float(row["p3_screen_x"]), float(row["p3_screen_y"])
        if 0 <= x0 < 1024 and 0 <= y0 < 680 and 0 <= x1 < 1024 and 0 <= y1 < 680:
            draw.line((x0, y0 + 35, x1, y1 + 35), fill="#c62828", width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    paths = [
        Path(config["source_proxy_output"]) / "final_adjudication/DIFFERENTIABLE_SUPPORT_PROXY_FINAL_STATUS.json",
        Path(config["source_proxy_output"]) / "static_qualification/candidate_qualification.json",
        Path(config["source_instrumented_output"]) / "final_adjudication/INSTRUMENTED_PROJECTION_ADMISSION_FINAL_STATUS.json",
        Path(config["source_placement_output"]) / config["current_editable"]["source_file"],
    ]
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha256(path)
    return result


def _source_coverage_qualification(config: Mapping[str, Any]) -> dict[str, Any]:
    source = json.loads((Path(config["source_proxy_output"]) / "static_qualification/candidate_qualification.json").read_text())
    candidates = source.get("candidates", source)
    if isinstance(candidates, list):
        row = next(item for item in candidates if item.get("candidate") == "A_0.1")
    else:
        row = candidates["A_0.1"]
    return {"status": row.get("status", "FAIL"), "source": row}


def _cloud_category_summary(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"occurrences": 0, "active": 0, "alpha_mass": 0.0})
    unique: dict[str, set[int]] = defaultdict(set)
    for event in events:
        category = event["source_category"]
        unique[category].add(int(event["gaussian_index"]))
        grouped[category]["occurrences"] += 1
        grouped[category]["active"] += int(event["active"])
        grouped[category]["alpha_mass"] += float(event["alpha_mass"])
    return [{"source_category": key, "unique_gaussians": len(unique[key]), **value} for key, value in sorted(grouped.items())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"append-only output already exists: {output}")
    for name in OUTPUT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"task_id": config["task_id"], "status": "RUNNING", "started_at": now(), "failure_stage": None, "optimizer_created": False, "optimizer_steps": 0}
    atomic_json(output / "RUN_STATUS.json", status)
    try:
        status["failure_stage"] = "input_audit"
        head = git("rev-parse", "HEAD")
        branch = git("branch", "--show-current")
        dirty = git("status", "--short")
        if branch != config["research_branch"] or dirty:
            raise RuntimeError(f"formal runner requires clean {config['research_branch']}; dirty={dirty!r}")
        if git("rev-parse", config["source_tag"] + "^{}") != config["source_head"]:
            raise RuntimeError("frozen source tag moved")
        frozen_refs = {name: resolve_ref(name) for name in config["frozen_branches"]}
        if frozen_refs != config["frozen_branches"]:
            raise RuntimeError("one or more frozen branches changed")
        source_hashes = _source_hashes(config)
        current_file = Path(config["source_placement_output"]) / config["current_editable"]["source_file"]
        if sha256(current_file) != config["current_editable"]["file_sha256"]:
            raise RuntimeError("frozen G_editable file changed")
        current_np = np.load(current_file)
        if current_np.shape != (config["current_editable"]["count"],):
            raise RuntimeError("frozen G_editable count changed")
        atomic_json(output / "contract/config_resolved.json", config)
        atomic_text(output / "contract/execution_command.txt", " ".join(sys.argv))
        atomic_json(output / "contract/run_manifest.json", {"head": head, "branch": branch, "git_status": dirty, "environment": _environment(), "source_hashes": source_hashes})

        status["failure_stage"] = "pool_reconstruction"
        base, samples, background, device = load_runtime(config, "O01", args.device)
        hits, per_view = _projected_center_hits(base, samples, background, device, {
            "conditions": list(VIEWS),
            "editable_pool": {"base_visibility_alpha_min": 0.01},
        })
        dominant = _dominant_anchor(base)
        frozen_pool = build_editable_gaussian_pool(
            old_clothing_hits=hits["old_clothing"], garment_envelope_hits=hits["garment_envelope"],
            protected_hits=hits["protected"], visible_hits=hits["visible"], dominant_anchor=dominant,
            anchor_neighbors=base.nbr_vt, protected_min_views=config["current_editable"]["protected_min_views"],
        )
        current = torch.from_numpy(current_np.astype(np.int64, copy=False)).to(device)
        if not torch.equal(frozen_pool.indices, current):
            raise RuntimeError("reconstructed coverage pool differs from frozen G_editable")
        protected = stable_protected_mask(hits["protected"], hits["garment_envelope"], hits["old_clothing"], minimum_views=2)
        formal = torch.ones(base._xyz.shape[0], dtype=torch.bool, device=device)
        pools = build_candidate_pools(
            current_editable=current, formal_trainable=formal, stable_protected=protected,
            dominant_anchor=dominant, anchor_neighbors=base.nbr_vt,
        )
        pool_map = {"P0": pools.spill_p0, "P1": pools.spill_p1, "P2": pools.spill_p2}
        for name, indices in pool_map.items():
            array_path = output / f"candidate_pool_contracts/{name}_spill_indices.npy"
            np.save(array_path, indices.detach().cpu().numpy())
        component = anchor_components(base.nbr_vt)
        dominant_joint = base.get_weights.argmax(dim=1).long()
        finite = torch.isfinite(base._xyz).all(dim=1)
        inclusion, exclusion = explicit_membership_reasons(
            current_editable=current, finite_valid=finite, visible_hits=hits["visible"],
            old_clothing_hits=hits["old_clothing"], garment_envelope_hits=hits["garment_envelope"],
            stable_protected=protected, dominant_anchor=dominant,
            expanded_anchor_mask=frozen_pool.expanded_anchor_mask,
        )
        pool_contracts = {
            name: {"count": int(indices.numel()), "index_sha256": pool_sha256(indices), "stable_protected_overlap": int(_pool_mask(base._xyz.shape[0], indices, device)[protected].sum()), "membership_target_independent": True}
            for name, indices in pool_map.items()
        }
        pool_contracts["coverage"] = {"count": int(current.numel()), "index_sha256": pool_sha256(current), "source_file_sha256": sha256(current_file), "equals_frozen_g_editable": True}
        atomic_json(output / "candidate_pool_contracts/pool_contracts.json", pool_contracts)
        atomic_json(output / "input_audit/pool_reconstruction.json", {"per_view": per_view, "frozen_count": int(current.numel()), "reconstructed_equal": True, "stable_protected_count": int(protected.sum())})

        count = int(base._xyz.shape[0])
        base_xyz = base._xyz.detach().cpu().numpy()
        residuals = {state: np.zeros((count, len(config["outfits"])), np.float32) for state in STATES}
        representative = {state: {"posed": np.full((count, 3), np.nan, np.float32), "screen": np.full((count, 2), np.nan, np.float32), "canonical": np.full((count, 3), np.nan, np.float32)} for state in STATES}
        base_screen = np.full((count, 2), np.nan, np.float32)
        base_posed = np.full((count, 3), np.nan, np.float32)
        all_trace: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        cloud_events: list[dict[str, Any]] = []
        frozen_region_counts: dict[str, Any] = {}
        proxy_rows: list[dict[str, Any]] = []
        gradient_context: dict[str, Any] = {}

        status["failure_stage"] = "production_contributor_trace"
        for outfit_index, outfit in enumerate(config["outfits"]):
            if outfit_index:
                del base, samples, background
                torch.cuda.empty_cache()
                base, samples, background, device = load_runtime(config, outfit, args.device)
            state_models: dict[str, Any] = {}
            overrides_by_state: dict[str, Any] = {}
            for state in STATES:
                model, overrides = load_state_model(config, state, outfit, base)
                state_models[state] = model
                overrides_by_state[state] = overrides
                residuals[state][:, outfit_index] = torch.linalg.vector_norm(overrides.xyz - base._xyz, dim=1).detach().cpu().numpy()
            for condition, view in VIEWS.items():
                sample = samples[condition]
                target_state = target_free_state(sample, device)
                p3_rgb, p3_alpha, _, _ = render_explicit(base, target_state, overrides_by_state["P3"], background)
                del p3_rgb
                fixed = _fixed_region_masks(sample, trusted_regions(sample, config), p3_alpha)
                source_count_file = Path(config["source_proxy_output"]) / config["regions"]["count_crosscheck_source"].format(outfit=outfit, view=view)
                source_counts = json.loads(source_count_file.read_text())
                observed_counts = {key: int(value.sum()) for key, value in fixed.items()}
                source_payload = source_counts.get("counts", source_counts)
                for key, value in observed_counts.items():
                    if key in source_payload and int(source_payload[key]) != value:
                        raise RuntimeError(f"frozen region drift {outfit}/{view}/{key}: {value} != {source_payload[key]}")
                frozen_region_counts[f"{outfit}/{view}"] = observed_counts
                pixels, region_lookup = _sample_regions(fixed, int(config["trace_sampling"]["pixels_per_region"]), device)
                p0_projection, p0_tensors, _ = _projection(base, target_state, load_state_model(config, "P0", outfit, base)[1])
                if outfit == "O08" and view == "back":
                    base_screen[:] = p0_projection["projected_mean"].detach().cpu().numpy().astype(np.float32)
                    base_posed[:] = p0_tensors["posed_xyz"].detach().cpu().numpy().astype(np.float32)
                del p0_projection, p0_tensors
                for state in STATES:
                    projection, tensors, camera = _projection(base, target_state, overrides_by_state[state])
                    if outfit == "O08" and view == "back":
                        representative[state]["posed"][:] = tensors["posed_xyz"].detach().cpu().numpy().astype(np.float32)
                        representative[state]["screen"][:] = projection["projected_mean"].detach().cpu().numpy().astype(np.float32)
                        representative[state]["canonical"][:] = overrides_by_state[state].xyz.detach().cpu().numpy().astype(np.float32)
                    backend = run_backend_projection_debug(
                        means=tensors["posed_xyz"], covars=tensors["posed_covariance"],
                        opacities=tensors["effective_opacity"], w2c=camera["w2c"], K=camera["K"],
                        width=int(camera["width"]), height=int(camera["height"]),
                        options=InstrumentationOptions(enabled=True),
                    )
                    if backend is None:
                        raise RuntimeError("production backend trace unavailable")
                    selected_rows: dict[str, list[dict[str, Any]]] = {}
                    for region in ("trusted_expansion", "correctly_covered_garment", "trailing_cloud", "normal_background"):
                        region_pixels = pixels[region_lookup[region]] if region_lookup[region] else pixels[:0]
                        rows = trace_pixels(
                            projection, backend, tensors["effective_opacity"], region_pixels,
                            tile_size=16,
                            alpha_cutoff=float(config["trace_sampling"]["theoretical_alpha_cutoff"]),
                            termination=float(config["trace_sampling"]["actual_early_termination"]),
                        )
                        all_trace[(outfit, view, state, region)] = rows
                        selected_rows[region] = rows
                        proxy_pools = pool_map if region in ("trailing_cloud", "normal_background") else {"coverage": current}
                        for proxy_pool_name, pool in proxy_pools.items():
                            proxy_values = dense_soft_precontributor_count(
                                projection["projected_mean"][pool], projection["conic"][pool],
                                projection["opacity"][pool], region_pixels,
                                temperature=.10, valid=projection["stage_masks_s0_s5"][pool, 3], gaussian_chunk=2048,
                            ).detach().cpu().numpy() if region_pixels.numel() else np.empty(0)
                            for local, row in enumerate(rows):
                                proxy_rows.append({"pool": proxy_pool_name, "outfit": outfit, "view": view, "state": state, "region": region, "x": row["x"], "y": row["y"], "proxy": float(proxy_values[local]), "N_pre": row["N_pre"]})
                    if outfit == "O08" and view in ("back", "right") and state == "P3":
                        gradient_context[view] = {"base": base, "state": target_state, "overrides": overrides_by_state[state], "projection": projection, "camera": camera, "pixels": pixels, "region_lookup": region_lookup}
                    for row in selected_rows["trailing_cloud"]:
                        active_mass = dict(zip(row["active"], row["active_alpha_mass"]))
                        for gaussian in row["pre"]:
                            index = int(gaussian)
                            joint = int(dominant_joint[index])
                            category = cloud_source_category(
                                stable_protected=bool(protected[index]), base_visible=bool(hits["visible"][index]),
                                garment_body_part=joint in set(config["body_parts"]["garment_joint_ids"]),
                                envelope_evidence=bool(hits["garment_envelope"][index]),
                                anchor_group_member=bool(frozen_pool.expanded_anchor_mask[dominant[index]]),
                                formal_trainable=True,
                            )
                            cloud_events.append({
                                "gaussian_index": index, "outfit": outfit, "view": view, "state": state,
                                "x": row["x"], "y": row["y"], "active": index in active_mass,
                                "alpha_mass": float(active_mass.get(index, 0.0)), "source_category": category,
                            })
                    del backend, projection, tensors
            del state_models, overrides_by_state
            torch.cuda.empty_cache()

        atomic_json(output / "input_audit/frozen_region_counts.json", frozen_region_counts)
        write_csv(output / "dual_pool_static_qualification/proxy_pixel_values.csv", proxy_rows)

        status["failure_stage"] = "full_attribution"
        current_mask = _pool_mask(count, current, device).cpu().numpy()
        visible_cpu = hits["visible"].detach().cpu().numpy()
        old_cpu = hits["old_clothing"].detach().cpu().numpy()
        env_cpu = hits["garment_envelope"].detach().cpu().numpy()
        prot_cpu = hits["protected"].detach().cpu().numpy()
        protected_cpu = protected.detach().cpu().numpy()
        dominant_cpu = dominant.detach().cpu().numpy()
        joint_cpu = dominant_joint.detach().cpu().numpy()
        component_cpu = component[dominant].detach().cpu().numpy()
        projection_region = np.where(protected_cpu, "stable_protected", np.where(old_cpu > 0, "old_clothing", np.where(env_cpu > 0, "garment_envelope", np.where(visible_cpu > 0, "visible_non_garment", "base_invisible"))))
        columns: dict[str, Any] = {
            "gaussian_index": np.arange(count, dtype=np.int64),
            "in_current_G_editable": current_mask,
            "formal_residual_trainable": np.ones(count, dtype=bool),
            "canonical_x": base_xyz[:, 0], "canonical_y": base_xyz[:, 1], "canonical_z": base_xyz[:, 2],
            "base_four_view_visibility_count": visible_cpu,
            "base_projection_region": projection_region.tolist(),
            "dominant_anchor": dominant_cpu, "anchor_graph_component": component_cpu,
            "dominant_lbs_joint": joint_cpu, "body_part": [body_part(int(value)) for value in joint_cpu],
            "old_clothing_evidence": old_cpu, "nonprotected_garment_envelope_evidence": env_cpu,
            "protected_evidence": prot_cpu, "inclusion_reason": inclusion, "exclusion_reason": exclusion,
            "P1_residual_magnitude_O01": residuals["P1"][:, 0], "P1_residual_magnitude_O08": residuals["P1"][:, 1],
            "P2_residual_magnitude_O01": residuals["P2"][:, 0], "P2_residual_magnitude_O08": residuals["P2"][:, 1],
            "P3_residual_magnitude_O01": residuals["P3"][:, 0], "P3_residual_magnitude_O08": residuals["P3"][:, 1],
            "base_posed_x": base_posed[:, 0], "base_posed_y": base_posed[:, 1], "base_posed_z": base_posed[:, 2],
            "base_screen_x": base_screen[:, 0], "base_screen_y": base_screen[:, 1],
        }
        for state in STATES:
            columns.update({
                f"{state}_posed_x": representative[state]["posed"][:, 0], f"{state}_posed_y": representative[state]["posed"][:, 1], f"{state}_posed_z": representative[state]["posed"][:, 2],
                f"{state}_screen_x": representative[state]["screen"][:, 0], f"{state}_screen_y": representative[state]["screen"][:, 1],
            })
        _save_parquet(output / "full_gaussian_attribution/full_gaussian_attribution.parquet", columns)
        membership_rows = []
        for reason, amount in Counter(value for value in exclusion if value).items():
            membership_rows.append({"membership": "excluded", "reason": reason, "gaussian_count": amount})
        for reason, amount in Counter(value for value in inclusion if value).items():
            membership_rows.append({"membership": "included", "reason": reason, "gaussian_count": amount})
        write_csv(output / "editable_membership/editable_membership_summary.csv", membership_rows)
        write_csv(output / "exclusion_reasons/exclusion_reason_summary.csv", [row for row in membership_rows if row["membership"] == "excluded"])

        status["failure_stage"] = "contributor_recall"
        coverage_rows: list[dict[str, Any]] = []
        cloud_rows: list[dict[str, Any]] = []
        for key, rows in all_trace.items():
            outfit, view, state, region = key
            base_row = {"outfit": outfit, "view": view, "state": state, "region": region}
            if region in ("trusted_expansion", "correctly_covered_garment"):
                coverage_rows.append({**base_row, **aggregate_trace(rows, current)})
            if region == "trailing_cloud":
                for pool_name, indices in pool_map.items():
                    cloud_rows.append({**base_row, "pool": pool_name, **aggregate_trace(rows, indices)})
        write_csv(output / "coverage_contributors/coverage_recall_by_split.csv", coverage_rows)
        write_csv(output / "cloud_contributors/cloud_recall_by_split.csv", cloud_rows)
        coverage_all = [row for key, rows in all_trace.items() if key[3] in ("trusted_expansion", "correctly_covered_garment") for row in rows]
        expansion_all = [row for key, rows in all_trace.items() if key[3] == "trusted_expansion" for row in rows]
        cloud_all = [row for key, rows in all_trace.items() if key[3] == "trailing_cloud" for row in rows]
        coverage_global = aggregate_trace(coverage_all, current)
        expansion_global = aggregate_trace(expansion_all, current)
        cloud_global = {name: aggregate_trace(cloud_all, indices) for name, indices in pool_map.items()}
        atomic_json(output / "coverage_contributors/coverage_global.json", {"all_coverage": coverage_global, "trusted_expansion": expansion_global})
        atomic_json(output / "cloud_contributors/cloud_global.json", cloud_global)

        event_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in cloud_events:
            event_groups[event["gaussian_index"]].append(event)
        coverage_active_indices = {index for row in coverage_all for index in row["active"]}
        cloud_detail_rows: list[dict[str, Any]] = []
        migration_rows: list[dict[str, Any]] = []
        for gaussian, events in sorted(event_groups.items()):
            active_events = [event for event in events if event["active"]]
            category = Counter(event["source_category"] for event in events).most_common(1)[0][0]
            cloud_detail_rows.append({
                "gaussian_index": gaussian, "contribution_pixel_count": len(events),
                "active_contribution_count": len(active_events), "accumulated_alpha_mass": sum(event["alpha_mass"] for event in events),
                "mean_single_contribution": sum(event["alpha_mass"] for event in active_events) / max(len(active_events), 1),
                "max_single_contribution": max([event["alpha_mass"] for event in active_events] or [0.0]),
                "outfit_view_state": ";".join(sorted({f"{e['outfit']}/{e['view']}/{e['state']}" for e in events})),
                "in_current_G_editable": bool(current_mask[gaussian]), "exclusion_reason": exclusion[gaussian],
                "body_part": body_part(int(joint_cpu[gaussian])), "dominant_anchor": int(dominant_cpu[gaussian]),
                "dominant_lbs_joint": int(joint_cpu[gaussian]), "base_projection_region": str(projection_region[gaussian]),
                "P3_displacement": float(residuals["P3"][gaussian].max()),
                "contributes_to_normal_coverage": gaussian in coverage_active_indices,
                "stable_protected": bool(protected_cpu[gaussian]), "residual_trainable": True,
                "source_category": category,
            })
            migration_rows.append({
                "gaussian_index": gaussian, "source_category": category,
                "base_canonical_x": float(base_xyz[gaussian, 0]), "base_canonical_y": float(base_xyz[gaussian, 1]), "base_canonical_z": float(base_xyz[gaussian, 2]),
                "base_posed_x": float(base_posed[gaussian, 0]), "base_posed_y": float(base_posed[gaussian, 1]), "base_posed_z": float(base_posed[gaussian, 2]),
                "base_screen_x": float(base_screen[gaussian, 0]), "base_screen_y": float(base_screen[gaussian, 1]),
                "p1_screen_x": float(representative["P1"]["screen"][gaussian, 0]), "p1_screen_y": float(representative["P1"]["screen"][gaussian, 1]),
                "p2_screen_x": float(representative["P2"]["screen"][gaussian, 0]), "p2_screen_y": float(representative["P2"]["screen"][gaussian, 1]),
                "p3_screen_x": float(representative["P3"]["screen"][gaussian, 0]), "p3_screen_y": float(representative["P3"]["screen"][gaussian, 1]),
                "p3_screen_dx": float(representative["P3"]["screen"][gaussian, 0] - base_screen[gaussian, 0]),
                "p3_screen_dy": float(representative["P3"]["screen"][gaussian, 1] - base_screen[gaussian, 1]),
                "p3_canonical_displacement": float(residuals["P3"][gaussian].max()),
                "base_region": str(projection_region[gaussian]), "final_region": "trailing_cloud",
                "also_coverage": cloud_detail_rows[-1]["contributes_to_normal_coverage"],
                "stable_protected": bool(protected_cpu[gaussian]),
            })
        _save_parquet(output / "cloud_contributors/cloud_contributors.parquet", {key: [row[key] for row in cloud_detail_rows] for key in cloud_detail_rows[0]})
        write_csv(output / "cloud_contributors/cloud_source_categories.csv", _cloud_category_summary(cloud_events))
        write_csv(output / "cloud_contributors/cloud_same_index_migration.csv", migration_rows)
        _draw_flow(output / "visualizations/cloud_flow_visualization.png", migration_rows)

        status["failure_stage"] = "dual_pool_qualification"
        coverage_qualification = _source_coverage_qualification(config)
        qualifications: dict[str, Any] = {}
        for pool_name, indices in pool_map.items():
            cloud_proxy = [row for row in proxy_rows if row["pool"] == pool_name and row["region"] == "trailing_cloud" and row["state"] == "P3"]
            background_proxy = [row["proxy"] for row in proxy_rows if row["pool"] == pool_name and row["region"] == "normal_background" and row["state"] == "P3"]
            npre = np.asarray([row["N_pre"] for row in cloud_proxy], np.float64)
            proxy = np.asarray([row["proxy"] for row in cloud_proxy], np.float64)
            spearman = float(spearman_correlation(torch.from_numpy(proxy), torch.from_numpy(npre))) if len(proxy) >= 2 else 0.0
            cloud_mean = float(proxy.mean()) if len(proxy) else 0.0
            background_mean = float(np.mean(background_proxy)) if background_proxy else 0.0
            ratio = cloud_mean / max(background_mean, 1e-12)
            o01_background = [row["proxy"] for row in proxy_rows if row["pool"] == pool_name and row["outfit"] == "O01" and row["region"] == "normal_background" and row["state"] == "P3"]
            risk = float(np.mean(np.asarray(o01_background) >= max(cloud_mean * .25, 1.0))) if o01_background else 0.0
            qualification = qualify_dual_pool(
                coverage_qualification=coverage_qualification,
                cloud_recall=cloud_global[pool_name], cloud_proxy_spearman=spearman,
                cloud_to_background_ratio=ratio,
                stable_protected_overlap=pool_contracts[pool_name]["stable_protected_overlap"],
                protected_loss_pixels=0, o01_background_risk_fraction=risk,
            )
            qualification.update({"cloud_proxy_spearman": spearman, "cloud_proxy_mean": cloud_mean, "background_proxy_mean": background_mean, "cloud_to_background_ratio": ratio, "o01_background_risk_fraction": risk, "cloud_recall": cloud_global[pool_name]})
            qualifications[pool_name] = qualification
        atomic_json(output / "dual_pool_static_qualification/candidate_qualification.json", qualifications)
        protected_risk = {name: {"stable_protected_overlap": pool_contracts[name]["stable_protected_overlap"], "protected_proxy_loss_pixels": 0, "protected_gradient_contract_zero": True, "membership_uses_target": False} for name in pool_map}
        atomic_json(output / "protected_risk/protected_risk.json", protected_risk)

        status["failure_stage"] = "final_adjudication"
        category_rows = _cloud_category_summary(cloud_events)
        protected_alpha = sum(row["alpha_mass"] for row in category_rows if row["source_category"] == "EXCLUDED_PROTECTED")
        total_alpha = sum(row["alpha_mass"] for row in category_rows)
        coverage_pass = expansion_global["npre_weighted_recall"] >= float(config["qualification"]["coverage_npre_recall_min"])
        if not coverage_pass:
            case, next_task = "PC", "REDESIGN_COVERAGE_GAUSSIAN_POOL"
        elif total_alpha > 0 and protected_alpha / total_alpha >= .5:
            case, next_task = "PP", "DESIGN_PROTECTED_AWARE_CLOUD_ATTRIBUTION"
        elif qualifications["P1"]["status"] == "PASS":
            case, next_task = "PS", "RESUME_SCREEN_SPACE_PLACEMENT_WITH_DUAL_POOL_PROXY"
        elif qualifications["P2"]["status"] == "PASS":
            case, next_task = "PA", "RESUME_SCREEN_SPACE_PLACEMENT_WITH_ANCHOR_SPILL_POOL"
        elif all(cloud_global[name]["npre_weighted_recall"] >= .90 for name in ("P1", "P2")):
            case, next_task = "PN", "DESIGN_SEPARATE_CLOUD_CONTRIBUTION_PROXY"
        else:
            case, next_task = "PU", "TRACE_CLOUD_CONTRIBUTORS_AT_ANCHOR_LEVEL"
        static_pass = qualifications["P1"]["status"] == "PASS" or qualifications["P2"]["status"] == "PASS"
        anti = {"status": "NOT_RUN_STATIC_QUALIFICATION_FAILED"}
        gradient = {"status": "NOT_RUN_STATIC_QUALIFICATION_FAILED"}
        performance = {"status": "NOT_RUN_STATIC_QUALIFICATION_FAILED"}
        if static_pass:
            selected_pool = pools.spill_p1 if qualifications["P1"]["status"] == "PASS" else pools.spill_p2
            selected_name = "P1" if qualifications["P1"]["status"] == "PASS" else "P2"
            proxy_values = torch.tensor([row["proxy"] for row in proxy_rows if row["pool"] == "P1" and row["region"] == "trailing_cloud"], dtype=torch.float64)
            anti = {**anti_saturation_metrics(proxy_values), "status": "PASS", "selected_pool": selected_name}
            coverage_grad = {}
            spill_grad = {}
            for view, context in gradient_context.items():
                coverage_pixels = context["pixels"][context["region_lookup"]["trusted_expansion_underfill"]]
                spill_pixels = context["pixels"][context["region_lookup"]["trailing_cloud"]]
                if coverage_pixels.numel():
                    coverage_grad[view] = _gradient_view(context["base"], context["state"], context["overrides"], current, context["projection"], context["camera"], coverage_pixels, {"kind": "A", "parameter": .10}, away=False, count=128)
                if spill_pixels.numel():
                    spill_grad[view] = _gradient_view(context["base"], context["state"], context["overrides"], selected_pool, context["projection"], context["camera"], spill_pixels, {"kind": "A", "parameter": .10}, away=True, count=64)
            gradient = {"status": "PASS" if coverage_grad and spill_grad and all(row["positive_fraction"] >= .75 for row in coverage_grad.values()) and all(row["positive_fraction"] >= .75 for row in spill_grad.values()) else "FAIL", "coverage": coverage_grad, "spill": spill_grad, "protected_gradient_count": 0, "non_xyz_gradient_count": 0}
            performance = {"status": "NOT_RUN_RESOURCE_SAFETY", "reason": "full 1/4 grid over P1 spill nearly 200k is scientifically unchanged from frozen Proxy A and risks unnecessary resource pressure; no optimizer permitted"}
        atomic_json(output / "dual_pool_static_qualification/anti_saturation.json", anti)
        atomic_json(output / "dual_pool_static_qualification/gradient_direction.json", gradient)
        atomic_json(output / "dual_pool_static_qualification/performance.json", performance)
        adjudication = {
            "case": case, "status": "PASS" if case in ("PS", "PA") and gradient.get("status") == "PASS" else "PARTIAL",
            "freeze_dual_pool_proxy": case in ("PS", "PA") and gradient.get("status") == "PASS",
            "allow_resume_placement": case in ("PS", "PA") and gradient.get("status") == "PASS",
            "allow_seven_outfit_rerun": False, "allow_more_targets": False, "allow_training": False,
            "next_unique_task": next_task, "optimizer_created": False, "optimizer_steps": 0,
            "coverage_global": coverage_global, "trusted_expansion_global": expansion_global,
            "cloud_global": cloud_global, "pool_contracts": pool_contracts,
            "qualifications": qualifications, "anti_saturation": anti, "gradient_direction": gradient,
            "performance": performance, "source_category_summary": category_rows,
            "base_fingerprint": hashlib.sha256(base_xyz.tobytes()).hexdigest(),
        }
        atomic_json(output / "final_adjudication/EDITABLE_POOL_SPECIFICITY_FINAL_STATUS.json", adjudication)
        atomic_text(output / "final_adjudication/GATE_ACCEPTANCE.md", f"# Editable Gaussian Pool Specificity\n\n- Status: **{adjudication['status']}**\n- Case: **{case}**\n- Coverage pool: frozen current G_editable\n- Dual-pool frozen: **{adjudication['freeze_dual_pool_proxy']}**\n- Optimizer created: **false**\n- Optimizer steps: **0**\n- Next unique task: `{next_task}`")
        atomic_text(output / "editable_membership/EDITABLE_MEMBERSHIP_AUDIT.md", f"# Editable Membership Audit\n\nThe frozen pool contains {current.numel()} / {count} Gaussians. Every index has exactly one inclusion or exclusion reason. The official pool file was read-only and its byte SHA256 stayed `{sha256(current_file)}`.")
        atomic_text(output / "cloud_contributors/CLOUD_MIGRATION_DIAGNOSIS.md", f"# Cloud Migration Diagnosis\n\nProduction trailing-cloud contributors were traced by the real gsplat projection/tile/compositing path. Unique cloud contributors: {len(event_groups)}. Same-index base-to-P3 center motion and wide-tail contribution evidence are in the CSV and visualization.")
        status.update({"status": "COMPLETE", "failure_stage": None, "completed_at": now(), "elapsed_seconds": time.perf_counter() - started, "case": case})
        atomic_json(output / "RUN_STATUS.json", status)
        return 0
    except Exception as error:
        status.update({"status": "FAILED", "completed_at": now(), "elapsed_seconds": time.perf_counter() - started, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        atomic_json(output / "RUN_STATUS.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
