from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene.full_dressable_dataset import (
    DUAL_TARGET_FIELDS,
    FORBIDDEN_INFERENCE_FIELDS,
    FullDressableInferenceDataset,
    FullDressableTrainingDataset,
    select_forward_conditioning_fields,
)


DUAL_RGB_FIELDS = ("target_edit_rgb", "target_base_rgb")
DUAL_MASK_FIELDS = tuple(sorted(
    (DUAL_TARGET_FIELDS - set(DUAL_RGB_FIELDS))
    | {"target_foreground_mask", "target_clothing_mask"}
))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CanonDressGS full dataset v1")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-count", type=int, default=2)
    parser.add_argument("--projection-json", help="actual posed-anchor projections: condition_id -> [[u,v],...]")
    parser.add_argument("--real-model-smoke-command", nargs=argparse.REMAINDER)
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    path = Path(args.manifest).resolve(); root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    official = not args.regression
    outfits, conditions = manifest["outfits"], manifest["conditions"]
    fixture = bool(manifest.get("fixture_mode", False))
    expected_outfits = 12 if official else len(manifest.get("expected_outfits", outfits))
    expected_conditions = 200 if official else int(manifest.get("expected_condition_count", len(conditions)))
    supervision_mode = manifest.get("supervision_mode", "single_target")
    condition_ids = [x["condition_id"] for x in conditions]
    checks: dict[str, object] = {}
    checks["outfit_count"] = len(outfits); checks["condition_count"] = len(conditions)
    top_level_duplicates = len(condition_ids) - len(set(condition_ids))
    sets = []
    outside_ratios = []
    dual_contract_failures = []
    identity_failures = []
    missing_by_outfit = {}
    unknown_by_outfit = {}
    duplicate_observations = 0
    for outfit in outfits:
        observations = outfit["observations"]
        ids = [x["condition_id"] for x in observations]; sets.append(set(ids))
        missing = sorted(set(condition_ids) - set(ids))
        unknown = sorted(set(ids) - set(condition_ids))
        if missing:
            missing_by_outfit[outfit["outfit_id"]] = missing
        if unknown:
            unknown_by_outfit[outfit["outfit_id"]] = unknown
        duplicate_observations += len(ids) - len(set(ids))
        if len(ids) != expected_conditions or len(set(ids)) != len(ids): checks["missing_or_duplicate"] = 1
        for obs in observations:
            images = {}
            for key, mode in (("rgb", "RGB"), ("foreground_mask", "L"), ("clothing_mask", "L")):
                p = Path(obs[key]); p = p if p.is_absolute() else root / p
                with Image.open(p) as image:
                    if image.mode != mode: raise ValueError(f"{p}: expected mode {mode}, got {image.mode}")
                    images[key] = np.asarray(image)
            cid = obs["condition_id"]; condition = conditions[condition_ids.index(cid)]
            expected_size = (int(condition["width"]), int(condition["height"]))
            if any((arr.shape[1], arr.shape[0]) != expected_size for arr in images.values()): raise ValueError(f"size mismatch {outfit['outfit_id']}/{cid}")
            fg, cloth = images["foreground_mask"] >= 128, images["clothing_mask"] >= 128
            outside_ratios.append(float(np.logical_and(cloth, ~fg).sum() / max(cloth.sum(), 1)))
            if obs.get("identity_audit_status") == "FAIL":
                identity_failures.append(f"{outfit['outfit_id']}/{cid}")
            if supervision_mode == "dual_target_region_aware_v1":
                missing = [name for name in (*DUAL_RGB_FIELDS, *DUAL_MASK_FIELDS) if name not in obs]
                if missing:
                    dual_contract_failures.append({"sample": f"{outfit['outfit_id']}/{cid}", "missing": missing})
                    continue
                dual_images = {}
                for key in DUAL_RGB_FIELDS:
                    p = Path(obs[key]); p = p if p.is_absolute() else root / p
                    with Image.open(p) as image:
                        if image.mode != "RGB": raise ValueError(f"{p}: expected RGB")
                        dual_images[key] = np.asarray(image)
                dual_masks = {}
                for key in DUAL_MASK_FIELDS:
                    p = Path(obs[key]); p = p if p.is_absolute() else root / p
                    with Image.open(p) as image:
                        if image.mode != "L": raise ValueError(f"{p}: expected L")
                        dual_masks[key] = np.asarray(image) >= 128
                values = [*dual_images.values(), *dual_masks.values()]
                if any((value.shape[1], value.shape[0]) != expected_size for value in values):
                    dual_contract_failures.append({"sample": f"{outfit['outfit_id']}/{cid}", "reason": "size_mismatch"})
                    continue
                edit = dual_masks["target_edit_mask"]
                core = dual_masks["target_edit_core_mask"]
                preserve = dual_masks["target_preserve_mask"]
                transition = dual_masks["target_transition_mask"]
                protected = dual_masks["target_protected_mask"]
                dual_checks = {
                    "protected_outside_preserve": int((protected & ~preserve).sum()),
                    "edit_protected_overlap": int((edit & protected).sum()),
                    "edit_preserve_overlap": int((edit & preserve).sum()),
                    "edit_preserve_uncovered": int((~(edit | preserve)).sum()),
                    "core_outside_edit": int((core & ~edit).sum()),
                    "transition_outside_edit": int((transition & ~edit).sum()),
                    "foreground_alias_mismatch": int(np.logical_xor(dual_masks["target_foreground_mask"], fg).sum()),
                    "clothing_alias_mismatch": int(np.logical_xor(dual_masks["target_clothing_mask"], cloth).sum()),
                }
                if any(dual_checks.values()):
                    dual_contract_failures.append({"sample": f"{outfit['outfit_id']}/{cid}", "checks": dual_checks})
    checks["shared_condition_ids"] = all(value == set(condition_ids) for value in sets)
    checks["missing_conditions_by_outfit"] = missing_by_outfit
    checks["unknown_conditions_by_outfit"] = unknown_by_outfit
    checks["duplicate_observations"] = duplicate_observations
    checks["missing_or_duplicate"] = (
        abs(len(outfits) - expected_outfits)
        + abs(len(conditions) - expected_conditions)
        + top_level_duplicates
        + sum(len(value) for value in missing_by_outfit.values())
        + sum(len(value) for value in unknown_by_outfit.values())
        + duplicate_observations
    )
    checks["clothing_mask_outside_foreground_max"] = max(outside_ratios, default=0.0)
    checks["supervision_mode"] = supervision_mode
    checks["dual_target_contract_failures"] = dual_contract_failures
    checks["raw_identity_failures"] = identity_failures
    excluded_identity_failures = manifest.get("identity_visual_adjudication", {}).get("excluded_samples", [])
    checks["excluded_identity_failures"] = excluded_identity_failures
    finite = True; k_consistent = True
    for item in conditions:
        arrays = [np.asarray(item[key], dtype=float) for key in ("pose", "Rh_raw", "R_global", "Th", "K", "w2c", "c2w")]
        finite &= all(np.isfinite(x).all() for x in arrays)
        K = arrays[4]; w, h = item["width"], item["height"]
        k_consistent &= K.shape == (3, 3) and K[0, 0] > 0 and K[1, 1] > 0 and 0 <= K[0, 2] < w and 0 <= K[1, 2] < h and np.allclose(arrays[5] @ arrays[6], np.eye(4), atol=1e-4)
    checks["state_camera_finite"] = finite; checks["K_resolution_consistency"] = k_consistent
    split_values = [x for values in manifest["splits"].values() for x in values]
    checks["split_leakage"] = len(split_values) - len(set(split_values))
    split = next((key for key, value in manifest["splits"].items() if value), None)
    training = FullDressableTrainingDataset(path, split, args.reference_count)
    inference = FullDressableInferenceDataset(path, split, args.reference_count)
    train_sample, infer_sample = training[0], inference[0]
    checks["target_reference_overlap"] = int(train_sample["target_condition_id"] in train_sample["reference_condition_ids"])
    checks["inference_forbidden_fields"] = len(FORBIDDEN_INFERENCE_FIELDS.intersection(infer_sample))
    forward_fields = select_forward_conditioning_fields(train_sample)
    checks["dual_target_fields_in_forward"] = len(DUAL_TARGET_FIELDS.intersection(forward_fields))
    checks["loader_shapes"] = {key: list(value.shape) for key, value in train_sample.items() if isinstance(value, torch.Tensor)}
    if args.projection_json:
        projections = json.loads(Path(args.projection_json).read_text(encoding="utf-8"))
        hits = []
        for outfit in outfits:
            for obs in outfit["observations"]:
                cid = obs["condition_id"]; xy = np.asarray(projections[cid], dtype=float)
                p = Path(obs["foreground_mask"]); p = p if p.is_absolute() else root / p
                mask = np.asarray(Image.open(p).convert("L")) >= 128
                u, v = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
                valid = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
                hits.append(float((valid & mask[np.clip(v, 0, mask.shape[0]-1), np.clip(u, 0, mask.shape[1]-1)]).mean()))
        checks["projection_hit_median"] = float(np.median(hits)); checks["projection_hit_min"] = min(hits)
    else:
        checks["projection_evidence"] = "MISSING"
    if args.real_model_smoke_command:
        result = subprocess.run(args.real_model_smoke_command, check=False)
        checks["real_model_smoke_returncode"] = result.returncode
    else:
        checks["real_model_smoke"] = "MISSING"
    passed = (checks["missing_or_duplicate"] == 0 and checks["shared_condition_ids"] and checks["clothing_mask_outside_foreground_max"] <= .005 and checks["state_camera_finite"] and checks["K_resolution_consistency"] and checks["split_leakage"] == 0 and checks["target_reference_overlap"] == 0 and checks["inference_forbidden_fields"] == 0 and checks["dual_target_fields_in_forward"] == 0 and not dual_contract_failures and not identity_failures and not excluded_identity_failures)
    if official:
        passed &= checks.get("projection_hit_median", 0) >= .90 and checks.get("projection_hit_min", 0) >= .80 and checks.get("real_model_smoke_returncode", 1) == 0
    report = {"status": "PASS" if passed else "FAIL", "official": official, "fixture": fixture, "checks": checks}
    if args.report: Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
