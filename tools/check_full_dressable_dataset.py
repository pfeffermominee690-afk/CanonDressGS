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
from scene.full_dressable_dataset import FORBIDDEN_INFERENCE_FIELDS, FullDressableInferenceDataset, FullDressableTrainingDataset


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
    expected_outfits, expected_conditions = (12 if official else len(outfits)), (200 if official else len(conditions))
    condition_ids = [x["condition_id"] for x in conditions]
    checks: dict[str, object] = {}
    checks["outfit_count"] = len(outfits); checks["condition_count"] = len(conditions)
    checks["missing_or_duplicate"] = int(len(outfits) != expected_outfits or len(conditions) != expected_conditions or len(set(condition_ids)) != len(condition_ids))
    sets = []
    outside_ratios = []
    for outfit in outfits:
        observations = outfit["observations"]
        ids = [x["condition_id"] for x in observations]; sets.append(set(ids))
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
    checks["shared_condition_ids"] = all(value == set(condition_ids) for value in sets)
    checks["clothing_mask_outside_foreground_max"] = max(outside_ratios, default=0.0)
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
    passed = (checks["missing_or_duplicate"] == 0 and checks["shared_condition_ids"] and checks["clothing_mask_outside_foreground_max"] <= .005 and checks["state_camera_finite"] and checks["K_resolution_consistency"] and checks["split_leakage"] == 0 and checks["target_reference_overlap"] == 0 and checks["inference_forbidden_fields"] == 0)
    if official:
        passed &= checks.get("projection_hit_median", 0) >= .90 and checks.get("projection_hit_min", 0) >= .80 and checks.get("real_model_smoke_returncode", 1) == 0
    report = {"status": "PASS" if passed else "FAIL", "official": official, "checks": checks}
    if args.report: Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
