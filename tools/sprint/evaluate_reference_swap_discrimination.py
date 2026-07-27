from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


PLAN_SCHEMA_VERSION = "canondressgs.reference_swap_plan.v1"
TARGET_IMAGE_FIELDS = frozenset({
    "target_rgb", "target_edit_rgb", "target_base_rgb", "target_foreground_mask",
    "target_clothing_mask", "target_protected_mask", "target_preserve_mask",
})


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _target_geometry(episode: Mapping[str, Any]) -> dict[str, Any]:
    target = episode["target"]
    return {
        "pose": target["pose"],
        "Rh": target["Rh"],
        "Th": target["Th"],
        "camera": target["camera"],
    }


def build_reference_swap_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    outfits = list(manifest["protocol"]["outfits"])
    conditions = list(manifest["protocol"]["conditions"])
    by_key = {
        (item["outfit_id"], item["target_condition_id"]): item
        for item in manifest["episodes"]
    }
    expected = {(outfit, condition) for outfit in outfits for condition in conditions}
    if set(by_key) != expected:
        raise ValueError("reference-swap evaluation requires a complete outfit/target grid")
    cases: list[dict[str, Any]] = []

    def add_case(
        kind: str,
        target_episode: Mapping[str, Any],
        reference_episode: Mapping[str, Any],
        reference_indices: list[int],
        suffix: str,
        dropped: str | None = None,
    ) -> None:
        references = [reference_episode["references"][index] for index in reference_indices]
        target_geometry = _target_geometry(target_episode)
        cases.append({
            "case_id": f"{target_episode['episode_id']}__{suffix}",
            "kind": kind,
            "target_episode_id": target_episode["episode_id"],
            "target_outfit_id": target_episode["outfit_id"],
            "target_condition_id": target_episode["target_condition_id"],
            "reference_source_outfit_id": reference_episode["outfit_id"],
            "reference_condition_ids": [item["condition_id"] for item in references],
            "references": references,
            "target_geometry": target_geometry,
            "target_pose_camera_fingerprint": _fingerprint(target_geometry),
            "dropped_reference_condition_id": dropped,
            "forward_contract": {
                "outfit_id_is_model_input": False,
                "target_images_are_model_input": False,
                "only_reference_fields_change_for_swap": True,
            },
        })

    for condition in conditions:
        for target_outfit in outfits:
            target_episode = by_key[(target_outfit, condition)]
            correct_indices = list(range(len(target_episode["references"])))
            add_case("correct_reference", target_episode, target_episode, correct_indices, "correct")
            if len(correct_indices) > 1:
                add_case(
                    "reference_permutation",
                    target_episode,
                    target_episode,
                    list(reversed(correct_indices)),
                    "permutation_reverse",
                )
            for index, reference in enumerate(target_episode["references"]):
                kept = [value for value in correct_indices if value != index]
                add_case(
                    "reference_dropout",
                    target_episode,
                    target_episode,
                    kept,
                    f"drop_{reference['condition_id']}",
                    dropped=reference["condition_id"],
                )
            for reference_outfit in outfits:
                if reference_outfit == target_outfit:
                    continue
                swapped_episode = by_key[(reference_outfit, condition)]
                add_case(
                    "swapped_reference",
                    target_episode,
                    swapped_episode,
                    list(range(len(swapped_episode["references"]))),
                    f"swapped_from_{reference_outfit}",
                )

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "outfits": outfits,
        "conditions": conditions,
        "cases": cases,
        "target_rgb_is_forward_input": False,
        "case_counts": {
            kind: sum(item["kind"] == kind for item in cases)
            for kind in ("correct_reference", "swapped_reference", "reference_permutation", "reference_dropout")
        },
    }
    validate_reference_swap_plan(plan)
    plan["content_sha256"] = _fingerprint(plan)
    return plan


def build_forward_payload(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields allowed to reach Ours; target images stay outside."""

    references = case["references"]
    geometry = case["target_geometry"]
    payload = {
        "reference_records": references,
        "target_pose": geometry["pose"],
        "target_Rh": geometry["Rh"],
        "target_Th": geometry["Th"],
        "target_camera": geometry["camera"],
    }
    leaked = TARGET_IMAGE_FIELDS.intersection(payload)
    if leaked or "outfit_id" in payload or "cloth_id" in payload:
        raise RuntimeError(f"forbidden fields entered reference-swap forward: {sorted(leaked)}")
    return payload


def validate_reference_swap_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(plan["cases"])
    correct = {
        (item["target_outfit_id"], item["target_condition_id"]): item
        for item in cases if item["kind"] == "correct_reference"
    }
    swaps = [item for item in cases if item["kind"] == "swapped_reference"]
    if not correct or not swaps:
        raise ValueError("reference-swap plan lacks correct or swapped cases")
    for case in cases:
        payload = build_forward_payload(case)
        if TARGET_IMAGE_FIELDS.intersection(payload):
            raise ValueError("target image field entered a forward payload")
        expected = correct[(case["target_outfit_id"], case["target_condition_id"])]
        if case["target_pose_camera_fingerprint"] != expected["target_pose_camera_fingerprint"]:
            raise ValueError("diagnostic changed target pose/camera")
    for case in swaps:
        expected = correct[(case["target_outfit_id"], case["target_condition_id"])]
        if case["reference_source_outfit_id"] == expected["reference_source_outfit_id"]:
            raise ValueError("swapped case did not change reference source")
        if case["target_geometry"] != expected["target_geometry"]:
            raise ValueError("reference swap changed more than references")
    return {
        "case_count": len(cases),
        "same_target_pose_camera": True,
        "swap_changes_only_references": True,
        "target_images_enter_forward": False,
    }


def _chw(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().cpu()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be HW/CHW/HWC")
    if tensor.shape[0] in {1, 3}:
        return tensor
    if tensor.shape[-1] in {1, 3}:
        return tensor.permute(2, 0, 1)
    raise ValueError(f"cannot infer channels for {name}")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    active = _chw(mask, "region_mask")[:1] >= 0.5
    if value.shape[-2:] != active.shape[-2:]:
        raise ValueError("region and prediction resolutions differ")
    active = active.expand(value.shape[0], -1, -1)
    return 0.0 if not active.any() else float(value[active].mean())


def target_error_metrics(
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    rgb = _chw(prediction["rgb"], "prediction rgb")
    alpha = _chw(prediction["alpha"], "prediction alpha")[:1]
    target_rgb = _chw(target["rgb"], "target rgb")
    target_alpha = _chw(target["foreground_mask"], "target foreground")[:1]
    if rgb.shape != target_rgb.shape or alpha.shape != target_alpha.shape:
        raise ValueError("prediction and target shapes differ")
    region_masks = {
        "garment_region": target["garment_mask"],
        "protected_region": target["protected_mask"],
        "full_foreground": target["foreground_mask"],
    }
    return {
        region: {
            "rgb": _masked_mean((rgb - target_rgb).abs(), mask),
            "alpha": _masked_mean((alpha - target_alpha).abs(), mask),
            "silhouette": _masked_mean(
                torch.logical_xor(alpha >= 0.5, target_alpha >= 0.5).float(), mask
            ),
        }
        for region, mask in region_masks.items()
    }


def prediction_distance(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    rgb_a, rgb_b = _chw(first["rgb"], "first rgb"), _chw(second["rgb"], "second rgb")
    alpha_a, alpha_b = _chw(first["alpha"], "first alpha")[:1], _chw(second["alpha"], "second alpha")[:1]
    masks = {
        "garment_region": target["garment_mask"],
        "protected_region": target["protected_mask"],
        "full_foreground": target["foreground_mask"],
    }
    return {
        region: {
            "rgb": _masked_mean((rgb_a - rgb_b).abs(), mask),
            "alpha": _masked_mean((alpha_a - alpha_b).abs(), mask),
            "silhouette": _masked_mean(
                torch.logical_xor(alpha_a >= 0.5, alpha_b >= 0.5).float(), mask
            ),
        }
        for region, mask in masks.items()
    }


def _subtract(second: Mapping[str, Any], first: Mapping[str, Any]) -> dict[str, Any]:
    return {
        region: {metric: float(second[region][metric] - first[region][metric]) for metric in first[region]}
        for region in first
    }


def score_reference_swap_plan(
    plan: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    targets: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    cases = {item["case_id"]: item for item in plan["cases"]}
    missing = sorted(set(cases).difference(predictions))
    if missing:
        raise ValueError(f"prediction bundle lacks cases: {missing[:5]}")
    by_target: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for case in cases.values():
        by_target.setdefault((case["target_outfit_id"], case["target_condition_id"]), []).append(case)
    rows: list[dict[str, Any]] = []
    for key, target_cases in sorted(by_target.items()):
        correct = next(item for item in target_cases if item["kind"] == "correct_reference")
        target = targets[correct["target_episode_id"]]
        correct_prediction = predictions[correct["case_id"]]
        correct_error = target_error_metrics(correct_prediction, target)
        for swapped in (item for item in target_cases if item["kind"] == "swapped_reference"):
            swapped_prediction = predictions[swapped["case_id"]]
            swapped_error = target_error_metrics(swapped_prediction, target)
            render_delta = prediction_distance(correct_prediction, swapped_prediction, target)
            rows.append({
                "target_outfit_id": key[0],
                "target_condition_id": key[1],
                "correct_case_id": correct["case_id"],
                "swapped_case_id": swapped["case_id"],
                "swapped_reference_outfit_id": swapped["reference_source_outfit_id"],
                "correct_target_error": correct_error,
                "swapped_target_error": swapped_error,
                "correct_vs_swapped_render_distance": render_delta,
                "reference_sensitivity": render_delta,
                "reference_swap_margin": _subtract(swapped_error, correct_error),
                "outfit_discrimination_margin": _subtract(swapped_error, correct_error),
            })
    diagnostics = []
    for case in cases.values():
        if case["kind"] not in {"reference_permutation", "reference_dropout"}:
            continue
        correct = next(
            item for item in by_target[(case["target_outfit_id"], case["target_condition_id"])]
            if item["kind"] == "correct_reference"
        )
        target = targets[correct["target_episode_id"]]
        diagnostics.append({
            "case_id": case["case_id"],
            "kind": case["kind"],
            "distance_from_correct": prediction_distance(
                predictions[correct["case_id"]], predictions[case["case_id"]], target
            ),
        })
    return {
        "schema_version": "canondressgs.reference_swap_metrics.v1",
        "definition": "error(swapped references, correct outfit target) - error(correct references, correct outfit target)",
        "positive_margin_means_correct_reference_is_better": True,
        "rows": rows,
        "permutation_and_dropout_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or score reference-swap discrimination without training.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-bundle", type=Path)
    parser.add_argument("--target-bundle", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = build_reference_swap_plan(manifest)
    if args.prediction_bundle is None and args.target_bundle is None:
        result: Mapping[str, Any] = plan
    elif args.prediction_bundle is not None and args.target_bundle is not None:
        predictions = torch.load(args.prediction_bundle, map_location="cpu", weights_only=False)
        targets = torch.load(args.target_bundle, map_location="cpu", weights_only=False)
        result = score_reference_swap_plan(plan, predictions, targets)
    else:
        raise ValueError("prediction and target bundles must be supplied together")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_counts": plan["case_counts"]}, indent=2))


if __name__ == "__main__":
    main()
