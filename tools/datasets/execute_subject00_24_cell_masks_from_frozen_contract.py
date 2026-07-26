"""Execute the frozen Subject00 24-cell mask contract after explicit authorization.

This module is prepared by the preflight task but MUST NOT be executed by that
task.  It performs local-only SegFormer inference, emits exactly one person mask
and one garment mask per accepted raw, and never modifies an input image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "paper_protocol"
    / "reviewer_risk"
    / "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
)
EXECUTION_AUTHORIZATION = (
    "EXECUTE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT"
)
MODEL_REPOSITORY = "mattmdjaga/segformer_b2_clothes"
MODEL_REVISION = "584abc1e1d260e23c0fc627c5217a09b2b461046"
MODEL_PATH = Path(
    r"E:\data_pre\audit_subject02_layered_composite_v3"
    r"\mask_backend_closure_v3a\m"
)
MODEL_WEIGHTS_SHA256 = (
    "8f86fd90c567afd4370b3cc3a7e81ed767a632b2832a738331af660acc0c4c68"
)
MODEL_CONFIG_SHA256 = (
    "4b5127ca00fe61187b6cc6c232c9e19326ed228683f8f5c221790be9cc196a6e"
)
PREPROCESSOR_SHA256 = (
    "a608e3a47dcfba8dc052a766babb4b6c963285ab4f176bc6c1eb2b257fd3ad93"
)

BACKGROUND = 0
BAG_OR_HANDHELD = 16
PERSON_LABELS = tuple(range(1, 16)) + (17,)
GARMENT_LABELS = (4, 5, 6, 7, 8, 17)
GARMENT_PROBABILITY_THRESHOLD = 0.35
MASK_VALUES = (0, 255)
LABELS = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(array.astype(np.uint8), "L").save(
        temporary,
        format="PNG",
        compress_level=9,
    )
    temporary.replace(path)


def largest_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.reshape(-1))
    keep = np.flatnonzero(areas >= minimum)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def person_mask_from_labels(label_map: np.ndarray) -> np.ndarray:
    """Recover full person foreground while excluding background/held bags."""

    from scipy import ndimage

    raw = np.isin(label_map, PERSON_LABELS)
    labels, count = ndimage.label(raw)
    if count == 0:
        raise ValueError("person foreground is empty")
    areas = np.bincount(labels.reshape(-1))
    largest = int(np.argmax(areas[1:]) + 1)
    person = labels == largest
    person = ndimage.binary_dilation(person, iterations=1)
    holes = ndimage.binary_fill_holes(person) & ~person
    hole_labels, hole_count = ndimage.label(holes)
    if hole_count:
        hole_areas = np.bincount(hole_labels.reshape(-1))
        maximum = max(64, int(person.sum() * 0.001))
        keep = np.flatnonzero(hole_areas <= maximum)
        keep = keep[keep != 0]
        person |= np.isin(hole_labels, keep)
    return person.astype(bool)


def garment_mask_from_probabilities(
    probabilities: np.ndarray,
    label_map: np.ndarray,
    person: np.ndarray,
) -> np.ndarray:
    """Apply frozen Subject02 Strategy C and protected-label exclusions."""

    import cv2
    from scipy import ndimage

    probability = probabilities[list(GARMENT_LABELS)].sum(axis=0)
    garment = (probability >= GARMENT_PROBABILITY_THRESHOLD) & person
    kernel = np.ones((3, 3), dtype=np.uint8)
    garment = cv2.morphologyEx(
        garment.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    )
    garment = cv2.morphologyEx(
        garment, cv2.MORPH_OPEN, kernel, iterations=1
    )
    garment = largest_components(
        garment > 0,
        max(24, int(person.sum() * 0.0005)),
    )
    protected = ~np.isin(label_map, GARMENT_LABELS)
    garment &= person & ~protected
    holes = ndimage.binary_fill_holes(garment) & ~garment
    hole_labels, hole_count = ndimage.label(holes)
    if hole_count:
        hole_areas = np.bincount(hole_labels.reshape(-1))
        maximum = max(64, int(person.sum() * 0.001))
        keep = np.flatnonzero(hole_areas <= maximum)
        keep = keep[keep != 0]
        garment |= np.isin(hole_labels, keep) & person & ~protected
    return garment.astype(bool)


def mask_metrics(mask: np.ndarray) -> dict[str, Any]:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    areas = np.bincount(labels.reshape(-1))[1:]
    ys, xs = np.nonzero(mask)
    bbox = None
    bbox_coverage = 0.0
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        bbox_coverage = (
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / mask.size
        )
    boundary = mask & ~ndimage.binary_erosion(mask)
    return {
        "nonzero_pixels": int(mask.sum()),
        "area_ratio": float(mask.mean()),
        "connected_components": int(count),
        "largest_component_ratio": float(
            (areas.max() if len(areas) else 0) / max(mask.sum(), 1)
        ),
        "bbox_xyxy": bbox,
        "bbox_coverage": float(bbox_coverage),
        "boundary_to_area_ratio": float(boundary.sum() / max(mask.sum(), 1)),
    }


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    value = rgb.astype(np.float32).copy()
    value[mask] = 0.55 * value[mask] + 0.45 * np.asarray(color)
    return Image.fromarray(np.clip(value, 0, 255).astype(np.uint8), "RGB")


def save_review_assets(
    rgb: np.ndarray,
    person: np.ndarray,
    garment: np.ndarray,
    paths: dict[str, str],
) -> None:
    person_page = overlay(rgb, person, (30, 180, 255))
    garment_page = overlay(rgb, garment, (40, 230, 90))
    pair = rgb.astype(np.float32).copy()
    pair[person] = 0.70 * pair[person] + 0.30 * np.asarray((30, 180, 255))
    pair[garment] = 0.45 * pair[garment] + 0.55 * np.asarray((40, 230, 90))
    for image, key in (
        (person_page, "person_review_path"),
        (garment_page, "garment_review_path"),
        (Image.fromarray(np.clip(pair, 0, 255).astype(np.uint8), "RGB"), "pair_review_path"),
    ):
        destination = Path(paths[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        image.save(temporary, format="PNG", compress_level=9)
        temporary.replace(destination)


def contact_sheet(paths: list[Path], destination: Path, columns: int = 4) -> None:
    """Create a deterministic review sheet from already-generated pair pages."""

    if not paths or not all(path.is_file() for path in paths):
        return
    cell_width, cell_height = 360, 320
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            panel = opened.convert("RGB")
            panel.thumbnail(
                (cell_width - 8, cell_height - 8),
                Image.Resampling.LANCZOS,
            )
        x = (index % columns) * cell_width + (cell_width - panel.width) // 2
        y = (index // columns) * cell_height + (cell_height - panel.height) // 2
        canvas.paste(panel, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    canvas.save(temporary, format="PNG", compress_level=9)
    temporary.replace(destination)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "subject00.mask_execution_state.v1", "records": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def completed_record_is_intact(state: dict[str, Any], record: dict[str, Any]) -> bool:
    previous = state["records"].get(record["request_id"], {})
    if previous.get("status") != "MASK_PAIR_GENERATED_QA_PENDING_HUMAN":
        return False
    for kind in ("person", "garment"):
        path = Path(record[f"{kind}_mask_output_path"])
        if not path.is_file() or sha256(path) != previous[f"{kind}_mask_sha256"]:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.authorization != EXECUTION_AUTHORIZATION:
        raise PermissionError("exact mask-generation authorization is required")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    if manifest["segmentation_method_id"] != (
        "SUBJECT02_SEGFORMER_V3A_STRATEGY_C_SUBJECT00_SPECIALIZATION_V1"
    ):
        raise ValueError("segmentation method changed")
    if manifest["model"]["revision"] != MODEL_REVISION:
        raise ValueError("model revision changed")
    if sha256(MODEL_PATH / "model.safetensors") != MODEL_WEIGHTS_SHA256:
        raise ValueError("model weights changed")
    if sha256(MODEL_PATH / "config.json") != MODEL_CONFIG_SHA256:
        raise ValueError("model config changed")
    if sha256(MODEL_PATH / "preprocessor_config.json") != PREPROCESSOR_SHA256:
        raise ValueError("model preprocessor changed")
    if sha256(Path(manifest["accepted_registry_path"])) != manifest[
        "accepted_registry_sha256"
    ]:
        raise ValueError("accepted registry changed")

    attempt_root = Path(manifest["attempt_root"])
    state_path = attempt_root / "08_logs" / "execution_state.json"
    if attempt_root.exists() and not args.resume:
        raise FileExistsError("attempt root already exists; overwrite is forbidden")
    if not attempt_root.exists():
        for relative in manifest["directory_layout"]:
            (attempt_root / relative).mkdir(parents=True, exist_ok=False)
    state = load_state(state_path)

    import torch
    import torch.nn.functional as functional
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen CUDA execution requested but CUDA is unavailable")
    processor = SegformerImageProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    ).to(device)
    model.eval()
    actual_labels = {
        int(key): value for key, value in model.config.id2label.items()
    }
    if actual_labels != LABELS:
        raise ValueError("model label taxonomy changed")

    for record in manifest["records"]:
        request_id = record["request_id"]
        if completed_record_is_intact(state, record):
            continue
        if request_id in state["records"]:
            # No automatic retry of either a failed or tampered cell.
            continue
        raw_path = Path(record["accepted_raw_path"])
        try:
            if raw_path.stat().st_size != record["accepted_raw_bytes"]:
                raise ValueError("accepted raw byte count changed")
            if sha256(raw_path) != record["accepted_raw_sha256"]:
                raise ValueError("accepted raw SHA256 changed")
            with Image.open(raw_path) as opened:
                image = opened.convert("RGB")
                rgb = np.asarray(image)
            height, width = rgb.shape[:2]
            if [width, height] != [
                record["native_resolution"]["width"],
                record["native_resolution"]["height"],
            ]:
                raise ValueError("accepted raw native resolution changed")

            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                logits = model(
                    **{key: value.to(device) for key, value in inputs.items()}
                ).logits[0]
                upsampled = functional.interpolate(
                    logits[None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                probabilities = torch.softmax(upsampled.float(), dim=0).cpu().numpy()
            label_map = probabilities.argmax(axis=0).astype(np.uint8)
            person = person_mask_from_labels(label_map)
            garment = garment_mask_from_probabilities(
                probabilities,
                label_map,
                person,
            )
            if not person.any() or not garment.any():
                raise ValueError("empty person or garment mask")
            outside = int((garment & ~person).sum())
            if outside:
                raise ValueError("garment mask is not a subset of person mask")
            label_pixels = {
                str(label_id): int((label_map == label_id).sum())
                for label_id in LABELS
            }

            person_path = Path(record["person_mask_output_path"])
            garment_path = Path(record["garment_mask_output_path"])
            atomic_png(person_path, person.astype(np.uint8) * 255)
            atomic_png(garment_path, garment.astype(np.uint8) * 255)
            save_review_assets(rgb, person, garment, record["review_assets"])
            qa = {
                "schema_version": "subject00.mask_pair_qa.v1",
                "request_id": request_id,
                "raw_sha256": record["accepted_raw_sha256"],
                "shape_hw": [height, width],
                "mask_value_set": list(MASK_VALUES),
                "person": mask_metrics(person),
                "garment": mask_metrics(garment),
                "garment_outside_person_pixel_count": outside,
                "garment_person_overlap_ratio": float(
                    (garment & person).sum() / max(garment.sum(), 1)
                ),
                "protected_region_area": int((person & ~garment).sum()),
                "empty_protected_region": bool(not (person & ~garment).any()),
                "semantic_label_pixels": label_pixels,
                "coverage_proxies": {
                    "head_hair_face_pixels": (
                        label_pixels["1"]
                        + label_pixels["2"]
                        + label_pixels["3"]
                        + label_pixels["11"]
                    ),
                    "torso_garment_pixels": (
                        label_pixels["4"]
                        + label_pixels["7"]
                        + label_pixels["8"]
                        + label_pixels["17"]
                    ),
                    "hands_arm_proxy_pixels": label_pixels["14"] + label_pixels["15"],
                    "feet_leg_proxy_pixels": label_pixels["12"] + label_pixels["13"],
                    "shoe_pixels": label_pixels["9"] + label_pixels["10"],
                    "bag_or_handheld_pixels_excluded": label_pixels["16"],
                },
                "automatic_status": "PASS_STRUCTURAL_QA_PENDING_STATISTICAL_AND_HUMAN_REVIEW",
                "person_mask_human_decision": None,
                "garment_mask_human_decision": None,
                "mask_pair_human_decision": None,
                "mask_accepted": False,
            }
            atomic_json(Path(record["qa_json_path"]), qa)
            response = {
                "schema_version": "subject00.segformer_response.v1",
                "request_id": request_id,
                "model_repository": MODEL_REPOSITORY,
                "model_revision": MODEL_REVISION,
                "garment_probability_threshold": GARMENT_PROBABILITY_THRESHOLD,
                "person_labels": list(PERSON_LABELS),
                "garment_labels": list(GARMENT_LABELS),
                "label_min": int(label_map.min()),
                "label_max": int(label_map.max()),
                "finite": bool(np.isfinite(probabilities).all()),
                "no_candidate_masks_saved": True,
            }
            atomic_json(Path(record["model_response_json_path"]), response)
            state["records"][request_id] = {
                "status": "MASK_PAIR_GENERATED_QA_PENDING_HUMAN",
                "person_mask_sha256": sha256(person_path),
                "garment_mask_sha256": sha256(garment_path),
                "retry_count": 0,
            }
        except Exception as error:  # Preserve failure evidence; never auto-retry.
            state["records"][request_id] = {
                "status": "TECHNICAL_FAILURE_NO_RETRY",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "retry_count": 0,
            }
        atomic_json(state_path, state)

    review_root = attempt_root / "07_review_assets"
    for garment in ("O01", "O03", "O04"):
        paths = [
            Path(record["review_assets"]["pair_review_path"])
            for record in manifest["records"]
            if record["garment"] == garment
        ]
        contact_sheet(paths, review_root / f"{garment}_8view_contact_sheet.png")
    contact_sheet(
        [
            Path(record["review_assets"]["pair_review_path"])
            for record in manifest["records"]
            if record["accepted_with_limitation"]
        ],
        review_root / "high_risk_cells.png",
        columns=3,
    )
    contact_sheet(
        [
            Path(record["review_assets"]["pair_review_path"])
            for record in manifest["records"]
            if record["human_override_status"] is not None
        ],
        review_root / "registration_override_cells.png",
        columns=2,
    )
    cross_view = {}
    for garment in ("O01", "O03", "O04"):
        rows = []
        for record in manifest["records"]:
            if record["garment"] != garment:
                continue
            qa_path = Path(record["qa_json_path"])
            if qa_path.is_file():
                qa = json.loads(qa_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "request_id": record["request_id"],
                        "slot": record["slot"],
                        "direction": record["direction"],
                        "person_area_ratio": qa["person"]["area_ratio"],
                        "garment_area_ratio": qa["garment"]["area_ratio"],
                        "structural_status": qa["automatic_status"],
                    }
                )
            else:
                rows.append(
                    {
                        "request_id": record["request_id"],
                        "slot": record["slot"],
                        "direction": record["direction"],
                        "structural_status": "MISSING_DUE_TO_TECHNICAL_FAILURE",
                    }
                )
        cross_view[garment] = {
            "expected_count": 8,
            "record_count": len(rows),
            "records": rows,
            "human_decision": None,
        }
    atomic_json(
        attempt_root / "06_quality_audit" / "cross_view_qa.json",
        {
            "schema_version": "subject00.cross_view_mask_qa.v1",
            "garments": cross_view,
            "automatic_status": "RECORDED_PENDING_HUMAN_REVIEW",
        },
    )
    atomic_json(
        review_root / "mask_decision_matrix.json",
        {
            "schema_version": "subject00.mask_decision_matrix.v1",
            "records": [
                {
                    "request_id": record["request_id"],
                    "garment": record["garment"],
                    "slot": record["slot"],
                    "person_mask_human_decision": None,
                    "garment_mask_human_decision": None,
                    "mask_pair_human_decision": None,
                    "mask_accepted": False,
                }
                for record in manifest["records"]
            ],
        },
    )
    atomic_json(
        attempt_root / "09_final_registry" / "mask_pair_registry_pending_human.json",
        {
            "schema_version": "subject00.mask_pair_registry.pending_human.v1",
            "task_id": manifest["task_id"],
            "records": state["records"],
            "person_mask_human_decision": None,
            "garment_mask_human_decision": None,
            "mask_pair_human_decision": None,
            "mask_accepted": False,
            "teacher_target_created": False,
        },
    )


if __name__ == "__main__":
    main()
