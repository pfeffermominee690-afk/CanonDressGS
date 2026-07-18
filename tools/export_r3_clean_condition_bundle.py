from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


SCHEMA = "canondressgs.r3_clean_condition_bundle.v1"
EXPECTED_IDS = (
    "cond_000000", "cond_000104", "cond_000318", "cond_000714",
    "cond_000017", "cond_000101", "cond_000347", "cond_000379",
    "cond_000108", "cond_000113", "cond_000598", "cond_000439",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = value / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _pose_165(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    jaw = np.zeros(3, dtype=np.float32)
    result = np.concatenate((
        np.asarray(archive["global_orient"], dtype=np.float32).reshape(3),
        np.asarray(archive["body_pose"], dtype=np.float32).reshape(63),
        jaw, np.zeros(6, dtype=np.float32),
        np.asarray(archive["left_hand_pose"], dtype=np.float32).reshape(45),
        np.asarray(archive["right_hand_pose"], dtype=np.float32).reshape(45),
    ))
    if result.shape != (165,) or not np.isfinite(result).all():
        raise ValueError("condition pose must be finite [165]")
    return result


def _condition_records(value: object) -> list[dict]:
    if not isinstance(value, dict):
        raise ValueError("V3 contract must be an object")
    for key in ("conditions", "records", "selected_conditions"):
        records = value.get(key)
        if isinstance(records, list) and records:
            return records
    raise ValueError("V3 contract has no condition record list")


def export_bundle(selection_path: Path, contract_path: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite condition bundle: {output}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    selected_ids = tuple(selection.get("selected_condition_ids", ()))
    if selected_ids != EXPECTED_IDS:
        raise ValueError(f"V3-ready condition order changed: {selected_ids}")
    selected_by_id = {row["condition_id"]: row for row in selection["selected_conditions"]}
    contract_by_id = {row["condition_id"]: row for row in _condition_records(contract)}
    if set(selected_by_id) != set(EXPECTED_IDS) or not set(EXPECTED_IDS).issubset(contract_by_id):
        raise ValueError("V3 selection and condition contract do not resolve the exact 12 conditions")
    (output / "images").mkdir(parents=True)
    (output / "masks").mkdir()
    records = []
    for condition_id in EXPECTED_IDS:
        selected = selected_by_id[condition_id]
        source = contract_by_id[condition_id]
        if not bool(selected.get("pose_camera_traceable")):
            raise ValueError(f"condition is not pose/camera traceable: {condition_id}")
        image_path = Path(source["clay_condition_path"])
        mask_path = Path(source["clay_foreground_mask_path"])
        pose_path = Path(source["subject02_pose_Rh_Th_path"])
        camera_path = Path(source["camera_metadata_path"])
        for path in (image_path, mask_path, pose_path, camera_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(pose_path, allow_pickle=False) as archive:
            pose = _pose_165(archive)
            rh_raw = np.asarray(archive["global_orient"], dtype=np.float64).reshape(3)
            th_raw = np.asarray(archive["transl"], dtype=np.float64).reshape(3)
            th_rendered = np.asarray(archive["transl_rendered"], dtype=np.float64).reshape(3)
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        if camera.get("cond_id") != condition_id or camera.get("camera_id") != selected["source_camera"]:
            raise ValueError(f"camera identity mismatch for {condition_id}")
        image_target = output / "images" / f"{condition_id}.png"
        mask_target = output / "masks" / f"{condition_id}.png"
        shutil.copyfile(image_path, image_target); shutil.copyfile(mask_path, mask_target)
        record = {
            "condition_id": condition_id,
            "view": selected["view"],
            "selection_role": selected["selection_role"],
            "source_frame": int(selected["source_frame"]),
            "source_camera": selected["source_camera"],
            "pose_index": int(selected["pose_index"]),
            "pose_fingerprint": selected["pose_fingerprint"],
            "pose": pose.tolist(),
            "Rh_raw": rh_raw.tolist(),
            "R_global": _axis_angle_to_matrix(rh_raw).tolist(),
            "Th_raw": th_raw.tolist(),
            "Th_rendered": th_rendered.tolist(),
            "camera": {
                "R": camera["R"], "T": camera["T"], "fov": camera["fov"],
                "width": camera["width"], "height": camera["height"],
                "view_name": camera["view_name"], "renderer": camera["renderer"],
            },
            "image": f"images/{condition_id}.png",
            "foreground_mask": f"masks/{condition_id}.png",
            "source_sha256": {
                "image": _sha256(image_path), "foreground_mask": _sha256(mask_path),
                "pose_archive": _sha256(pose_path), "camera_metadata": _sha256(camera_path),
            },
            "bundle_sha256": {"image": _sha256(image_target), "foreground_mask": _sha256(mask_target)},
            "conventions": {
                "pose": "global(3)+body(63)+jaw_zero(3)+eyes_zero(6)+left_hand(45)+right_hand(45)",
                "global_transform": "R_global from source global_orient; Th_rendered used for rendered condition",
                "camera": "unaltered PyTorch3D FoVPerspectiveCameras R/T/FOV; OpenCV conversion and deterministic mask-bbox fit occur in the audited cloud runner",
            },
        }
        records.append(record)
    manifest = {
        "schema_version": SCHEMA,
        "source_selection": {"path": str(selection_path), "sha256": _sha256(selection_path)},
        "source_condition_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
        "condition_count": len(records),
        "view_distribution": selection["view_distribution"],
        "conditions": records,
    }
    manifest_path = output / "condition_protocol.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the frozen V3-ready 12-condition R3-CLEAN bundle")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--condition-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_bundle(args.selection.resolve(), args.condition_contract.resolve(), args.output.resolve())
    print(result)


if __name__ == "__main__":
    main()
