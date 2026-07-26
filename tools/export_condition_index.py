from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotmat_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a traced 200-condition CanonDressGS index")
    parser.add_argument("--poses-json", required=True)
    parser.add_argument("--cameras-json", required=True)
    parser.add_argument("--selection-json", required=True, help="exactly 200 {condition_id,frame_id,camera_id} records")
    parser.add_argument("--source-width", required=True, type=int)
    parser.add_argument("--source-height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--background", nargs=3, type=float, default=(1, 1, 1))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(args.poses_json).resolve(), Path(args.cameras_json).resolve(), Path(args.selection_json).resolve()]
    poses, cameras, selection = (json.loads(path.read_text(encoding="utf-8")) for path in paths)
    if len(selection) != 200 or len({x["condition_id"] for x in selection}) != 200:
        raise ValueError("selection must contain exactly 200 unique condition_id records")
    pose_map = {str(x["frame_id"]): x for x in poses}
    camera_map = {str(x.get("cam_id", x.get("camera_id"))): x for x in cameras}
    if len(pose_map) != len(poses) or len(camera_map) != len(cameras):
        raise ValueError("pose/camera source contains duplicate identifiers")
    sx, sy = args.width / args.source_width, args.height / args.source_height
    conditions = []
    for selected in selection:
        frame, camera_id = str(selected["frame_id"]), str(selected["camera_id"])
        if frame not in pose_map or camera_id not in camera_map:
            raise KeyError(f"unresolved source pair frame={frame}, camera={camera_id}")
        state, camera = pose_map[frame], camera_map[camera_id]
        pose = np.asarray(state["pose"], dtype=np.float64).reshape(-1)
        rotation = np.asarray(state["Rh"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(state["Th"], dtype=np.float64).reshape(3)
        K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
        w2c = np.asarray(camera["w2c"], dtype=np.float64).reshape(4, 4)
        if pose.shape != (165,) or not all(np.isfinite(x).all() for x in (pose, rotation, translation, K, w2c)):
            raise ValueError(f"invalid state/camera for {selected['condition_id']}")
        scaled_K = K.copy(); scaled_K[0, :] *= sx; scaled_K[1, :] *= sy
        conditions.append({
            "condition_id": selected["condition_id"], "source_frame_id": frame, "source_camera_id": camera_id,
            "pose": pose.tolist(), "Rh_raw": rotmat_to_rotvec(rotation).tolist(), "R_global": rotation.tolist(),
            "Th": translation.tolist(), "K": scaled_K.tolist(), "w2c": w2c.tolist(),
            "c2w": np.linalg.inv(w2c).tolist(), "width": args.width, "height": args.height,
            "background": list(args.background),
            "conventions": {"pose": "55 SMPL-X axis-angle joints; flattened length 165", "Rh_raw": "axis-angle converted from source R_global", "global_transform": "x_world=R_global@x_posed+Th", "camera": "x_cam=w2c[:3,:3]@x_world+w2c[:3,3]", "pixels": "u=fx*x/z+cx, v=fy*y/z+cy", "K_scaling": f"source {args.source_width}x{args.source_height} to {args.width}x{args.height}"},
            "source_checksum": {"poses_json_sha256": sha256(paths[0]), "cameras_json_sha256": sha256(paths[1]), "selection_json_sha256": sha256(paths[2])},
        })
    output = {"schema_version": "canondressgs.condition_index.v1", "conditions": conditions}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
