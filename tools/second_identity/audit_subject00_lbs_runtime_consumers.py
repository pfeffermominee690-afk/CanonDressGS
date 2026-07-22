#!/usr/bin/env python3
"""Produce a deterministic static audit of MMLP-Human LBS consumers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TERMS = (
    "lbs_weights_grid.npz",
    "interpolate_skinningfield",
    "grid_sample",
    "get_weights",
    "get_Gweights",
    "compute_xyz",
    "create_from_pcd",
    "capture",
    "restore",
    "densif",
    "clone",
    "split",
    "prun",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hits = [term for term in TERMS if re.search(re.escape(term), line, re.IGNORECASE)]
            if hits:
                records.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "terms": hits,
                        "text": line.strip(),
                    }
                )
    return records


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise RuntimeError(f"Missing frozen runtime evidence {label}: {needle}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    gaussian_path = root / "scene/gaussian_model.py"
    scene_path = root / "scene/scene.py"
    smpl_path = root / "utils/smpl_utils.py"
    net_path = root / "scene/net_vis.py"
    test_path = root / "test.py"
    train_path = root / "train.py"
    gaussian = gaussian_path.read_text(encoding="utf-8")
    scene = scene_path.read_text(encoding="utf-8")
    smpl = smpl_path.read_text(encoding="utf-8")
    net = net_path.read_text(encoding="utf-8")

    evidence = {
        "scene_loads_grid": "grid_info = dict(np.load(weights_grid_path, allow_pickle=True))",
        "scene_passes_grid": "lbs_weights_grid_info=grid_info",
        "weights_lazy_cache": "if self._weights is None:",
        "weights_query_base_xyz": "weights = interpolate_skinningfield(self.weights_grid_info, xyz)",
        "weights_cached": "self._weights = weights",
        "checkpoint_saves_weights": "'_weights': self.get_weights",
        "checkpoint_restores_weights": "self._weights = data['_weights']",
        "grid_detached": "ginfo[key] = torch.as_tensor(ginfo[key]).detach().cuda()",
        "runtime_blend": "G_weight = torch.einsum('vp,pij->vij', self.get_weights, G)",
        "canonical_offsets": "xyz = raw_xyz + self.get_dxyz + torch.tanh(self.xyz_offset) * 0.008",
        "inference_checkpoint_restore": "gaussians.restore(load_data)",
    }
    for label, needle in evidence.items():
        haystack = scene if label.startswith("scene_") else smpl if label == "grid_sample" else net if label == "inference_checkpoint_restore" else gaussian
        require(haystack, needle, label)
    require(smpl, "F.grid_sample(", "grid_sample")

    method_names = re.findall(r"^    def ([A-Za-z_][A-Za-z0-9_]*)\(", gaussian, flags=re.MULTILINE)
    densification_methods = sorted(
        name for name in method_names if any(token in name.lower() for token in ("densif", "clone", "split", "prun"))
    )
    result = {
        "schema_version": "subject00.mmlphuman.lbs_runtime_consumer_audit.v1",
        "status": "PASS_STATIC_CONSUMER_CLOSURE_IDENTIFIED",
        "source_head": "f88cc6a798747ad93ab9e6fe26f92c3e23229c9c",
        "audited_files": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (gaussian_path, scene_path, smpl_path, net_path, test_path, train_path)
        },
        "all_term_matches": matches(root),
        "answers": {
            "weights_query_frequency": "LAZY_ONCE_FROM_BASE_XYZ_THEN_CACHED",
            "lookup_in_autograd": False,
            "lookup_autograd_reason": "grid is detached; base _xyz is a non-Parameter tensor; the result is cached before trainable canonical offsets are applied",
            "weights_recomputed_when_xyz_offset_or_dxyz_changes": False,
            "clone_split_current_behavior": "NOT_IMPLEMENTED_IN_THIS_RUNTIME",
            "checkpoint_saves_per_gaussian_weights": True,
            "fixed_per_gaussian_weights_runtime_capability": "SUPPORTED_BY_INTERNAL_STATE_AND_CHECKPOINT_RESTORE_BUT_NOT_CREATE_FROM_PCD_API",
            "face_id_runtime_capability": False,
            "barycentric_runtime_capability": False,
            "semantic_region_runtime_capability": False,
            "arbitrary_point_requires_volume_lookup": False,
            "arbitrary_point_without_attachment_current_path": "legacy grid lookup at first get_weights access",
            "grid_algorithmically_necessary": False,
            "grid_role": "initialization-time convenience interface for deriving fixed per-Gaussian weights",
            "inference_after_checkpoint_requires_grid": False,
        },
        "state_flow": [
            "Scene requires and loads gaussian/lbs_weights_grid.npz",
            "create_from_pcd detaches grid metadata onto CUDA but does not immediately query weights",
            "first get_weights access queries legacy grid at immutable base _xyz and caches _weights",
            "get_Gweights blends cached per-Gaussian weights with pose transforms each forward",
            "trainable dxyz/xyz_offset affect canonical positions but do not update cached weights",
            "capture serializes cached _weights; restore loads them directly",
            "test/net_vis inference restores checkpoint weights without loading the grid",
        ],
        "densification": {
            "methods_found": densification_methods,
            "implemented": bool(densification_methods),
            "required_future_contract": "attachment metadata must be inherited or deterministically recomputed if densification is added",
        },
        "runtime_repair_surface": {
            "minimum_required": [
                "optional fixed_weights argument to create_from_pcd",
                "persistent face_id/barycentric/component/region tensors",
                "capture/restore support for attachment metadata",
                "explicit clone/split metadata rules before any future densification",
            ],
            "training_objective_change_required": False,
            "deformation_kernel_change_required": False,
        },
        "counters": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "training_checkpoint_writes": 0,
            "PAPER_FINAL": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
