"""Seal Controller V2 design-smoke evidence into repository artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-COMPATIBILITY-GATED-CONTROLLER-V2-DESIGN-001"
CLASSIFICATION = "CONTROLLER_V2_DESIGN_READY"
NEXT_TASK = (
    "TRAIN_AND_EVALUATE_COMPATIBILITY_GATED_CONTROLLER_V2_CROSSFIT_MICRO_PILOT"
)
REPORT_PATHS = (
    "docs/PAPER/AAAI27_COMPATIBILITY_GATED_CONTROLLER_V2_DESIGN_20260723.md",
    "docs/PAPER/AAAI27_TRI_MODE_GARMENT_COMPOSITION_20260723.md",
    "docs/PAPER/AAAI27_CONTROLLER_V2_CROSSFIT_PROTOCOL_20260723.md",
)
MACHINE_PATHS = (
    "paper_protocol/reviewer_risk/controller_v2_design_protocol.yaml",
    "paper_protocol/reviewer_risk/controller_v2_crossfit_splits.json",
    "paper_protocol/reviewer_risk/controller_v2_compatibility_manifests.json",
    "paper_protocol/reviewer_risk/controller_v2_forward_boundary.json",
    "paper_protocol/reviewer_risk/controller_v2_optimizer_training_plan.json",
    "paper_protocol/reviewer_risk/controller_v2_evaluator_contract.json",
    "paper_protocol/reviewer_risk/controller_v2_dry_run_summary.json",
)
HANDOFF_PATH = "project_control_handoff/controller_v2_design_handoff.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_once_text(path: Path, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"append-only collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_once_json(path: Path, value: Any) -> None:
    write_once_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def exact_descriptor_archive(smoke_archive: dict[str, Any]) -> dict[str, Any]:
    archive = copy.deepcopy(smoke_archive)
    archive["schema_version"] = (
        "canondressgs.research.controller_v2_compatibility_manifests.v2"
    )
    archive["required_descriptor_names_materialized"] = True
    for manifest in archive["manifests"]:
        manifest["smoke_manifest_content_sha256"] = manifest.pop(
            "manifest_content_sha256"
        )
        for entry in manifest["entries"]:
            raw = entry["intrinsic_and_calibration_descriptors"]
            evidence = entry["rule_evidence"]
            entry["compatibility_descriptors"] = {
                "canonical_geometry_support_overlap": raw[
                    "canonical_support_overlap_proxy_mean_silhouette_iou"
                ],
                "symmetric_nearest_support_distance": raw[
                    "symmetric_nearest_support_distance_proxy_mean_displacement_rms"
                ],
                "opacity_weighted_support_overlap": raw[
                    "opacity_weighted_support_overlap_proxy_one_minus_outside_opacity"
                ],
                "garment_coverage_ratio": raw[
                    "garment_support_coverage_ratio_active_over_total"
                ],
                "canonical_bbox_ratio": raw["canonical_bbox_ratio"],
                "canonical_bbox_ratio_status": raw["canonical_bbox_ratio_status"],
                "silhouette_overlap_on_calibration_conditions": raw[
                    "canonical_support_overlap_proxy_mean_silhouette_iou"
                ],
                "oracle_dual_support_core_artifact_maximum_grade": evidence[
                    "maximum_core_artifact_grade"
                ],
                "oracle_severe_patch_cloud_mottle_full_body_count": evidence[
                    "severe_patch_cloud_mottle_full_body_count"
                ],
                "identity_contamination_maximum_grade": evidence[
                    "identity_contamination_maximum_grade"
                ],
                "grade3_ghosting_count": evidence["grade3_ghosting_count"],
                "boundary_consistency": raw["boundary_consistency_mean_fscore"],
            }
            entry["descriptor_measurement_contract"] = {
                "canonical_geometry_support_overlap": (
                    "calibration-fold frozen Oracle silhouette-IoU proxy; no "
                    "target/test query is used"
                ),
                "symmetric_nearest_support_distance": (
                    "calibration-fold frozen Oracle mean displacement RMS proxy"
                ),
                "opacity_weighted_support_overlap": (
                    "one minus calibration-fold frozen Oracle mean "
                    "outside-garment opacity"
                ),
                "garment_coverage_ratio": (
                    "calibration-fold active Gaussian count divided by total count"
                ),
                "canonical_bbox_ratio": (
                    "not present in the frozen archive; null is retained rather "
                    "than fabricating a value"
                ),
                "boundary_consistency": (
                    "calibration-fold frozen Oracle mean boundary F-score"
                ),
            }
            entry.pop("entry_content_sha256")
            entry["entry_content_sha256"] = canonical_sha256(entry)
        manifest["manifest_content_sha256"] = canonical_sha256(manifest)
    archive["archive_content_sha256"] = canonical_sha256(
        {key: value for key, value in archive.items() if key != "archive_content_sha256"}
    )
    return archive


def main_report(summary: dict[str, Any], manifests: dict[str, Any]) -> str:
    counts = summary["execution_counts"]
    label_rows = "\n".join(
        f"| {row['rotation']} | {row['compatible']} | {row['incompatible']} | "
        "O01_O03, O02_O03 |"
        for row in summary["compatibility_label_counts_per_rotation"]
    )
    return f"""# Compatibility-Gated Controller V2 Design

**RESEARCH METHOD DESIGN — NOT PAPER FINAL**

- Task: `{TASK_ID}`
- Final design classification: `{CLASSIFICATION}`
- Source: `research/controller-calibration-compatibility-diagnostic-20260723` at
  `25318857b4cc4ee11011cd852c5cfc237a8b3cee`
- Formal V1 ancestor: `f45a518f330fb407942055756373e83f65717853`
- Design branch: `research/compatibility-gated-controller-v2-design-20260723`
- Scope: closed seen-garment wardrobe only.

## Inherited causal evidence

The inherited formal diagnosis is `MULTIPLE_FACTORS`. The primary residual
source is `DUAL_SUPPORT_COMPATIBILITY_DOMINANT`; secondary sources are weight
calibration, fallback calibration, and reference robustness. Pair selection is
high-accuracy and is not the primary failure source.

V1 evidence is preserved without alteration: top-2 pair accuracy 0.962500,
ordering accuracy 0.897222, mixed Dual-Support activation 0.508333, weight MAE
0.187604, weight RMSE 0.212702, pure top-1 20/20 per seed, pure
`SINGLE_ENDPOINT` 100%, identity contamination 0, target-forward leakage 0,
and GT pair/weight inference use 0.

The causal gains remain: pair-fix mean LPIPS 0.000350 (31.81% improved),
weight-fix 0.021878 (99.71%), fallback-removal 0.018765 (55.97%), and
full-oracle 0.041626. `SIMPLE_GATE_NOT_SEPARABLE` is retained: the illustrative
0.50 top-2-mass / 0.05 secondary-weight gate reaches pure SINGLE 1.0 and mixed
DUAL 0.819444 but wrong-pair DUAL 0.666667.

The design does not erase the archived scientific failures. O01_O03 and
O02_O03 both have pair accuracy 1.0 yet retain exact-pair/exact-weight grade-3
contamination and archived Oracle severe count 8. These failures motivate the
compatibility gate; they are not treated as classifier errors, insufficient
training, or tuned away.

## V2 factorization

Frozen F2 spatial maps are reduced by masked mean/max into 256-dimensional
reference rows. Deterministic validity-aware set mean/max aggregation produces
a 512-dimensional vector, followed by LayerNorm and L2 normalization. Three
independent linear heads then emit:

- five garment logits for top-1, unordered top-2 pair, and dominant ordering;
- one mixedness logit for pure versus mixed;
- ten pair-conditioned weight logits in frozen unordered-pair order.

The random-init adapter has 9,232 parameters: 1,024 normalization parameters,
2,565 garment-head parameters, 513 mixedness-head parameters, and 5,130
pair-weight-head parameters. Every forward computes all ten weights. Inference
selects a scalar only with the predicted pair; the V1 probability ratio is
forbidden as the final mixture weight.

## Compatibility prior

The prior is a garment-bank property for a closed wardrobe. It uses only
canonical/intrinsic proxies and frozen Oracle records from the rotation's
calibration fold. Test-fold data, current-query GT, target pose/camera/RGB/mask,
current-query renders, and pair blacklists are excluded. Missing canonical bbox
ratios remain explicit nulls with `UNAVAILABLE...NOT_FABRICATED` provenance.

The fixed uniform rule requires maximum core grade <=2, zero severe
patch/cloud/mottle/full-body records, identity grade 0, zero grade-3 ghosting,
and endpoint parity PASS.

| Rotation | Compatible | Incompatible | Rule-generated incompatible pairs |
|---:|---:|---:|---|
{label_rows}

All four rotations retain their calibration-fold records and SHA-256 values
even though the observed labels are the same. Explicit pair blacklist count is
0. The repository archive contains {manifests["total_pair_records"]} pair
records.

## Cross-fit and routing

The frozen scientific name is **CONDITION-FOLD CROSS-FIT EVALUATION**, not
novel-view evaluation. Each of four rotations uses two train folds, one
calibration fold, and one held-out test fold. Thresholds and compatibility are
calibration-only; all held-out folds must be aggregated without best-rotation,
best-seed, or best-checkpoint selection.

Pair confidence is uniquely defined as the top-2 versus top-3 probability
margin. Calibration grids are mixedness 0.10–0.90 in steps of 0.10 and pair
confidence {{0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}}.

The three reachable runtime modes are `SINGLE_ENDPOINT`, `DUAL_SUPPORT`, and
`HARD_GEOMETRY_SOFT_VA`. The safe mixed fallback uses one dominant predicted
geometry support and two visibility/appearance sources; it never interpolates
geometry and never silently substitutes a pure top-1 image.

## No-training dry-run

The smoke exercised five pure, three AAB, three ABB, and seven boundary cases
(compatible, incompatible, low mixedness, low pair confidence, dropout,
single-reference, and exact tie). All three modes were reached. Seed 0 was
identical across two fresh processes, while seeds 0/1/2 had distinct parameter
hashes.

No formal image or metric was produced. Counts are: training
{counts["training_steps"]}, training-forward batches
{counts.get("training_forward_batches", 0)}, backward {counts["backward_calls"]}, optimizer
creation {counts["optimizer_creations"]}, optimizer step
{counts["optimizer_steps"]}, scheduler {counts["scheduler_steps"]}, checkpoint
load/write {counts["checkpoint_loads"]}/{counts["checkpoint_writes"]}, renderer
{counts["renderer_calls"]}, formal renders/metrics/reviews
{counts["formal_renders"]}/{counts["formal_metrics"]}/{counts["formal_visual_reviews"]},
and PAPER_FINAL {counts["paper_final"]}.

## Scientific boundary

This design authorizes no empirical V2 quality claim. A future passing pilot
may claim only compatibility-aware routing within a closed seen-garment
wardrobe. It may not claim unseen-pair compatibility, unseen garment
generation, arbitrary synthesis, cross-identity generalization, novel-view
success, or novel-pose success.
"""


def tri_mode_report() -> str:
    return """# Tri-Mode Garment Composition Contract

**RESEARCH METHOD DESIGN — NOT PAPER FINAL**

## Decision order

The controller receives only reference-derived frozen-F2 features and validity.
It predicts garment probabilities, mixedness, and all ten pair weights. Stable
ranking uses frozen outfit order for exact ties. The unordered top-2 prediction
indexes both the compatibility prior and the pair-specific weight; GT pair and
GT weight are absent from prediction forward.

Routing priority is:

1. fewer than two valid references → `SINGLE_ENDPOINT`;
2. mixedness below the calibration-frozen threshold → `SINGLE_ENDPOINT`;
3. top2-vs-top3 margin below threshold → `SINGLE_ENDPOINT`;
4. compatible predicted pair → `DUAL_SUPPORT`;
5. otherwise → `HARD_GEOMETRY_SOFT_VA`.

## SINGLE_ENDPOINT

This mode constructs one predicted-top-1 geometry support and one
visibility/appearance source. A second Gaussian branch is not constructed.
Pure references, dropout, single-reference, low mixedness, and low pair
confidence use this safe path.

## DUAL_SUPPORT

This mode constructs two immutable endpoint geometry supports for the predicted
pair. Opacity weights come from the selected predicted-pair weight head.
Geometry interpolation, geometry averaging, and basis-coefficient interpolation
are all false.

## HARD_GEOMETRY_SOFT_VA

This compatibility-safe mixed fallback constructs exactly one geometry support:
the predicted dominant garment endpoint. It still composes visibility and
appearance from both predicted endpoints using the pair-conditioned weight.
It therefore preserves a mixed output contract without interpolating geometry,
forcing Dual-Support, or silently returning a pure top-1 image.

The adapter has no pair-specific rendering branch. O01_O03 and O02_O03 reach
this mode only because their calibration manifests fail the same uniform rule
used for every pair.

## Information boundary

Target pose and camera are renderer inputs only after the controller decision.
They are not controller inputs. GT garment ID, pair, composition, alpha,
target RGB/mask, teacher residual, target render, and visual artifact label are
forbidden. The audited target-forward leakage count is 0.

## Interface-only evidence

The no-render smoke reached all three modes and verified geometry-support
counts of 1/2/1 and visibility/appearance source counts of 1/2/2. Endpoint
paths and the frozen pair orientation are retained in every runtime request.
No formal render, metric, or visual review was executed.
"""


def crossfit_report() -> str:
    return """# Controller V2 Condition-Fold Cross-Fit Protocol

**RESEARCH METHOD DESIGN — NOT PAPER FINAL**

The condition order is fixed as `cond_000000`, `cond_000318`, `cond_000017`,
`cond_000347`. These are condition folds, not strict novel views.

| Rotation | Train folds | Calibration fold | Held-out test fold |
|---:|---|---:|---:|
| 0 | 0, 1 | 2 | 3 |
| 1 | 1, 2 | 3 | 0 |
| 2 | 2, 3 | 0 | 1 |
| 3 | 3, 0 | 1 | 2 |

Train folds are reserved for future V2 fitting. The calibration fold alone
selects mixedness and pair-confidence thresholds and builds the compatibility
prior. The held-out test fold is excluded from fitting, threshold choice, and
compatibility construction. Final reporting must aggregate all four held-out
folds; best-rotation selection is forbidden.

## Frozen calibration objectives

Threshold selection is lexicographic: minimize pure false-mixed rate, minimize
wrong-pair DUAL rate, then maximize mixed safe-mode coverage. Render quality on
the test fold cannot tune thresholds. Pair confidence has one definition:
top2-vs-top3 probability margin.

## Future training contract

Each rotation/seed combination uses a fresh independent process and random
initialization for seeds 0, 1, and 2. V1, B6, and Ours-v2 checkpoints are not
initializers. F2 remains frozen. No best seed, rotation, or checkpoint may be
selected.

The preregistered loss is soft-target garment CE + mixedness BCE +
mixed-only pair-specific SmoothL1 + nuisance-only consistency. Training
supervises only the GT-pair scalar for mixed records while forward always emits
all ten weights. Blur, mild mask erosion/dilation, and assignment permutation
are consistency nuisances. Dropout and single-reference are information
ablations whose target is safe SINGLE fallback, not reconstruction of an
unseen second garment.

## Future evaluator contract and gates

Every rotation and seed reports pair accuracy/ordering/confusion; mixedness
AUROC and errors; weight MAE/RMSE by pair/composition/rotation; three mode
rates and wrong/incompatible-pair DUAL; visual metrics and artifact grades; and
mode-specific efficiency.

Preregistered pilot gates include macro pair accuracy >=90% and every rotation
>=80%; pure SINGLE >=95% and false-mixed <=5%; compatible-mixed DUAL >=80%;
incompatible-mixed HARD >=80%; incompatible-pair and wrong-pair DUAL <=10%;
correct-pair weight MAE <=0.12; severe artifacts at least 50% below V1;
no Dual-Support grade-3 contamination for O01_O03/O02_O03; identity grade 0;
grade-3 ghosting in <=1/10 pairs; and dropout/single-reference safe SINGLE
>=95%.

This task performs no training and produces PAPER_FINAL=0.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    smoke_root = args.smoke_root.resolve()
    audit_root = smoke_root / "audits"

    summary = read_json(audit_root / "design_smoke_summary.json")
    if summary["status"] != "READY":
        raise RuntimeError(f"smoke is not READY: {summary['status']}")
    if any(summary["execution_counts"].values()):
        raise RuntimeError("no-training gate is not zero")
    if not summary["frozen_assets_unchanged"]:
        raise RuntimeError("frozen asset audit failed")

    manifests = exact_descriptor_archive(
        read_json(audit_root / "controller_v2_compatibility_manifests.json")
    )
    machine_sources = {
        "controller_v2_compatibility_manifests.json": manifests,
        "controller_v2_forward_boundary.json": read_json(
            audit_root / "controller_v2_forward_boundary.json"
        ),
        "controller_v2_optimizer_training_plan.json": read_json(
            audit_root / "controller_v2_optimizer_training_plan.json"
        ),
        "controller_v2_evaluator_contract.json": read_json(
            audit_root / "controller_v2_evaluator_contract.json"
        ),
        "controller_v2_dry_run_summary.json": read_json(
            audit_root / "controller_v2_dry_run_summary.json"
        ),
    }
    for value in machine_sources.values():
        if "execution_counts" in value:
            value["execution_counts"].setdefault("training_forward_batches", 0)
    manifests.pop("archive_content_sha256")
    manifests["archive_content_sha256"] = canonical_sha256(manifests)
    risk_root = repo_root / "paper_protocol/reviewer_risk"
    for name, value in machine_sources.items():
        write_once_json(risk_root / name, value)

    write_once_text(repo_root / REPORT_PATHS[0], main_report(summary, manifests))
    write_once_text(repo_root / REPORT_PATHS[1], tri_mode_report())
    write_once_text(repo_root / REPORT_PATHS[2], crossfit_report())

    required = list(REPORT_PATHS) + list(MACHINE_PATHS) + [HANDOFF_PATH]
    handoff = {
        "schema_version": "canondressgs.research.controller_v2_design_handoff.v1",
        "task_id": TASK_ID,
        "status": "PASS",
        "label": "RESEARCH METHOD DESIGN - NOT PAPER FINAL",
        "final_classification": CLASSIFICATION,
        "source_head": "25318857b4cc4ee11011cd852c5cfc237a8b3cee",
        "run_branch": "research/compatibility-gated-controller-v2-design-20260723",
        "smoke_status": summary["status"],
        "architecture_contract": "PASS",
        "three_mode_runtime": "PASS",
        "crossfit_protocol": "PASS",
        "compatibility_manifests": "PASS",
        "forward_boundary": "PASS",
        "dry_run": "PASS",
        "no_training_gate": "PASS",
        "formal_v1_and_diagnostic_archives_unchanged": True,
        "compatibility_manifest_archive_sha256": manifests[
            "archive_content_sha256"
        ],
        "required_artifact_count": len(required),
        "required_artifacts": required,
        "paper_final": 0,
        "next_task_started": False,
        "next_task": NEXT_TASK,
        "containing_git_commit_semantics": (
            "The final archive HEAD is the commit containing this handoff; "
            "the 40-character value is verified after commit and push."
        ),
    }
    write_once_json(repo_root / HANDOFF_PATH, handoff)

    missing = [relative for relative in required if not (repo_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing sealed artifacts: {missing}")


if __name__ == "__main__":
    main()
