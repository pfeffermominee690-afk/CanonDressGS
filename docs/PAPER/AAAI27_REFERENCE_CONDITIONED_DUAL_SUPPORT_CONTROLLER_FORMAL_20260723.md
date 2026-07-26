# Reference-Conditioned Dual-Support Controller: Formal Results

Task: `AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002`

Classification: `REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_PARTIAL`

PAPER_FINAL: `0`

## Governance and frozen contract

- Source: `research/dual-support-controller-training-contract-repair-20260723` at `1d4b416929617d09bf65122ff5d6dbf23dfe564a`.
- Run branch: `research/reference-conditioned-dual-support-controller-formal-20260723`.
- Protocol / manifest / cycle / schedule SHA256: `44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49` / `a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3` / `f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77` / `63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf`.
- The earlier pre-result stop record is preserved. It documents training=0, backward=0, optimizer=0, checkpoint=0, render=0, metric=0, visual-review=0, and PAPER_FINAL=0 for the rejected incomplete contract.
- Formal attempts are append-only. Attempts 001--003 preserve RNG-context and cross-process-baseline failures; attempt 004 is the valid paired same-process evaluation/render.
- No result-driven retraining, extra steps, threshold change, temperature tuning, seed selection, pair selection, or scientific rerun occurred.

## Environment gate

Cloud preflight passed on RTX 4090, Python 3.10.20, PyTorch 2.4.1+cu121, CUDA 12.1, driver 580.76.05, gsplat 1.5.3+pt24cu121, and PyTorch3D 0.7.8. CUDA tensor, frozen-F2 forward, Controller forward, endpoint-bank load, Dual-Support renderer, LPIPS, mask/silhouette, scheduled-batch, and repaired-schedule-hash smokes passed before training.

## Training audit

The fixed 320-record protocol scope was used, with all 80 consistent duplicates retained and no exposure of the 20 formal-pure evaluation records. All seeds used the same frozen data-order hash and distinct seeded fresh-process initialization.

- Training / forward-batch / backward / optimizer / scheduler: `900 / 900 / 900 / 900 / 900`.
- Checkpoint writes: `18` (steps 50, 100, 150, 200, 250, and final 300 for each seed).
- Each seed completed exactly 300 steps and uses only `final_step_300.pt`; there was no best-checkpoint selection.
- Formal-pure record overlap is 0, while reference-asset overlap is 20/20 logical inputs.

## Pure endpoint evaluation

Each seed is 20/20 top-1, 100% SINGLE_ENDPOINT, endpoint-parity PASS, and 80/80 swap success. Across seeds this is 60/60 formal-pure cases and 240/240 swaps. Candidate-versus-teacher RGB MAE, LPIPS, protected LPIPS, and identity metric are exactly zero. Human review found only minor teacher-inherited edge scatter/silhouette roughness; Controller-added identity contamination maximum grade is 0.

## Closed-wardrobe protocol-fit results

| Seed | Dominant top-1 | Top-2 pair | AAB/ABB ordering | DUAL activation | Weight MAE | ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.931250 | 0.970833 | 0.908333 | 0.504167 | 0.184623 | 0.121745 |
| 1 | 0.915625 | 0.950000 | 0.887500 | 0.491667 | 0.189791 | 0.108786 |
| 2 | 0.921875 | 0.966667 | 0.895833 | 0.529167 | 0.188398 | 0.114222 |
| macro | 0.922917 | 0.962500 | 0.897222 | 0.508333 | 0.187604 | 0.114918 |

All 320 protocol records per seed and 960 total queries are retained. The parallel 260-unique-query summary is also retained rather than substituted for the protocol-weighted result: unique top-2=0.962500, ordering=0.897222, weight MAE=0.187604. Assignment-position and target-fold inconsistencies are preserved.

The top-2 pair and ordering gates pass. The activation gate fails decisively: macro activation is 0.508333, below 0.80, and every seed is below the 0.70 floor. SINGLE_ENDPOINT fallback therefore frequently produces a clean endpoint image by abandoning the requested mixture. This is a semantic failure, not a visual improvement.

## Controller-driven renders

Across the three seeds, mixed Controller-versus-oracle means are RGB MAE=0.052429, LPIPS=0.031219, silhouette IoU=0.713354, boundary F-score=0.354757, protected LPIPS=0.000987, and numeric identity metric=0.006668. Pure renders remain exact endpoint parity.

Historical FULL_LINEAR, HARD_GEOMETRY_SOFT_VA, and Oracle Dual-Support raw metrics are preserved with their original provenance. Those archived alpha-grid baselines use a different information boundary and are not silently treated as like-for-like Controller protocol-fit gates. Oracle Dual-Support is explicitly `ORACLE / NON-DEPLOYABLE`.

## Human visual review: 88/88

All 15 pure sheets, 60 mixed sheets, and 13 diagnostics were actually opened at original detail. The review preserves active-DUAL patch, cloud, mottle, edge scatter, silhouette discontinuity, and worst-pair full-surface contamination. It also records severe endpoint mismatch and wrong-garment mixture where fallback collapses the requested AAB/ABB composition. No severe double outline was observed; worst-seed grade-3 ghosting pair count is 0/10; identity-contamination maximum grade is 0.

The review does not credit fallback as artifact removal. It keeps both failure modes: over-fallback composition loss and residual artifacts when DUAL_SUPPORT does activate.

## Perturbation robustness

480 perturbation queries (20 representatives x 8 variants x 3 seeds) are complete. Assignment permutation is stable, but blur, mask morphology, reference dropout, and especially single-reference perturbations produce large top-1/pair instability, mode switches, and weight drift. No threshold or model was changed after observing these failures.

## Decision

`REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_PARTIAL` is the only defensible classification. Pure safety, top-2 pairing, ordering, leakage, ground-truth-information, geometry, identity, and frozen-asset gates pass; excessive fallback/low activation, weak calibration, perturbation fragility, and non-comparable visual-improvement gates prevent PASS. Partial value remains because pairing/ordering are strong and some activated dual-support outputs are useful.

Scientific scope is fixed identity, closed seen-garment wardrobe, protocol-fit mixed evaluation, record-isolated but reference-overlapping formal-pure evaluation, and view-transductive operation. This is not evidence of unseen-garment, unseen-reference, cross-identity, novel-view, novel-pose, or second-dataset generalization.

Final Git HEAD is resolved as the commit containing this report and is verified in the archive handoff/final chat; a report cannot self-contain its own Git object ID. PAPER_FINAL remains 0. The next task is `DIAGNOSE_CONTROLLER_FALLBACK_CALIBRATION_AND_RESIDUAL_GHOSTING` and was not started.
