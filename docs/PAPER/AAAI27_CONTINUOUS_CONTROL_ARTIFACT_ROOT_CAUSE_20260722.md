# AAAI27 Continuous-Control Artifact Root Cause

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

- Source: `371d812864614cd561e33edfe3f6c043b38415d2` on the sealed P0 evaluation branch.
- Protocol SHA-256: `4d5abd9c368b40e9e622573b43ceb9ba15cf2774a0907be5d798be2fc41db98a`.
- FULL interpolation was reused read-only: 440/440; no FULL render was regenerated.
- New channel-group renders: 720/720.
- Manual visual review: 82/82 sheets opened.
- PRIMARY_FAILURE_SOURCE: **MULTIPLE_FACTORS**.
- SECONDARY_FAILURE_SOURCES: `["CROSS_CHANNEL_COUPLING"]`.
- NEXT_TASK: **RUN_GEOMETRY_APPEARANCE_DISENTANGLED_BASIS_MICRO_PILOT** (not started).

## Repaired no-training gate

- Previous attempt: `attempt_001`, permanently classified `FAILED_GATE_FALSE_POSITIVE_LEGACY_OPTIMIZER_OBJECT`; none of its renders or metrics were reused.
- Additional failed attempt: `attempt_002`, permanently classified `FAILED_GATE_FALSE_POSITIVE_PRE_FREEZE_PARAMETER_BASELINE`; its 720 renders and metrics were also retained but not reused. The completed diagnosis is `attempt_003`.
- Diagnostic optimizer created: `False`; step count: `0`.
- Legacy context optimizer created: `True` (3 runtime contexts); step count: `0`; discarded: `True`.
- Backward: `0`; checkpoint writes: `0`; frozen parameter changes: `0`.

## Channel attribution

| variant | FULL reproduction pairs | NONE/MINOR pairs |
|---|---|---|
| XYZ_ONLY | 10 | 0 |
| SCALE_ROT_ONLY | 10 | 0 |
| OPACITY_ONLY | 10 | 0 |
| SH_ONLY | 10 | 0 |
| GEOMETRY_ALL | 10 | 0 |
| APPEARANCE_ALL | 10 | 0 |

At alpha=0.5 every preregistered variant equals the same six-channel midpoint because all selected and non-selected channels use weight 0.5. Midpoint reproduction counts are therefore descriptive but cannot identify a unique channel cause.

## Stable versus unstable preregistered correlations

| feature | stable median | unstable median | rho | exact p | strong |
|---|---|---|---|---|---|
| top_10_support_overlap | 0.2901 | 0.27845 | 0.1899 | 1.0000 | False |
| top_20_support_overlap | 0.42485 | 0.4118 | 0.2659 | 0.7333 | False |
| weighted_jaccard | 0.52698 | 0.51487 | 0.0380 | 0.6000 | False |
| exclusive_fraction_sum | 0.00057031 | 0.00024442 | 0.2659 | 0.2000 | False |
| overlap_residual_cosine | 0.27877 | 0.21414 | 0.1899 | 0.6000 | False |
| xyz_direction_conflict | 0.30756 | 0.33235 | -0.1140 | 1.0000 | False |
| opacity_activation_conflict_fraction | 0.42294 | 0.45247 | -0.1140 | 0.5333 | False |
| sh_disagreement | 0.64473 | 0.64324 | -0.0380 | 1.0000 | False |
| teacher_silhouette_iou | 0.90143 | 0.88032 | 0.4938 | 0.1667 | False |
| boundary_distance | 6.6574 | 7.2004 | -0.2659 | 0.4667 | False |

Strong preregistered metrics: 0/10; exclusive-support visual alignments: 0/10.

## Per-Gaussian role conflict

| pair | magnitude median | magnitude p95 | xyz conflict | opacity conflict | SH0 conflict |
|---|---|---|---|---|---|
| O01_O02 | 1.4303 | 4.3165 | 0.3076 | 0.4229 | 0.3392 |
| O01_O03 | 1.4861 | 5.0452 | 0.4024 | 0.4824 | 0.5501 |
| O01_O04 | 1.5870 | 5.6657 | 0.3940 | 0.5072 | 0.5106 |
| O01_O08 | 1.4548 | 4.8507 | 0.3065 | 0.4588 | 0.3343 |
| O02_O03 | 1.4124 | 4.6522 | 0.3847 | 0.4627 | 0.5701 |
| O02_O04 | 1.4629 | 4.5938 | 0.3726 | 0.4525 | 0.5307 |
| O02_O08 | 1.3825 | 4.0499 | 0.2841 | 0.4219 | 0.3566 |
| O03_O04 | 1.3705 | 4.0134 | 0.2486 | 0.3460 | 0.2512 |
| O03_O08 | 1.4591 | 4.5591 | 0.3323 | 0.4392 | 0.4476 |
| O04_O08 | 1.3693 | 4.0635 | 0.2922 | 0.3506 | 0.2965 |

Every pair was quantified in the frozen Gaussian index order. The only frozen regional partition available was protected versus non-protected membership; sleeves/trousers labels were not invented (`REGION_MAPPING_UNAVAILABLE`).

## Scientific boundary

The sealed source findings remain part of the evidence: Ours-v2 continuous outputs retain patch, mottle, cloud, edge scatter, and silhouette discontinuity; B6/B7 switch only among discrete endpoints; M3/M4/B4 retain their severe full-body contamination records; the sealed identity-contamination maximum remains 0. The prior 57/57 review is unchanged.

All observed patch, mottle, cloud, edge-scatter, full-body contamination, identity-contamination, and silhouette-discontinuity grades are retained item by item. No pair, channel, or failure was removed; no parameter, threshold, stable label, teacher, basis, or protocol was changed.

Training steps, backward calls, diagnostic optimizer creations/steps, legacy optimizer calls, scheduler steps, and checkpoint writes were all zero. Three legacy context Adam objects were constructed and discarded without zero_grad, step, scheduler, or state-save calls. PAPER_FINAL=0.
