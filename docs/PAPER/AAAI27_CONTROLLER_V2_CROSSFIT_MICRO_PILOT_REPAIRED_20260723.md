# Compatibility-Gated Controller V2 Repaired Cross-Fit Micro-Pilot

## Scope and provenance

Task `AAAI27-CONTROLLER-V2-CROSSFIT-MICRO-PILOT-REPAIRED-001` executed the
repaired contract from source branch
`research/controller-v2-micro-pilot-contract-repair-20260723` at
`1a2059301e3a0ec7fa0f73591b69b7b428e986c4` on branch
`research/controller-v2-crossfit-micro-pilot-from-repaired-contract-20260723`.
The execution used `attempt_004`; no result-driven extension, threshold
change, pair exclusion, best-checkpoint selection, or scientific rerun was
performed.

The repaired protocol LF SHA-256 is
`44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49`,
the source-manifest LF SHA-256 is
`a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3`,
and the global contract SHA-256 is
`e5b720ccedfb4b0b39892011123010b6522ece7dd1d534032ec2e2e26f7863f7`.
All hashes matched before execution.

Historical `attempt_001` remains read-only: 4 files, 21,558 bytes, combined
content/metadata fingerprint
`95ea354e4da7af9dd50f30d6cfd9e2d99ba968dbe33283d32437440d85b38129`.
`attempt_002` stopped before a completed forward or training step because the
deterministic CuBLAS workspace variable was missing. `attempt_003` stopped
before backward and before a completed training step because the all-pure
first batch produced an empty mixed-only loss list. Both step-zero attempts
and their audits remain preserved; neither contributed scientific results.

The resource gate passed on an NVIDIA RTX 4090 (24,564 MiB total, 0 MiB used,
24,081 MiB free), with no competing GPU/heavy-I/O or subject00 task, more than
25 GiB free disk, and sufficient inodes.

## Frozen rotations and execution

Each of four rotations contains 160 train, 80 calibration, and 80 test
records. Three seeds were executed for each rotation and family. Frozen
cycle/schedule hashes were:

| Rotation | Cycle SHA-256 | Schedule SHA-256 |
|---:|---|---|
| 0 | `e32dc5c98494f51673ef0d42cc03e71ead26d993da93521291b315ebc33d41fd` | `d6221272b531e01e848563b8c7bb5e924209e0b341f45cfc3a9e816c17f8fa36` |
| 1 | `93a8f5873a7ac3f8d01492c01cd4a2b39068db3c988fce6aca14d58bf4b094bb` | `f2305c089e843a78045cd43200bbfd7c9ba7e967cecb89395422253f7776530a` |
| 2 | `9e03c9fffc1eefcd9abaeae6e2140fec7b8018a022fc6d606fd1ebbfa0cdd849` | `e6c87095797384f46142cb93c1e5a407d808287fdc83d7b1df99aa3e9309dfa9` |
| 3 | `01fb6be48611fb5fb5e63b0205053337e8ab48d5c50325ba1c0ff44c2bb4ac2a` | `0b45485d601ae49991195e66947457370f9d2b08474bd85e3d8bd5182b828ebd` |

V2 and matched V1 each ran 12 fresh processes and 1,800 steps. Across both
families, the scientific execution produced 3,600 training steps, 3,600
backward calls, 24 optimizer creations, 3,600 optimizer steps, 3,600 scheduler
steps, and 144 checkpoints. Clean forward batches totalled 3,600. V2 added
1,800 augmented forward batches. Record-level forwards totalled 18,000 clean
and 9,000 augmented. Only the final step-150 checkpoint of each run was
evaluated.

All 24 runs had finite losses and gradients. V2 per-run total loss moved from
2.2902–2.3318 initially to 1.1386–1.4001 finally, with maximum recorded
gradient norm 1.2279 and zero NaN/Inf values. Matched V1 moved from
1.6362–1.6560 to 0.7893–1.0314, with maximum gradient norm 21.5854 and zero
NaN/Inf values. Parameter counts were 9,232 for V2 and 3,589 for matched V1.

## Calibration and held-out evaluation

Each rotation/seed selected from 81 calibration-only threshold candidates
using the frozen lexicographic objective. No test record, target render, or
ground-truth pair entered calibration or controller inference.

| Rotation/seed | tau_mix | tau_pair |
|---|---:|---:|
| r0/s0 | 0.80 | 0.20 |
| r0/s1 | 0.90 | 0.10 |
| r0/s2 | 0.90 | 0.10 |
| r1/s0 | 0.80 | 0.05 |
| r1/s1 | 0.80 | 0.10 |
| r1/s2 | 0.80 | 0.10 |
| r2/s0 | 0.70 | 0.20 |
| r2/s1 | 0.90 | 0.05 |
| r2/s2 | 0.70 | 0.20 |
| r3/s0 | 0.70 | 0.15 |
| r3/s1 | 0.70 | 0.15 |
| r3/s2 | 0.70 | 0.15 |

The evaluator performed exactly 6,240 unique controller forwards: 960
calibration, 1,920 primary test, 480 formal-pure secondary, and 2,880
perturbation. All 6,240 used frozen feature rows; failed inference count was
zero.

## Results and preregistered gates

V2 protocol-weighted unordered top-2 pair accuracy was **0.601042**, against
the required 0.90. Rotation macros were 0.608333, 0.579167, 0.600000, and
0.616667, all below the required 0.80. This is a severe core pair
identification failure and determines the final classification.

Other V2 results were:

- mixedness AUPRC 0.962465, AUROC 0.874722, Brier 0.143964,
  cross-entropy 0.441496, ECE 0.160270;
- pure false-mixed 0.050000 (passes <=0.05) and pure SINGLE 1.000000;
- mixed false-SINGLE 0.830556;
- correct compatible DUAL 0.189378 (fails >=0.80);
- correct incompatible HARD 0.313889 (fails >=0.80);
- incompatible DUAL 0.000000 (passes <=0.10);
- wrong-pair DUAL 0.072824 (passes <=0.10);
- correct-pair weight MAE 0.158058 and RMSE 0.198143 (MAE fails <=0.12).

Formal-pure secondary evaluation gave V2 top-1 1.0 and SINGLE 1.0. Complete
reference dropout and single-reference ablations both gave safe SINGLE 1.0
and wrong DUAL 0.0, although pair flips were respectively 0.970833 and
0.662500. No information missing from these ablations was treated as
recoverable ground truth.

The manual visual audit opened all 240/240 main sheets at original detail.
V2 retained 27 sheets with grade-3 core patch/cloud/mottle/full-body
contamination versus 100 for matched V1, a 73% reduction that passes the
relative visual gate. This improvement is mainly routing exposure reduction:
V2 used 205 SINGLE, 32 DUAL, and 3 HARD sheets, while matched V1 used 135
SINGLE and 105 DUAL sheets. It does not establish clean continuous control.
V2 maximum grades remained 3 for patch, cloud, mottle, full-body
contamination, and wrong-garment mixture; edge scatter and silhouette
discontinuity reached grade 2. Identity contamination, double outline, and
ghosting all had maximum grade 0.

Rendering preserved all failures. It executed 5,280 logical render records,
1,035 new physical signatures, 4,245 exact-signature reuses, and zero
failures. Historical oracle/target/endpoint outputs were reused 5,280 times.
No output, attempt, or failure was deleted.

## Validation

All required JSON files parsed, all frozen YAML files parsed, all five
Markdown reports were non-empty, every one of the 240 local review paths
existed, and `git diff --check` passed. The current repaired
training/evaluation/rendering and V2 design tests passed 56/56 under the
frozen cloud environment. The optional legacy contract-repair test module
passed 19/20; its sole failure asserts that the checked-out branch must still
be the preceding repair branch, so it is inapplicable after the mandated
execution-branch transition and was not edited or suppressed.

## Classification

`CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL`

The unique primary failure is core pair identification. Routing calibration,
weight accuracy, and residual visual contamination are additional failures,
but there was no cross-fit leakage, target-forward leakage, GT inference use,
identity contamination, or numerical instability. `PAPER_FINAL=0`.

The next task is
`DIAGNOSE_CONTROLLER_V2_PAIR_IDENTIFICATION_FAILURE`. It was not started.
