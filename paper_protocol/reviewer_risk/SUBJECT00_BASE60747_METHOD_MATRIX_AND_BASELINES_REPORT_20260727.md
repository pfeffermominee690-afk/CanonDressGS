# Subject00 Base60747 Method Matrix and Fair Baselines Report

Task: `AAAI27-SUBJECT00-BASE60747-METHOD-MATRIX-AND-FAIR-BASELINES-001`

Status: `SUBJECT00_BASE60747_METHOD_MATRIX_ENGINEERING_FAIL`

## Outcome

No new optimizer was created. The requested 12-run matrix cannot be
instantiated under the simultaneously frozen rotation-fold and quarantine
contracts. Slot04 is the right-view fold, but the authoritative 22-record
manifest contains only O04 for slot04:

| slot04 garment | eligible record count |
|---|---:|
| O01 | 0 |
| O03 | 0 |
| O04 | 1 |

O01 and O03 slot04 are the two records that the task permanently excludes.
Rotation0 therefore lacks O01/O03 in its test fold, rotation1 lacks them in
calibration, and rotations2/3 lack them in training. Using the quarantined
records, imputing records, moving a fold, changing a denominator, or changing
the garment-balanced batch would all violate the task.

## Completed rotation0/seed0 audit

The prior `METHOD-R0-S0` training is authentic: 300 strictly monotonic
optimizer steps, exact checkpoints 0/20/50/100/200/300, finite loss and
gradients, all four controller tensors changed, immutable bindings match, and
quarantine use is zero. Its output tree contains 19 files, 53,138,437 bytes,
with fingerprint
`ce281935d12e20c6bab2e3e8fc3256b512594fd8388c3ed299f71199072ec363`.

It is not a complete formal cross-fit cell. Its recorded 6/6 result evaluates
the two training folds, not the frozen rotation0 test fold. The test fold is
slot04 and contains only O04 after quarantine. The training result is
preserved without mutation but is not promoted to a formal matrix result.

## Fair baselines

The exact Subject00 paper-facing set is uniquely recovered:

1. Reference Classifier Lookup
2. Nearest-Centroid Lookup
3. Outfit-ID Oracle
4. Teacher Endpoint

The task requires the method matrix to complete before baseline execution.
Consequently baseline runs, comparison statistics, and ETA remain `null`;
historical Subject02 metrics are not substituted.

## Immutability

Base60747 CPU load and internal step60747 pass. The three Teacher SHA values
match the frozen registry. All 330 target/raw/mask checksum bindings match.
Formal Base remains `USER_AUTHORIZED_PAUSED`; no paper text, checkpoint,
target, raw, mask, camera record, or completed method output was modified.

`PAPER_ELIGIBLE=false`, `SCIENTIFIC_PASS=null`, and `PAPER_FINAL=false`.

The only next task is
`USER_REVIEW_SUBJECT00_BASE60747_METHOD_MATRIX_ENGINEERING_FAILURE`.
