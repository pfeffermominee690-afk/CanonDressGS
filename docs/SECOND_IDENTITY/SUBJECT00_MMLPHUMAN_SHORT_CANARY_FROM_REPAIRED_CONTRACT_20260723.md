# Subject00 MMLP-Human short canary from repaired contract

## Outcome

`SUBJECT00_MMLPHUMAN_SHORT_CANARY_PASS`.

The repaired contract at `ebcf40da0fca0345f749f7c111e1a3a00f19dda3` was frozen and verified before
optimizer creation. One logical training run completed exactly 384 unique
forward/backward/optimizer steps with five checkpoints at 0/96/192/288/384.
No scientific rerun, tuning, early stop, best-checkpoint selection or extra
step occurred. The infrastructure-only step-96 continuation is fully recorded;
no completed record was repeated and no `attempt_002` was created.

## Runtime and signal

- Total loss first48/last48 median:
  `0.009627176449` /
  `0.009059807751` (reduction
  `5.893407%`).
- L1 first48/last48 median:
  `0.009598378558` /
  `0.008991606068`
  (reduction
  `6.321614%`).
- Train-pose/train-camera mean LPIPS:
  `0.137431931061` to
  `0.121086820339`.
- Query-level LPIPS improvement: `24/24` (required `>=18/24`).
- Held-out pose quadrants: `48/48` finite and alpha-nonempty, `0` explosions.
- Formal renders: `216/216` successful; roundtrip adds 8 fresh renders.
- Final fresh-process roundtrip: `PASS`, maximum array difference `0.0`.
- Peak VRAM: `1640268288` bytes.

## Visual audit and limitation

All 192 main visuals were covered by opening the 24 original-resolution contact
sheets (eight labeled query composites per sheet). No gross body explosion,
head-eye contamination, hand-finger contamination, component separation,
detached cloud, empty/black render, full-frame opacity, silhouette collapse or
camera mismatch was observed.

The RGB predictions remain low-texture gray bodies and the visual change is
small at contact-sheet scale. This is a runtime/optimization canary result, not
a formal reconstruction-quality claim.

## Provenance

The formal reference remains: subject02 formal MMLP-Human run (output
directory: `subject02_formal_800k`, archived checkpoint iteration: `100000`).
Its status is `LIMITED_HISTORICAL_PROVENANCE`; historically exact source
recovery is not claimed.

All frozen source/data/split/template/attachment/LBS mutation counts are zero.
`PAPER_FINAL=0`. The next task is `RUN_SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_WITH_STRICT_SPLITS` and was not started.
