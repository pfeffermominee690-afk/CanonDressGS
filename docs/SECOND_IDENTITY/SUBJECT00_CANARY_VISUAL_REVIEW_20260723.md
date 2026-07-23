# Subject00 canary visual review (2026-07-23)

## Completion

`PASS`: step 0 `96/96`, step 384 `96/96`, total `192/192`.

All 24 original-resolution contact sheets were actually opened. Each contains
eight labeled per-query composites with ground-truth RGB, ground-truth mask,
predicted RGB, predicted alpha, predicted depth, pose ID, camera ID, quadrant
and step, so all 192 persisted visual paths were covered.

## Gross runtime findings

- Body explosion: 0
- Head-eye contamination: 0
- Hand-finger contamination: 0
- Component separation or severe component contamination: 0
- Detached clouds: 0
- Empty/black render: 0
- Full-frame opacity: 0
- Silhouette collapse: 0
- Camera mismatch: 0

## Required limitation

The predicted RGB is a low-texture gray body at both step 0 and step 384, and
the visual change is small at contact-sheet scale. Fine head/eye, hand/finger
and clothing texture quality is not established. This is only a
runtime/optimization canary visual audit and must not be cited as formal
reconstruction quality. `PAPER_FINAL=0`.
