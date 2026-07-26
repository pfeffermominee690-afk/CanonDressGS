# Full Avatar Fine-tuning Contract

Status: `READY_FOR_USER_AUTHORIZED_MICRO_PILOT`

## Fixed inputs

- Start each run from the exact formal subject02 Base Avatar initialization.
- Run O01/O02/O03/O04/O08 independently with the same registered multi-view garment targets and frozen train/calibration/test split used by the Teacher Endpoint protocol.
- Do not read reference images. Garment identity selects the training target only; no target-test RGB, mask, pose, camera, or Teacher residual may enter optimization or tuning.
- The complete avatar backbone may update. Test-set tuning, best-seed selection, hidden identity drift, and cross-garment resume are forbidden.
- Every formal run uses an independent attempt directory. Resume is allowed only from a recorded optimizer-step checkpoint with model, optimizer, scheduler, RNG, and elapsed-time state.

## Budgets

- Equal-step: for garment `g`, use exactly `N_teacher(g)`, the optimizer-step count of its formal Teacher Endpoint construction. The frozen bundle does not export this count, so execution preflight must bind it from sealed Teacher metadata; it must not be guessed or replaced by the 300 controller steps.
- Equal-wall-time: for garment `g`, stop at the first completed optimizer step at or beyond the measured formal Teacher construction wall time `T_teacher(g)`. Report any overshoot. The controller cost is not hidden: report Teacher construction, controller training amortized over five garments, and total five-garment CanonDressGS onboarding separately.
- Use fixed preregistered seeds and aggregate all seeds. Never select the best seed.

## Required reporting

- Trainable parameters, optimizer steps, wall time, peak VRAM, checkpoint bytes, optimizer-state bytes, and five-garment incremental storage.
- All target-space and protected-region metrics in `external_target_space_metric_protocol.md`.
- Identity contamination, catastrophic drift, original animation compatibility, original pose/camera rendering, and exact resume behavior.
- A run is catastrophic drift if protected-region failure is grade 2-3, original animation cannot render, or the Base Avatar identity review fails.

No training is authorized by this document. `TRAINING_STEPS=0` for this audit.
