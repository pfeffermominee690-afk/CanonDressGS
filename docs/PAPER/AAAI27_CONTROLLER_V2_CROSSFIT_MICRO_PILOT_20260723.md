# Controller V2 Cross-Fit Micro-Pilot: Pre-Result Contract Gate

**RESEARCH MICRO-PILOT — NOT PAPER FINAL**

- Task: `AAAI27-CONTROLLER-V2-CROSSFIT-MICRO-PILOT-001`
- Source: `research/compatibility-gated-controller-v2-design-20260723@4f8c94405e68932e791f69c59ae129c863b1280d`
- Branch: `research/compatibility-gated-controller-v2-crossfit-micro-pilot-20260723`
- Classification: `CONTROLLER_V2_MICRO_PILOT_CONTRACT_INCOMPLETE`

## Stage 0 resource and activity gate

The RTX 4090 was idle with 24,081 MiB free GPU memory and no compute process.
`/root/autodl-tmp` had 48,504,643,584 free bytes (about 45.2 GiB), above the
25 GiB gate, and 704,915,826 free inodes. No AvatarReX extraction/full JPEG
audit, subject00 runtime, formal renderer, or CPU/disk-heavy task was active.
Stage 0 therefore passed.

## Stage 1 contract result

Only 8/30 required fields were uniquely frozen;
22/30 were missing. The absent fields are:

- `rotation_record_ids`
- `train_calibration_test_counts`
- `duplicate_preservation`
- `batch_size`
- `per_step_record_exposure`
- `data_order_algorithm`
- `optimizer_type`
- `learning_rate`
- `weight_decay`
- `total_optimizer_steps`
- `checkpoint_cadence`
- `final_checkpoint_rule`
- `lambda_mix`
- `lambda_weight`
- `lambda_cons`
- `augmentation_severity`
- `augmentation_exposure_schedule`
- `information_ablation_training_contract`
- `calibration_tie_break`
- `final_evaluator_denominators`
- `perturbation_representative_ids`
- `visual_grade_schema`

The design correctly freezes cross-fit membership, nuisance types, both
threshold grids, the pair-confidence scalar, the calibration objective, seeds,
and success gates. It does not freeze the data records or executable training
schedule. Supplying those values from code defaults would violate the task.

## Matched V1 comparability

A strict matched V1 baseline cannot be instantiated without the same exact
record IDs, data order, exposures, optimizer-step budget, and final-checkpoint
rule. Historical Formal V1 is not substituted because its protocol exposed all
condition folds.

## Mandatory stop

Training runs, forward batches, backward calls, optimizer creations/steps,
scheduler steps, checkpoint loads/writes, calibration, inference, render calls,
metrics, and visual sheets all remain 0. No downstream scientific result or
success-gate judgment was manufactured. PAPER_FINAL remains 0.
