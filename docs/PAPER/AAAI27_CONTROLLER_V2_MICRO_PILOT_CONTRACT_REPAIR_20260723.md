# Controller V2 micro-pilot training-contract repair

Task `AAAI27-CONTROLLER-V2-MICRO-PILOT-CONTRACT-REPAIR-001` repairs the failed pre-result contract without running training,
inference, calibration, rendering, metrics, or visual review. The original audit
was 8 PASS / 22 MISSING / 30 total. The repaired audit is 30 PASS / 0 MISSING /
0 AMBIGUOUS.

## Frozen records and schedules

Each rotation contains exactly 160 training, 80 calibration, and 80 test
protocol records. Training/calibration/test record intersections are zero.
Consistent duplicates remain independent protocol records; no deduplication or
extra weighting occurs. Each filtered historical cycle has 32 five-record
batches in outfit order O01, O02, O03, O04, O08. The exposure-density formula
gives 150 optimizer steps and 750 clean record exposures per run. V2 and
matched V1 use identical clean order for seeds 0, 1, and 2.

The six checkpoints are steps 0, 30, 60, 90, 120, and 150. Only step 150 may be
evaluated. There is no early stopping, best-checkpoint selection, or
result-driven schedule change.

## Preserved boundaries

The formal-pure 20 records are an evaluation-only secondary safety set with
zero training exposure, zero calibration-threshold use, and zero contribution
to the primary cross-fit denominator. `attempt_001` remains the failed archive
and `attempt_002` was not created. All execution counters are zero and
PAPER_FINAL=0.

Global pre-result contract SHA256: `e5b720ccedfb4b0b39892011123010b6522ece7dd1d534032ec2e2e26f7863f7`.

Classification: `CONTROLLER_V2_MICRO_PILOT_TRAINING_CONTRACT_REPAIRED`.
