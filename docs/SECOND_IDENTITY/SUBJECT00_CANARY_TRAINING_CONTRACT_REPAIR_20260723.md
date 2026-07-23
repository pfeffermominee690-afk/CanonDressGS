# Subject00 short-canary training contract repair

Task: `MMLPHUMAN-SUBJECT00-CANARY-CONTRACT-REPAIR-001`

Source is `research/mmlphuman-subject00-surface-lbs-runtime-20260723` at `dca5df55354b45d5abf1cd493e718f294558edc6`. The repair changes only the
deterministic fixed evaluation-pose selection. The 64 training poses, six
training cameras, 384 pose-camera exposures, and exposure order are unchanged.

## Frozen training records

- Pose count: 64
- Camera IDs: `[1, 5, 10, 14, 19, 23]`
- Record count and uniqueness: `384/384`
- Exposure per record: exactly one
- Batch size: 1; shuffle: false; replacement: false; seed: 0
- Held-out camera / held-out pose / buffer exposure: `0/0/0`
- Data-order SHA256 before and after: `a64a8d40876946b8f0919a4761e7b814e9ca7182cd0434b0cf97a6b2737386ca`

## Repaired evaluation

The only selection policy is `AVAILABILITY_FILTER_THEN_EQUAL_SPACING`.
The repaired train-evaluation poses are `[0, 673, 1555, 2493]` and the repaired
held-out-evaluation poses are `[56, 931, 1763, 2499]`. Each is valid on the same
12-camera union. Four quadrants contain 24 queries each; total validity is
96/96. The stable query-order SHA256 is `38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a`.

## Recovered formal training contract

The contract uses the precise term: subject02 formal MMLP-Human run (output
directory: `subject02_formal_800k`, archived checkpoint iteration: `100000`).
The prospective runtime closure is commit
`3382078fbdd77bba8bb9df54452647456a4c9b00`, with archived checkpoint SHA256
`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`.

`LIMITED_HISTORICAL_PROVENANCE` remains mandatory: the closure is the accepted,
clean prospective baseline, but the historical git ancestry of the executed
subject02 run is not claimed as recovered.

The machine contract persists optimizer types and 14 parameter groups, learning
rates, weight decay, four loss components and weights, trainable and frozen
inventories, image scaling, Gaussian count, batch size, mixed-precision state,
gradient accumulation and clipping, scheduler formulas and horizon, random
background and mask handling, renderer settings, and the 42-key checkpoint v2
schema.

## Authorization boundary

`formal_training_enabled=false`, `canary_training_enabled=false`, and
`PENDING_MANUAL_CONFIRMATION=0`. This task creates no optimizer, training
forward, backward, checkpoint, render, visual, or formal training output.
Topology mutation remains fail-closed.

Classification: `SUBJECT00_CANARY_TRAINING_CONTRACT_REPAIRED`.
Next task (not started):
`RUN_SUBJECT00_MMLPHUMAN_SHORT_CANARY_FROM_REPAIRED_CONTRACT`.
