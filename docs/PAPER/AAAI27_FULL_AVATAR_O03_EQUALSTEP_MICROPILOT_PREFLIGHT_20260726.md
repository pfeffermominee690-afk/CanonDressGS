# Full Avatar O03 Equal-Step Micro-Pilot Preflight

Task: `AAAI27-FULL-AVATAR-FINETUNING-O03-EQUAL-STEP-MICRO-PILOT-001`

## Decision

`FULL_AVATAR_O03_EQUALSTEP_MICROPILOT_PREFLIGHT_BLOCKED`

The source branch, source HEAD, 45 feasibility tests, frozen bundle hash, formal Base Avatar, O03 Teacher checkpoint, O03 target manifest, cloud storage, single-GPU environment, and renderer import all passed. The equal-step budget is uniquely `1200` optimizer steps.

Training did not start because the Full Avatar execution contract is under-specified in four scientifically consequential places. Four frozen condition rotations exist and none is selected for this one-run pilot. The Base and Teacher provenance define different loss contracts, but no Full Avatar loss is selected. The Base checkpoint contains optimizer and scheduler state, but restore-versus-reset and group learning-rate mapping are not selected. Finally, seed `0` and seed `20260718` are both provenance-backed, but neither is designated as the Full Avatar seed.

Choosing any of these values automatically would violate the instruction not to guess the split, loss, learning rate, trainable contract, or deterministic seed. The cloud output root and `attempt_001` were therefore not created. Optimizer steps, backward calls, checkpoints, renders, generated images, data mutations, and paper modifications all remain zero.

## Bound facts

- Base checkpoint: `/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth` at step `100000`, SHA256 `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- O03 Teacher: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003/stage_a_teacher_bank/O03/checkpoints/step_001200.pth`, SHA256 `16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93`
- Equal-step budget: `1200`
- O03 observations: `4` RGB, `4` foreground masks, `4` garment masks, `4` cameras
- Cloud free bytes before output creation: `46450372608`
- Storage gate: `PASS`; estimated five full checkpoints plus 2 GiB auxiliary data leave `35769837450` bytes
- GPU: one NVIDIA GeForce RTX 4090; PyTorch 2.4.1+cu121; CUDA 12.1; gsplat import passed

## Required resolution

Freeze exactly one split/evaluation contract, one loss contract, one optimizer initialization and learning-rate contract, and one seed. No other garment or baseline may start from this task.

Unique next task: `USER_RESOLVE_FULL_AVATAR_MICROPILOT_PREFLIGHT_BLOCKER`

`PAPER_FINAL=false`
