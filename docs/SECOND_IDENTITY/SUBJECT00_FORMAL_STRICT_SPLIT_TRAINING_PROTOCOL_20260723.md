# Subject00 formal strict-split training protocol

Task `MMLPHUMAN-SUBJECT00-FORMAL-STRICT-SPLIT-PROTOCOL-001` freezes a pre-result protocol only. It authorizes no optimizer construction, training, checkpoint write, render, metric computation, or result selection during this task.

## Provenance and initialization

The sole source is `research/mmlphuman-subject00-one-pass-medium-pilot-20260723` at `2d0913eb1b163a79e81c1804c216f91b0a737b44` with classification `SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS`. Runtime provenance is canary `b0e8096589fe18069d95d2137e1db3979b4fe89f` plus medium runner commit `36c43d9dd29abfe546a5352b46378244f3bd9f1b`. The only authorized initialization is `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/checkpoints/step_000000.pth` (`29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a`, 701938720 bytes). Medium and canary nonzero checkpoints, subject02 checkpoints, and random initialization are forbidden.

## Frozen training schedule

The unchanged source manifest has 20,340 theoretical, 20,249 valid, and 91 official-missing records; content SHA is `865118c2f216046008925ef14b049db6e1d2921922117f6e3c3e3e2fdacd3537` and stable single-pass order SHA is `0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8`. Formal training is exactly five complete repetitions of that order: 101,245 exposures and optimizer steps, every record exactly five times, no shuffle, replacement, oversampling, skip, within-pass repeat, early stop, extension, or result-driven decision. Final is step 101,245; step 100,000 is a secondary subject02 matched-iteration reference and can never compete with final.

The full schedule SHA is `79e007e3d4884ea199a273287bb42388473cddd63278efff6490ad28ddecaf78`. Checkpoints are `[0, 20249, 40498, 60747, 80996, 100000, 101245]`: step0 is a pointer and six future checkpoints must be written and retained. There is no best-checkpoint selection.

## Runtime and recovery

The 14 optimizer groups, Adam/AdamW settings, learning rates, nine ExponentialLR schedules with horizon 800,000, five unscheduled BS groups, batch size 1, seed 0, disabled AMP, accumulation 1, no gradient clipping, renderer, random background, mask rules, L1, LPIPS 0.1 after step 6000, dxyz smooth 0.1, and scaling regularizer 1.0 at threshold 0.01 are copied without tuning. `LIMITED_HISTORICAL_PROVENANCE` remains explicit.

Infrastructure recovery uses the nearest complete hash-verified checkpoint in the same attempt and restores model, optimizer, scheduler, all RNG states, and exact data-order position. It may neither duplicate nor skip a record. Scientific failure cannot trigger a restart, extension, contract change, or replacement attempt.

## Safety

The frozen inventory is 200,000 Gaussians, 200,000 surface attachments, LBS `[200000,55]`, and zero off-surface elements. Clone, split, densification, topology-changing prune, off-surface rebind, legacy-grid load, and spatial-LBS query counts must remain zero. No frozen dataset, derived asset, template, attachment, LBS, split, subject02, AvatarReX, canary, or medium artifact may mutate.
