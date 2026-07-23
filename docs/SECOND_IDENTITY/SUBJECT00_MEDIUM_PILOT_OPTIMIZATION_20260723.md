# Subject00 medium-pilot optimization audit

`PAPER_FINAL=0`

The completed one-pass run used 14 frozen optimizer groups,
9 schedulers, and scheduler horizon
800,000. Optimizer-contract audit status:
`PASS`. There were no omitted trainable parameters, duplicate
parameter memberships, LR mismatches, or excluded-state optimizer entries.

Training executed exactly 20,249 forward batches, 20,249 backward calls, and 20,249 optimizer
steps. All records were unique and no record was repeated. No infrastructure resume or silent
restart occurred.

Total-loss first/last 5% medians were
0.008396284189075 and
0.007115250686184, a
15.257148% reduction. Primary L1 medians were
0.008318810723722 and
0.002171005820855, a
73.902450% reduction. LPIPS became
nonzero at step 6001 under the frozen
`step > 6000` rule and remained nonzero for 14,249
steps.

Losses, gradients, intended gradients, and optimizer state were finite as required. No topology or
surface-LBS mutation operation ran. This audit reports the frozen one-pass result and does not tune
or extend it.
