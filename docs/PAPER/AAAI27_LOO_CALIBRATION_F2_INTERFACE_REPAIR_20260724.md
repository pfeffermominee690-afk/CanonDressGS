# LOO Calibration F2 Interface Repair

## Repair

The repair adds a strict typed reference adapter and changes only the LOO runner's
direct F2 boundary. K1/K2 use the original producer batch contract. Four calibration
observations are extracted as legal singleton `per_reference_f2` records and aggregated
with the frozen mean+max rule. Render calibration remains an RGB/mask interface.

## Pre-Optimizer Closure

The matrix closes `40/40` unique tasks: adaptation
`40/40`, calibration
`40/40`, test
`40/40`, and initialization
`40/40`. Basis construction is 5/5 PASS,
renderer parity is 20/20 PASS, hard lookup remains 15 same / 5 different, and
CACHE_KEY_V2 remains 54,960 logical / 54,845 physical / 115 hits.

Scientific semantic drift is `0`.
No formal attempt, renderer, optimizer, step, checkpoint, evaluation, metric, visual
sheet, or PAPER_FINAL artifact was created. Authorization: `READY_FOR_LOO_ATTEMPT_003_BEFORE_OPTIMIZER`.
