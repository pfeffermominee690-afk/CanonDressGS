# Repaired Controller V2 Routing and Weight Analysis

Calibration used only each rotation's frozen calibration fold and selected one
`tau_mix`/`tau_pair` pair per seed from 81 candidates with the fixed
lexicographic objective. Cross-seed threshold sharing, pair-specific
thresholds, test-fold calibration, target renders, and GT pair lookup were
all absent.

## Pair identification

V2 protocol-weighted unordered top-2 accuracy was 0.601042. Rotation macros
were r0 0.608333, r1 0.579167, r2 0.600000, and r3 0.616667. The macro gate
requires 0.90 and every rotation requires 0.80, so all pair gates fail.
Matched V1 reached 0.631250 and also failed.

The focus pairs expose the same problem across all primary records:

| Pair | Family | Records | unordered top-2 | dominant order | modes |
|---|---|---:|---:|---:|---|
| O01_O03 | V2 | 96 | 0.166667 | 0.697917 | 88 SINGLE, 5 HARD, 3 DUAL |
| O01_O03 | matched V1 | 96 | 0.270833 | 0.697917 | 80 SINGLE, 16 DUAL |
| O02_O03 | V2 | 96 | 0.375000 | 0.937500 | 78 SINGLE, 13 HARD, 5 DUAL |
| O02_O03 | matched V1 | 96 | 0.708333 | 0.927083 | 52 SINGLE, 44 DUAL |

Correct-incompatible DUAL was zero because correct incompatible predictions
were routed away from DUAL. Nevertheless, wrong pair predictions can select a
compatible lookup and expose DUAL; this is why the all-test focus-pair mode
counts and the aggregate wrong-pair DUAL rate must both be retained.

## Mixedness and compatibility routing

V2 pure false-mixed is exactly 0.05 and formal-pure SINGLE is 1.0. The
controller therefore satisfies the pure safety boundary. It fails mixed
coverage: mixed false-SINGLE is 0.830556 and correct compatible DUAL is only
0.189378. Correct incompatible HARD is 0.313889, also below the 0.80 gate.
Incompatible DUAL is 0.0 and wrong-pair DUAL is 0.072824.

These figures show that the calibration objective protects against the most
dangerous DUAL exposure chiefly by falling back, not by learning robust
mixedness and compatibility decisions.

## Pair-conditioned weight

For correctly identified pairs, V2 weight MAE is 0.158058 and RMSE is
0.198143. The preregistered MAE gate is 0.12, so weight calibration fails.
Matched V1 is worse at MAE 0.203023 and RMSE 0.236070, but the matched baseline
does not relax the V2 gate.

The pair recognition failure is the unique primary source for the final FAIL
classification. Mixedness/routing and weight calibration are secondary
failure sources. No threshold was changed and no seed, rotation, duplicate,
pair, or failed query was excluded. `PAPER_FINAL=0`.
