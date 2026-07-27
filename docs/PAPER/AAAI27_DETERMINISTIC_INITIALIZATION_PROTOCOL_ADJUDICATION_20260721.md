# Deterministic Initialization Protocol Adjudication

Date: 2026-07-21  
Task: `AAAI27-DETERMINISTIC-INITIALIZATION-PROTOCOL-001`

## Decision

The protocol passes without training. The earlier `SEED-PROPAGATION-FAIL` is preserved as the truthful outcome of the earlier preregistered hard gate. It has not been deleted, overwritten, or relabeled. The root cause is now adjudicated as `DETERMINISTIC_INITIALIZATION_CONTRACT_MISMATCH`.

`MultiOutfitLinearCoefficientControl` consumes the configured seed while constructing a default `Linear`, then explicitly sets that Linear's weight and bias to zero. Its final trainable state is therefore seed-independent by design. The historical shared module remains unchanged.

## Frozen candidate policies

| Candidate | Policy | Step-0 contract |
| --- | --- | --- |
| Ours-v2 | `DETERMINISTIC_ZERO_INITIALIZATION` | LayerNorm weight 1/bias 0; Linear(4) weight 0/bias 0; 3,076 trainable parameters; same- and cross-seed fresh processes are bitwise exact. |
| B6 | `RANDOM_SEEDED_INITIALIZATION` | Same seed is bitwise exact; seeds 0/1/2 produce three unique state hashes. |
| M3/M4 | `PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD` | Same-seed trunks are shared exactly; cross-seed trunk hashes are unique; both output heads are zero; initial output semantics match. |

Ours-v2 no longer uses cross-seed initialization uniqueness as an acceptance condition. B6 and the M3/M4 trunks remain seed-dependent. The protocol-only candidate implementations are isolated from historical A6, B5, and shared formal builders.

## Semantic and fairness decisions

- Standardized coefficient zero means `MEAN_COEFFICIENT`.
- Raw coefficient zero means `MEAN_GARMENT_RESIDUAL` in the explicit basis.
- Physical zero residual means `BASE_AVATAR`.
- Evaluator zero replacement does not equal base replacement.
- Required term: **MEAN-GARMENT INITIAL PREDICTION**.
- `M1_M2_INITIALIZATION_FAIRNESS = PASS`.
- `M3_M4_PAIRED_INITIALIZATION_FAIRNESS = PASS`.
- `M1_M3_INITIAL_OUTPUT_SEMANTICS_FAIRNESS = PASS`.
- `M2_M4_INITIAL_OUTPUT_SEMANTICS_FAIRNESS = PASS`.
- Cross-architecture fairness means `INITIAL_OUTPUT_SEMANTICS_MATCHED`, not identical parameter initialization.

## Historical interpretation

Formal Ours, A1-A7 trainable zero-initialized cells, B3, and B4 cannot be uniformly described as independent random initializations. The sealed evidence supports shared deterministic initialization and a conservative `PARTIALLY_STOCHASTIC_SHARED_INITIALIZATION` classification. Historical B5 remains `INDEPENDENT_RANDOM_INITIALIZATIONS`.

The reviewer-risk registry remains unchanged: `RR-OURS-V2-S0`, `RR-OURS-V2-S1`, and `RR-OURS-V2-S2` are still `FAILED`, because those statuses describe the earlier hard-gate attempt. This adjudication records an independent protocol status rather than rewriting history.

## Optimizer provenance and no-training gate

Candidate and legacy namespaces are separate:

- Candidate optimizer: not created; 0 parameters; 0 `zero_grad`; 0 steps; no saved state.
- Legacy/context optimizer: one `torch.optim.adam.Adam` object was constructed while loading legacy context, contained 128,629 legacy/context parameters in five groups, contained no candidate parameters, executed 0 `zero_grad` and 0 steps, saved no state, and was discarded by its caller.

Optimizer-object construction is not a training step. No candidate optimizer, backward pass, scheduler step, checkpoint, render, formal metric, or formal registry transition occurred.

## Immutability and final status

Before/after evidence is identical:

- Formal registry SHA256: `1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e`
- Formal output metadata SHA256: `7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc` (`4,127` files; `964,043,888` bytes)
- Frozen manifest SHA256: `ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb`
- Old seed audit SHA256: `0b2e068203031b34e0004725923d452e185f319f3e9088843ecbc42ba5307f14`
- `PAPER_FINAL = 0`

Final classifications:

- `DETERMINISTIC_INITIALIZATION_PROTOCOL = PASS`
- `MEAN_GARMENT_ZERO_SEMANTICS = RESOLVED`
- `M1_M4_INITIALIZATION_FAIRNESS = PASS`
- `HISTORICAL_THREE_SEED_INTERPRETATION = CORRECTED`
- `OPTIMIZER_PROVENANCE_SEPARATION = PASS`

The next and only recommended task is `IMPLEMENT_P0_CANDIDATE_ADAPTERS_AND_RUNNER_WITHOUT_FORMAL_TRAINING`. It is not started by this adjudication.
