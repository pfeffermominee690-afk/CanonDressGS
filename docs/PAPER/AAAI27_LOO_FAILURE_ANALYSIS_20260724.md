# AAAI27 LOO Failure Analysis

## Failure Point

The cache-repaired execution materialized `attempt_001` from frozen execution HEAD `a868df3c3811483c5dd07456d8cabfdc3577d5f4`. Source, historical immutability, credentials, GPU, storage, task manifest, hard-lookup 15/5 replay, and `54,960/54,845/115` cache-plan preflights all passed. The attempt then failed at the first `LOO-O01` basis renderer-parity check.

Residual parity passed with aggregate normalized RMSE `2.2268962e-7`, maximum garment RMSE `2.6056503e-7`, deterministic repeat fingerprint, rank 3, and no held-out input. Despite that, reconstructed-render maximum errors ranged above the frozen `1e-5` threshold for every training endpoint. The largest alpha error was `0.0038456917`; the largest RGB error was `0.0029371381`.

## Governance Response

The attempt was preserved. No basis rerun, threshold change, seed change, retry, checkpoint resume, optimizer, or `attempt_002` occurred. The formal logical/physical render and cache-hit counts remained zero; the basis validator made eight auxiliary parity renders outside the formal cache plan. Temporary files were zero.

## Repair Scope

The next task is `REPAIR_LOO_ADAPTATION_EXECUTION_FAILURE`. It must adjudicate why residual-space parity at approximately `1e-7` does not satisfy the renderer-space `1e-5` max-absolute gate, without weakening a gate after seeing results. It must also make the runner's final verification and routing explicitly support `LOO_ADAPTATION_EXECUTION_INVALID`; the current success-only allowlist omits that failure classification.

No automatic repair or new scientific attempt is authorized here. `PAPER_FINAL=false`, `paper_final_count=0`, and no method claim is supported.
