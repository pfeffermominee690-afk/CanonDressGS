# CanonDressGS Editable Gaussian Pool Specificity Audit

## Formal identity

- Task: `SUBJECT02-EDITABLE-POOL-SPECIFICITY-001`
- Formal candidate: `attempt_007`
- Run commit: `9dcb1472dc25ed3bfa6d37c12b53f3f6b32b491a`
- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-EDITABLE-POOL-SPECIFICITY-001/attempt_007`
- Status: **PASS / Case PP**
- Optimizer created: `false`; optimizer steps: `0`

`attempt_001` stopped before model load because a clean clone exposes frozen branches as remote refs. `attempt_002` was stopped after discovering an audit-only quadratic aggregation defect. `attempt_003` stopped on a historical JSON list/dict compatibility error. None is a model candidate and all remain preserved. `attempt_006` is a complete 64-pixel protocol check. `attempt_007` is formal because it restores the frozen source protocol of 48 deterministic pixels per region and exactly reproduces the decisive P3/O08/back N_pre mean `78.166667`.

## What Case PF meant

The earlier failure was not a Proxy A formula failure. Proxy A-0.10 preserved the required coverage ordering and correlations: global Spearman `0.999130`, O01/O08 Spearman `1.0/1.0`, median per-pixel Spearman `0.998141`, and O08 back/right P1/P3 ratios `4.712522/4.593608`. The failure occurred because the single target-independent pool was also asked to represent spill contributors that its protected contract intentionally excludes.

## Frozen current pool and all 200k attribution

- Official `.npy` SHA256: `232124458848a684cd98bef1b162ec889f1e3e4c80c0ef1a5caebcceb4edb011`.
- Raw int64 index SHA256: `80c1d4952dfcf54be1c53bf9a853e502af03a362707bc380c29af980c3faabac`.
- Membership: `169106 / 200000`.
- Included: `169103` `BASE_OLD_CLOTHING_SEED`, `3` `BASE_GARMENT_ENVELOPE_SEED`.
- Excluded: exactly `30894` `STABLE_PROTECTED`; no unexplained `other` bucket exists.

The Parquet attribution contains one row for every index and records canonical coordinates, base visibility/evidence, anchor/component/LBS/body-part identity, explicit membership reason, and frozen P1/P2/P3 residual/posed/screen evidence. The official pool file was only read and cross-checked; it was never overwritten.

## Coverage responsibility

For trusted expansion alone, current-pool index/N_pre/active/alpha recall is `0.987596 / 0.990725 / 0.989760 / 0.991668`. For trusted expansion plus correctly covered garment, it is `0.994686 / 0.995134 / 0.993740 / 0.992817`. Thus the current pool passes the preregistered `>=0.95` N_pre coverage criterion and remains the correct `G_coverage`.

## Cloud contributors and source

Across all frozen P1/P2/P3, O01/O08, and four-view cloud regions, current-pool index/N_pre/active/alpha recall is `0.693612 / 0.621402 / 0.661802 / 0.665771`.

The decisive source-failure split is much sharper. At P3/O08/back, production N_pre is `78.166667` while Proxy A-0.10 is `1.368e-30`; all four recalls are exactly `0`. All `2646` unique contributors (`3752` N_pre occurrences, `1951` active contributions, alpha mass `42.120168`) are `EXCLUDED_PROTECTED`, so the protected alpha fraction is `1.0`.

The focus body/LBS attribution is uniquely concentrated in right-leg joint 8 (`1727`) and foot/shoe joint 11 (`919`). The largest anchor groups contain only 18–25 Gaussians each, so there is no single dominant anchor explaining the failure. Every cloud contributor is formally movable under fixed-open P2/P3, but the pool contract correctly blocks placement supervision on protected identity Gaussians.

## Same-index migration and valid coverage overlap

For the P3/O08/back focus set, `2522 / 2646` Gaussian centers enter the cloud component and `124 / 2646` are wide-tail-only contributors. Canonical displacement has mean/median/max `0.007769 / 0.004772 / 0.060381`; screen displacement has mean/median/max `3.503678 / 1.933661 / 32.928474` pixels. Only one focus Gaussian also appears in sampled valid coverage. The focus cloud is predominantly same-index protected leg/shoe center migration, not merely broad Gaussian tails or useful garment points overshooting the target.

Across all splits, `6117 / 19818` cloud contributors also support sampled valid coverage; `14103` centers enter a cloud and `5715` are wide-tail-only. These global facts do not override the focus-specific Case PF attribution.

## P0/P1/P2 qualification

All three contracts produce exactly the same spill pool: `169106` members, raw-index SHA256 `80c1d495...`, protected overlap `0`.

- P0 reproduces the Case PF focus miss.
- P1 equals P0 because fixed-open formal trainability covers all 200k and the only excluded set is stable protected.
- P2 also equals P0 after the same stable-protected subtraction.

All candidates fail the `0.90` cloud recall gates. P1/P2 cannot recover the focus cloud without violating the frozen protected constraint. Cloud/background separation and positive proxy correlation pass, O01 background risk is `0`, target independence passes, and protected overlap/loss/gradient contracts are zero. These partial passes do not authorize a dual-pool freeze.

## Conditional stages and final decision

The preregistered static gate did not pass for P1 or P2. Therefore anti-saturation, full-quarter-resolution performance, and gradient-direction qualification were correctly **not run**; reporting values for them would violate the protocol.

Final case: **PP**. The current pool is a valid coverage pool, but the missing decisive cloud contributors are stable protected identity Gaussians. Direct spill supervision on them is prohibited. `DIFFERENTIABLE_DUAL_POOL_SUPPORT_PROXY_V1` is not frozen, placement does not resume, and no seven-outfit rerun, target generation, or training is authorized.

The only next task is `DESIGN_PROTECTED_AWARE_CLOUD_ATTRIBUTION`.

## Integrity and regression

Base fingerprint: `6cd91a21bf351d67812f68986d0862b8e16a29a582bd82eaad90fd83643bb448`.

New contract tests: `20/20`. Regressions: proxy `23/23`, placement `28/28`, instrumented projection `21/21`, alpha raster `20/20`, V6.1 `24/24`, V6 `27/27`, V5.3 `28/28`, R2 CUDA `12/12`, differentiable renderer unit checks PASS, full checkpoint checks PASS, image-conditioned dataset checks PASS, `py_compile` PASS, and `git diff --check` PASS.
