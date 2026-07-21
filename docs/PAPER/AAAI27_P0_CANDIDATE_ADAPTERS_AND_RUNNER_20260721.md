# P0 Candidate Adapters and No-Training Runner

Date: 2026-07-21  
Task: `AAAI27-P0-CANDIDATE-ADAPTERS-RUNNER-001`

## Scope

This task implements the P0 candidate interfaces and a unified step-0 runner. It does not authorize or execute formal training. No backward pass, optimizer step, scheduler step, checkpoint, render, formal evaluator, formal metric, registry transition, or `PAPER_FINAL` artifact belongs to this task.

The runner consumes the sealed frozen-reference feature cache, rank-4 basis, coefficient normalization, and teacher-residual asset identities. Target pose/camera may be consumed later by the frozen MMLP-Human and renderer, but they are not inputs to any garment prediction branch.

## Runtime population

`paper_protocol/reviewer_risk/p0_candidate_runtime_manifest.yaml` contains exactly 13 entries:

- Ours-v2: three deterministic protocol replicates, replicate indices 0/1/2.
- B6: three random-seeded classifier runs, seeds 0/1/2.
- B7: one fixed non-trainable nearest-centroid baseline.
- M3: three random-seeded complex-fusion runs.
- M4: three random-seeded complex-fusion runs paired with M3 by seed.

Twelve entries are trainable and one is non-trainable. Every static entry is `PREFLIGHT_READY` and has `formal_run_authorized=false`. This manifest is independent of the frozen reviewer-risk registry and does not add the 27 Ours-v2-centered ablations.

## Candidate contracts

### Ours-v2

Implementation: `scene.p0_candidate_adapters.OursV2CandidateAdapter`.

The frozen reference extractor supplies per-reference F2 clothing mean/max features. The adapter performs deterministic valid-reference set mean/max pooling, then LayerNorm and Linear(4). It has 3,076 trainable parameters and uses `DETERMINISTIC_ZERO_INITIALIZATION`.

The only loss is SmoothL1 between predicted and teacher standardized coefficients. Pairwise geometry, tanh, sign, absolute-pair, classification, outfit ID, target appearance, target pose/camera, and teacher residual prediction inputs are absent. No A6 checkpoint or state dictionary is loaded.

At step 0 the standardized prediction is zero. De-standardization maps this to the frozen coefficient mean and the frozen basis maps it to the mean-garment endpoint. The required term remains **MEAN-GARMENT INITIAL PREDICTION**, not zero-residual prediction.

### B6 reference classifier hard lookup

Implementation: `scene.p0_candidate_adapters.B6ReferenceClassifierHardLookupAdapter`.

The adapter has 3,589 trainable parameters and policy `RANDOM_SEEDED_INITIALIZATION`. Reference-only F2 features produce five logits. `torch.argmax` selects the first maximum in frozen outfit order `O01,O02,O03,O04,O08`; the predicted class, not the ground-truth outfit, selects the frozen teacher-residual asset.

The outfit label appears only as the CrossEntropy loss target. It is absent from inference forward and hard lookup. Invalid class indices fail closed. A five-by-five confusion-matrix schema is frozen, but no confusion values or performance conclusions are produced by the dry-run.

### B7 F2 nearest-centroid hard lookup

Implementation: `scene.p0_candidate_adapters.B7F2NearestCentroidHardLookupAdapter`.

B7 has no trainable parameters and creates no candidate optimizer. Each of the four target-view folds constructs five outfit centroids from exactly the fold's three legal non-target reference views. The target condition, target RGB, and target mask are excluded. Outfit labels are used only for offline centroid grouping.

Distance is frozen as per-feature standardization followed by squared L2. Zero-variance dimensions use scale one. Ties use the registered seen-outfit order. The all-fold centroid manifest records conditions, reference RGB hashes, frozen feature hashes, normalization hashes, centroid hashes, queries, per-class distances, nearest outfit, and target-exclusion proof.

### M3/M4 fair matrix

Implementation: `scene.p0_candidate_adapters.P0ComplexCandidateAdapter` and the paired factory `build_m3_m4_candidate_adapters`.

For a given seed, M3 and M4 receive deep copies of the exact same randomly initialized complex trunk. Cross-seed trunk initialization varies. Both 234,771-parameter adapters use zero-initialized coefficient heads and therefore start at the same mean-garment endpoint.

M3 uses an unconstrained standardized coefficient and SmoothL1 only. M4 applies tanh and matches historical A5 supervision field-for-field: SmoothL1 weight 1.0, sign margin 0.8/weight 0.25, absolute-pair margin 1.5/weight 0.10, and diagnostic pairwise geometry with zero weight in the total. The only primary matrix change is Linear fusion to Complex fusion. Historical B5 remains `OFF_MATRIX_HISTORICAL_EVIDENCE`.

## Unified runner

Implementation: `tools/paper/p0_candidate_runner.py`.

Supported commands:

- `validate`: validate branch, manifest, immutable registries, frozen assets, formal output tree, previous deterministic-protocol artifacts, and CUDA.
- `plan`: show all 13 candidates, replicate/seed identity, trainability, planned 300-step budgets, isolated paths, and the false formal-authorization flag.
- `dry-run`: construct fresh candidates, create readiness optimizers only for the 12 trainable entries, perform a fixed legal reference-only forward, calculate step-0 loss without backward, reconstruct the basis endpoint or perform predicted hard lookup, and write an append-only smoke report.
- `status`: report implementation status, latest dry-run status, fingerprints, authorization, and failure reason.

There are no `train`, `resume`, `evaluate`, `aggregate`, or `export-paper` commands.

Smoke output is isolated under `AAAI27-P0-CANDIDATE-ADAPTERS-RUNNER-SMOKE/<method>/<seed-or-replicate>/attempt_xxx/`. A failed attempt is preserved and the next invocation allocates a new attempt. The runner never writes into the formal 51-run root or the previous P0 closure roots.

## Optimizer provenance

The candidate and legacy namespaces are separate. For readiness, each trainable adapter receives an Adam optimizer over all and only its `requires_grad` parameters (`lr=0.02`, no weight decay). B7 receives none. The candidate runner loads the sealed feature cache and basis directly, so it constructs no legacy/context optimizer.

All candidate, legacy, and scheduler step counts are zero. Candidate/frozen, optimizer/frozen, and candidate/legacy overlap counts are zero. No optimizer state is saved. Optimizer construction alone is not training.

## Evaluator compatibility

The output schema retains exactly 20 unique correct episodes and 80 unique cross-outfit swap tuples. Permutation, single-reference, two-reference dropout, zero replacement, and base replacement records are required. The existing 27-metric evaluator schema is not extended and the evaluator is not executed here.

Ours-v2 aggregates by deterministic replicate and must not receive an independent-random-initialization sample-standard-deviation narrative. B6, M3, and M4 aggregate by random seed; B7 is one fixed result. O07 remains outside the seen macro and may appear only as a held-out diagnostic.

## Frozen state

The formal registry, reviewer-risk registry, old seed failure audit, frozen manifest, 4,127-file formal output tree, previous deterministic-protocol output, and historical shared modules remain read-only. `PAPER_FINAL` remains zero.
