# AAAI27 Reference-Conditioned Dual-Support Controller Design

**RESEARCH METHOD CANDIDATE — NOT PAPER FINAL**

## Scope and source

This design task starts from `research/dual-support-all-pair-evaluation-20260722` at exact source HEAD `d802f427f1e9c23595e1bdb4f10135f7de2c3f08`, whose sealed conclusion is `DUAL_SUPPORT_ALL_PAIR_PASS`. It closes only the controller/interface design gap. It does not train a controller, run a formal render, aggregate formal metrics, alter a teacher endpoint, modify the sealed Dual-Support renderer, or create a paper-final result.

The run branch is `research/reference-conditioned-dual-support-controller-20260722`. The protocol was committed before implementation. Its frozen training plan was resolved unambiguously from `tools/paper/p0_formal_candidate_runtime.py`: each optimizer step is a batch of five outfit episodes in order `O01,O02,O03,O04,O08`; target-view folds cycle `cond_000000,cond_000318,cond_000017,cond_000347`, with step 1 using `cond_000000`.

## Controller adapter

The adapter is `scene/reference_conditioned_dual_support_controller.py`. It uses the legal B6/Ours-v2 reference feature contract:

`reference RGB + clothing mask → frozen F2 spatial map → per-reference masked mean/max → set mean/max → LayerNorm(512) → Linear(512,5) → softmax(T=1)`.

The fixed logit/outfit order is `O01,O02,O03,O04,O08`. The adapter contains 3,589 trainable parameters: 1,024 LayerNorm parameters plus 2,565 linear-head parameters. Seeds 0, 1, and 2 use fresh `RANDOM_SEEDED_INITIALIZATION`. No B6 checkpoint, Ours-v2 checkpoint, teacher residual, or teacher-derived classifier weights are loaded.

The prediction `forward` signature is exactly `(reference_f2, reference_valid)`. The frozen F2 backbone remains external and read-only. Garment labels and soft distributions are loss targets only. Target RGB, mask, pose, camera, garment ID, teacher endpoint ID, teacher residual, ground-truth mixture weight, target render, and loss target cannot enter prediction forward. Target pose/camera are reserved for downstream deformation/render after support selection.

## Frozen dataset and soft targets

`tools/paper/build_dual_support_controller_dataset.py` generated the manifest from the frozen `aaai_gate_28_manifest.json` image/mask checksum index. For each of 10 unordered garment pairs and four target-view folds, it records eight sets in fixed order: `AAA`, the three AAB minority positions, the three ABB minority positions, and `BBB`.

- Pair/fold query sets: 320.
- Pure pair/fold query sets: 80.
- Mixed AAB/ABB query sets: 240.
- Retained formal pure endpoint episodes: 20.
- Total manifest records: 340.
- Unique logical inputs: 260.
- Duplicate records: 80, all consistent with the canonical formal-pure soft target.
- Inconsistent duplicate labels: 0.
- Target-view/reference overlap: 0.
- AAB position counts: 40/40/40; ABB position counts: 40/40/40.

Every record persists its target-view fold, three legal reference condition IDs, ordered garment labels, five-class target distribution, assignment type/position, image paths and SHA256 values, clothing-mask paths and SHA256 values, target exclusion, logical-input hash, and duplicate provenance. No best assignment position is selected.

The only loss is soft-label cross entropy

`L_soft = -Σ_g y_g log softmax(logits)_g`.

Targets are one-hot for AAA/BBB, `(2/3,1/3)` for AAB, and `(1/3,2/3)` for ABB. Mixed labels are never collapsed to hard labels. Temperature is fixed at 1.0. There is no SmoothL1, teacher-residual/render loss, entropy regularization, pair-specific loss, label smoothing, temperature tuning, auxiliary loss, or result-conditioned weight.

## Stable selection and sealed runtime interface

Probabilities are sorted descending with frozen outfit order as the stable tie-break. Let `p1,p2` be the first two probabilities, `m2=p1+p2`, `w2=p2/m2`, and `w1=1-w2`.

- If `m2 < 0.90`, mode is `SINGLE_ENDPOINT`, reason is `LOW_TOP2_MASS`, only top-1 is resolved, and its runtime weight is 1.
- Otherwise, if `w2 < 0.10`, mode is `SINGLE_ENDPOINT`, reason is `LOW_SECONDARY_WEIGHT`, only top-1 is resolved, and its runtime weight is 1.
- Otherwise mode is `DUAL_SUPPORT`; two immutable endpoint branches receive normalized opacity weights `w1,w2`.

The low-top2-mass guard has explicit precedence when both predicates hold. A single-endpoint decision never constructs branch 2. Unsupported multimodality never constructs top-3 support.

Prediction and endpoint resolution are separated. `stable_top2_selection` emits outfit identities and confidence only; `construct_dual_support_runtime` subsequently maps them through the frozen five-entry endpoint bank. The request schema records selected paths, branch indices, endpoint identities, opacity weights, mode, fallback reason, stable ranking, renderer contract, and `target_forward_leakage=0`.

Each runtime branch is marked `IMMUTABLE_ENDPOINT_NO_INTERPOLATION`. The adapter cannot perform geometry interpolation, geometry averaging, four-dimensional basis interpolation, pair-specific thresholds, pair-specific ghosting repair, ground-truth pair selection, or manual alpha selection. It targets the already sealed `SEALED_DUAL_SUPPORT_OPACITY_GATE` renderer contract without changing that renderer.

## Step-0 dry run and determinism

The cloud smoke root is `/root/autodl-tmp/canondressgs_work/outputs/REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-SMOKE/`, containing `seed_0/`, `seed_1/`, `seed_2/`, and `audits/`. For every seed, two independent fresh Python processes built the model and ran the same fixed cases:

- five pure endpoint examples;
- three AAB assignments;
- three ABB assignments;
- a low-top2-mass synthetic guard;
- a low-secondary-weight synthetic guard;
- a frozen-order tie guard.

For all three seeds, initialization state, fixed-case logits, distributions, step-0 loss, and top-2 choices were bitwise exact across the two fresh processes. The three initialization hashes were unique across seeds. The actual F2 feature cache was used. Because this is untrained random initialization, the observed class choices are interface evidence only and are not reported as endpoint accuracy or scientific performance.

Allowed operations were forward, probability/selection computation, soft step-0 loss computation under `torch.no_grad`, and runtime request construction. Formal rendering and metric computation were both zero.

One readiness optimizer was instantiated as `torch.optim.Adam` over all and only the 3,589 candidate parameters. Candidate optimizer zero-grad calls=0, steps=0, and state saves=0. No legacy optimizer or scheduler was created. Backward calls=0, scheduler steps=0, checkpoint writes=0, candidate/frozen overlap=0, and candidate/legacy overlap=0.

## Frozen later training and evaluation contracts

Later training is frozen as Adam, learning rate 0.02, weight decay 0, betas `(0.9,0.999)`, epsilon `1e-8`, constant scheduler, 300 steps, L2 gradient clipping at 5, and seeds 0/1/2. This task executed none of those steps.

The later evaluator schema preserves the pure task (20 correct episodes, 80 swaps, single-reference/dropout/color counterfactual, endpoint parity, fallback rate), the mixed task (320 queries per seed, 960 total; pair accuracy, ordering, calibration, activation/fallback, LPIPS, silhouette IoU, boundary F-score, ghosting, identity contamination), and perturbations (grayscale, hue, blur, mask morphology, mode-switching stability). No formal evaluator was run here.

The preregistered PASS thresholds are fixed before training: pure top-1 `20/20`; pure single rate at least 95%; mixed top-2 pair accuracy at least 90%; AAB/ABB ordering at least 85%; mixed dual activation at least 80%; macro LPIPS better than FULL_LINEAR; silhouette IoU no lower than FULL_LINEAR; severe artifact count reduced at least 50%; identity contamination maximum 0; grade-3 ghosting at most 1/10 pairs; no ground-truth ID/manual alpha; frozen mutation 0.

## Verification and governance

The required controller/dual-support/B6-boundary/determinism selection completed with `29 passed`. A broader compatibility invocation completed `70 passed` and exposed one unrelated environment dependency failure: the legacy M4/A5 test imports `gsplat`, which is absent from this cloud Python environment. That failure did not execute or change controller science and is retained here rather than hidden. Python compilation, JSON/YAML parsing, archive/file existence checks, and `git diff --check` are separate final archive gates.

Formal output and sealed all-pair output tree metadata were captured before and after the dry run and were exactly equal. Canonical-LF SHA256 values for the six sealed repository archive files also matched. Teacher mutation=0, renderer mutation=0, formal output mutation=0, all-pair archive mutation=0, training=0, backward=0, optimizer step=0, scheduler step=0, checkpoint write=0, and `PAPER_FINAL=0`.

Final design classifications are:

- `CONTROLLER_ADAPTER=PASS`
- `SOFT_TARGET_DATASET=PASS`
- `TOP2_SELECTION=PASS`
- `SINGLE_ENDPOINT_FALLBACK=PASS`
- `DUAL_SUPPORT_RUNTIME_INTERFACE=PASS`
- `FORWARD_BOUNDARY=PASS`
- `NO_FORMAL_TRAINING_GATE=PASS`

The next task is `TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER`. It has not been started.
