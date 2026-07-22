# CanonDressGS Project Status

## 1. Canonical Source of Truth

- Current latest sealed experiment HEAD: `371d812864614cd561e33edfe3f6c043b38415d2` on `paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722`.
- Current active research HEAD: `67a8b08be7c4a07b55a07c8169a093b6c4cfe1d3` on `research/continuous-control-causal-attribution-20260722`.
- Current paper branch: `paper/aaai27-manuscript-stable-sections-20260721` at `1a5a55fee0f6a3344debdc63f93bad60944f9e96`. These are stable sections, not a frozen final paper.
- Current teaser branch: `paper/aaai27-real-teaser-figure-20260722` at `1f77b4ed220c50de8671e9db6233155f0e0dc41d`; formal-asset audit passed, but author signoff is still required.
- Current data preparation status: subject00 archive exists locally at `E:\data_pre\thuman4_second_identity_staging\downloads\subject00.7z` and is `DOWNLOADED_LOCAL_PENDING_VERIFICATION`. Upload and extraction are not claimed.

The detailed sources of truth are [WORKTREE_REGISTRY.yaml](WORKTREE_REGISTRY.yaml), [MILESTONE_REGISTRY.yaml](MILESTONE_REGISTRY.yaml), [ARTIFACT_REGISTRY.yaml](ARTIFACT_REGISTRY.yaml), and [DATASET_REGISTRY.yaml](DATASET_REGISTRY.yaml).

## 2. Active Mainline Task

- Branch: `research/continuous-control-causal-attribution-20260722`
- HEAD: `67a8b08be7c4a07b55a07c8169a093b6c4cfe1d3`
- Task: `AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001`
- Current stage: endpoint-anchored factorial protocol is frozen and the worktree is clean. `attempt_001` is preserved as `FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH` with zero render, metric, evaluation, training, backward, optimizer, and checkpoint counts. The next append-only attempt is `attempt_002`; no causal final summary exists in the local branch.
- Blockers: the actual `attempt_002` output and visual/factorial adjudication are absent from the scanned branch; the cloud host was unreachable during the control-center scan, so the output root could not be inspected live.
- Expected next decision: after one complete preregistered causal attempt, select exactly one micro-pilot from the frozen decision rule. No micro-pilot is authorized by the present evidence.

## 3. Parallel Tasks

| Track | Branch / location | Status | Immediate boundary |
|---|---|---|---|
| causal attribution | `research/continuous-control-causal-attribution-20260722` | `ACTIVE` | Continue only as append-only `attempt_002`; do not reinterpret the midpoint-degenerate historical result. |
| real teaser | `paper/aaai27-real-teaser-figure-20260722` | `DOCUMENTATION_ACTIVE` | Asset audit passed; visual record remains `MANUAL_REVIEW_REQUIRED` and `PAPER_FINAL=0`. |
| paper rewrite | `paper/aaai27-manuscript-stable-sections-20260721` | sealed stable sections; rewrite gate blocked | Do not freeze method claims until P0 and causal adjudication are complete. |
| subject00 dataset preparation | `E:\data_pre\thuman4_second_identity_staging` | `DATA_STAGING` | Archive is downloaded; integrity, upload, and extraction remain unverified. |
| second identity planning | no experiment branch registered | blocked planning | Requires verified subject00 archive plus a documented camera/pose/data contract before evaluation. |

## 4. Sealed Milestones

| Class | Branch | HEAD | Evidence boundary |
|---|---|---|---|
| `SEALED_CANONICAL` | `paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722` | `371d812864614cd561e33edfe3f6c043b38415d2` | P0 color/spatial/soft-control archive; not paper final. |
| `SEALED_CANONICAL` | `paper/aaai27-p0-formal-candidate-runs-20260721` | `32e6058ea64078f5ccded6f86e8bffe3cb81b30e` | Formal candidate archive; origin matches, cached cloud ref is older. |
| `SEALED_CANONICAL` | `paper/aaai27-deterministic-initialization-protocol-20260721` | `42a28b386f6f32e23e5e16680408820efd8a65c7` | Deterministic initialization protocol. |
| `SEALED_CANONICAL` | `paper/aaai27-p0-candidate-adapters-runner-20260721` | `d75c4fa5a426fe90314eba08e94509940ecafb23` | Candidate adapters and runner. |
| `SEALED_CANONICAL` | `paper/aaai27-p0-evaluation-protocol-repair-20260721` | `8578fb3143dd916ff7e42240e7508a29c784d995` | Evaluation-protocol repair. |
| `SEALED_CANONICAL` | `paper/aaai27-manuscript-stable-sections-20260721` | `1a5a55fee0f6a3344debdc63f93bad60944f9e96` | Stable manuscript sections; paper claims remain unfrozen. |
| `SEALED_CANONICAL` | `paper/aaai27-frozen-experiment-batches-20260720` | `c60d61a67a87eb35d5debca7eb111778b637c252` | Frozen seen-outfit batches. |
| `SEALED_CANONICAL` | `paper/aaai27-seen-outfit-protocol-20260720` | `0336e763201ff081710837b13eb49d8b08f5ba2d` | Seen-outfit protocol. |
| `SEALED_CANONICAL` | `paper/aaai27-unified-runner-evaluator-20260720` | `1d596c218fab7a099412e5ccc9869ef6ee64a083` | Unified runner/evaluator. |
| `SEALED_CANONICAL` | `paper/aaai27-unified-smoke-acceptance-20260720` | `16a48bbcbc28e0050e5b2f1777974775e025c9f7` | Smoke acceptance evidence. |
| `SEALED_CANONICAL` | `research/explicit-gaussian-residual-basis-20260720` | `4ff14cd52b5b204908b47173915b3d66ff51f58b` | Explicit-basis ladder. |
| `SEALED_CANONICAL` | `research/multi-outfit-explicit-basis-20260720` | `4baf319843f2e8f4faf08043dcd40a590a61d992` | Five seen teachers and rank-4 basis; O07 remains a limitation. |
| `SEALED_CANONICAL` | `research/reference-coefficient-supervision-calibration-20260720` | `28f2b3358f28b64c3cb35b64f130013ee519129a` | CS-PASS/SmoothL1 calibration evidence. |
| `SEALED_HISTORICAL` | `research/continuous-control-artifact-root-cause-20260722` | `68623c36eee70c0b41aefb480f8f85e668eae201` | Engineering archive only: channel attribution has midpoint degeneracy and is not a final causal conclusion. |
| `SEALED_HISTORICAL` | `paper/aaai27-p0-reviewer-risk-closure-20260721` | `71dd86c1c01e00e44e8135a45ce29db788141fe0` | Preserved failure audit. |
| `SEALED_HISTORICAL` | `paper/aaai27-reviewer-risk-adjudication-20260721` | `4bd47b7b667d1ade230b6b0ae3d611f550548c84` | Historical evidence adjudication. |
| `SEALED_HISTORICAL` | `paper/aaai27-reviewer-risk-paper-materials-20260721` | `0dd4cbcc049fdc15616701b78f2f08d625c63b65` | Historical paper materials. |
| `SEALED_HISTORICAL` | `research/reference-basis-coefficient-fusion-20260720` | `fbb162ad58fa7810f78a222f7c669d696b00fefd` | Fusion diagnosis retained as historical evidence. |

The authoritative complete enumeration, including failed, superseded, staging, documentation, and unknown worktrees, is in `WORKTREE_REGISTRY.yaml`.

## 5. Decision Gates

1. causal attribution complete → one uniquely selected micro-pilot;
2. micro-pilot `PASS` → formal final-method adjudication;
3. final method frozen → second-identity evaluation;
4. expanded dataset verified → novel-view / novel-pose canary;
5. canary evidence plus independent review → only then reconsider the currently unconfirmed generalization claims.

## 6. Blocked Tasks

| Task | Blocking reason | Release condition |
|---|---|---|
| unique micro-pilot | No completed endpoint-anchored factorial attribution | Final causal summary satisfies the frozen classification rule and selects exactly one next task. |
| final paper method | Continuous-control failure source and P0 adjudication are not frozen | Causal result plus explicit final-method decision; `PAPER_FINAL` must not be inferred automatically. |
| paper claim freeze | Final method and generalization evidence are incomplete | Update `CLAIM_STATUS.md` only from new sealed evidence. |
| second identity evaluation | subject00 archive not verified/extracted and no identity contract | Integrity verification, safe extraction, dataset manifest, pose/camera contract, and formal branch. |
| novel view / novel pose claims | No expanded verified dataset/canary | Run separately preregistered canaries after the dataset gate. |
| cloud parity | `canondress-cloud` DNS failed during read-only scan | Restore host resolution and repeat `git ls-remote` plus artifact-root verification. |

## 7. Next Three Tasks

1. Complete append-only causal-attribution `attempt_002` under its frozen no-training protocol and archive a final summary.
2. Apply the causal decision rule to run exactly one unique micro-pilot, then adjudicate and freeze the final method only if it passes.
3. Verify the subject00 archive and establish the second-identity dataset/pose/camera contract before any second-identity evaluation.

## 8. Risks

- Hard lookup risk: teacher endpoints can be reached without a demonstrated soft-control advantage.
- Continuous-control artifacts: patching, mottle, cloud, edge scatter, and silhouette discontinuity remain visible.
- Single identity: confirmed evidence is centered on subject02.
- View-transductive risk: fixed evaluation views do not establish novel-view generalization.
- Second dataset missing: subject00 is only a downloaded, unverified archive.
- Paper claims not frozen: stable prose exists, but final method and generalization claims are not paper-final.
- Cloud verification gap: cached cloud refs are not equivalent to live remote verification.

## 9. Last Update

- Date: `2026-07-22` (Asia/Shanghai).
- Control-center scan base HEAD: `371d812864614cd561e33edfe3f6c043b38415d2`; the final control-center commit necessarily advances the branch after this self-referential snapshot.
- Update sources: `git worktree list --porcelain`, per-worktree Git metadata, live `origin` `ls-remote`, cached cloud refs, formal Markdown reports, summary JSON, figure manifests, frozen asset manifests, and local dataset path checks.
- Cloud caveat: live `cloud` `ls-remote` failed because `canondress-cloud` could not be resolved. No cloud-clean or cloud-upload claim is made.
