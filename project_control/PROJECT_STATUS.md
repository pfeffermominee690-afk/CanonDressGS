# CanonDressGS Project Status

## 1. Canonical Source of Truth

| Source | Branch / location | HEAD / fingerprint | Status |
|---|---|---|---|
| canonical P0 archive | `paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722` | `371d812864614cd561e33edfe3f6c043b38415d2` | `SEALED_CANONICAL` |
| causal attribution | `research/continuous-control-causal-attribution-20260722` | `8c43524b7ca0ee8b3c795dbe349466f30b29354f` | `SEALED_CANONICAL`; `GEOMETRY_MAIN_EFFECT` |
| dual-support micro-pilot | `research/geometry-dual-support-micro-pilot-20260722` | `b216acd02323c822a7aca12f424efed6ba2e0a81` | `SEALED_CANONICAL`; `DUAL_SUPPORT_MICRO_PILOT_PASS` |
| dual-support all-pair | `research/dual-support-all-pair-evaluation-20260722` | `d802f427f1e9c23595e1bdb4f10135f7de2c3f08` | `SEALED_CANONICAL`; `DUAL_SUPPORT_ALL_PAIR_PASS` |
| controller design | `research/reference-conditioned-dual-support-controller-20260722` | `174655ce6aabcae5d60ce45f3b4be319eb71e914` | `ACTIVE`; design/dry-run PASS, seal blocked by unverified cloud HEAD |
| manuscript stable sections | `paper/aaai27-manuscript-stable-sections-20260721` | `1a5a55fee0f6a3344debdc63f93bad60944f9e96` | stable sections only; not `PAPER_FINAL` |
| real teaser | `paper/aaai27-real-teaser-figure-20260722` | `1f77b4ed220c50de8671e9db6233155f0e0dc41d` | `DOCUMENTATION_ACTIVE`; author signoff pending |
| subject00 raw data | `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00/` | `2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b` | `SUBJECT00_CLOUD_DATA_READY` |
| subject00 MMLP-Human preflight | no branch/report discovered | none | `QUEUED_NOT_STARTED` |

Detailed machine-readable sources are [WORKTREE_REGISTRY.yaml](WORKTREE_REGISTRY.yaml), [MILESTONE_REGISTRY.yaml](MILESTONE_REGISTRY.yaml), [ARTIFACT_REGISTRY.yaml](ARTIFACT_REGISTRY.yaml), and [DATASET_REGISTRY.yaml](DATASET_REGISTRY.yaml).

## 2. Active Mainline Task

- Active task: `DESIGN_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER`.
- Branch/HEAD: `research/reference-conditioned-dual-support-controller-20260722` at `174655ce6aabcae5d60ce45f3b4be319eb71e914`.
- Evidence completed: all seven required design classifications are PASS; focused tests report `29 passed`, broader compatibility tests report `70 passed`; the worktree is clean; local equals live origin; training, backward, optimizer-step, checkpoint, frozen mutation, and `PAPER_FINAL` counts are zero.
- Open strict gate: live cloud branch equality is not verified because `canondress-cloud` DNS is unreachable. Therefore the milestone is not declared `SEALED_CANONICAL`.
- Current next task: `COMPLETE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_DESIGN`.
- Conditional next authorized task after sealing and explicit authorization: `TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER`.

No training/evaluation task was started by this refresh.

## 3. Parallel Tasks

| Track | Status | Boundary / next action |
|---|---|---|
| controller design/training mainline | design `ACTIVE`; training not started | verify cloud HEAD and seal design before any explicitly authorized formal training |
| subject00 MMLP-Human preflight | `QUEUED_NOT_STARTED` | raw dataset gate is satisfied; create a separate formal preflight with no training |
| real teaser | `DOCUMENTATION_ACTIVE` | source audit passed; author visual signoff remains |
| manuscript rewrite | stable sections sealed; refresh queued | update method/experiment structure from sealed evidence, without freezing final claims |
| related work / BibTeX | no current registered branch | queued documentation work; do not infer completion |
| novel-view/pose protocol | blocked | requires final controller plus second-identity avatar/garment benchmark |
| project-control maintenance | `DOCUMENTATION_ACTIVE` | this unique control center remains the only writer |

## 4. Sealed Milestones

| Milestone | HEAD | Evidence boundary |
|---|---|---|
| canonical P0 evaluation archive | `371d812864614cd561e33edfe3f6c043b38415d2` | deterministic P0 color/spatial/soft-control evidence; not paper final |
| continuous-control causal attribution | `8c43524b7ca0ee8b3c795dbe349466f30b29354f` | `GEOMETRY_MAIN_EFFECT`; G sufficiency/necessity `10/10`, interactions `0/10` |
| geometry dual-support micro-pilot | `b216acd02323c822a7aca12f424efed6ba2e0a81` | endpoint parity and `3/3` severe-artifact reduction; `DUAL_SUPPORT_MICRO_PILOT_PASS` |
| dual-support all-pair evaluation | `d802f427f1e9c23595e1bdb4f10135f7de2c3f08` | `10/10` pairs, `20/20` directions; `DUAL_SUPPORT_ALL_PAIR_PASS` within the seen-garment boundary |

The controller design is deliberately absent from this table until its live cloud equality gate can be checked. The older midpoint-degenerate archive at `68623c36eee70c0b41aefb480f8f85e668eae201` remains `SEALED_HISTORICAL`. The complete older milestone list is retained in `MILESTONE_REGISTRY.yaml`.

## 5. Decision Gates

1. Controller design strict PASS and cloud parity -> controller formal training/evaluation may be proposed, but requires explicit authorization.
2. Controller formal PASS -> final CanonDressGS method candidate.
3. subject00 preflight ready -> second-identity avatar training may be proposed.
4. final controller plus second identity -> strict novel-view/pose canary.
5. all required experiments -> paper result freeze.
6. paper result freeze -> explicit `PAPER_FINAL` decision.

## 6. Blocked Tasks

| Task | Blocking reason | Release condition |
|---|---|---|
| controller-design seal | cloud DNS failure | live `git ls-remote cloud` proves controller branch HEAD equals local/origin |
| controller formal training/evaluation | controller design not sealed; no explicit authorization | close the design gate and receive authorization |
| subject00 avatar training | MMLP-Human preflight not started; template/LBS/loader compatibility unknown | formal no-training preflight returns a training-ready classification |
| second-identity benchmark | avatar and garment benchmark not ready | ready avatar plus preregistered garment protocol |
| strict novel-view/pose | final controller and second-identity benchmark missing | both upstream gates complete |
| final abstract/results | no formal controller result | sealed controller training/evaluation and claim refresh |
| cleanup | no candidate meets every gate; remote verification incomplete | separately reviewed cleanup proposal after live remote checks |

## 7. Next Three Tasks

1. Restore read-only cloud reachability and complete the controller-design seal check.
2. After the seal and explicit authorization, train and evaluate the reference-conditioned dual-support controller.
3. Run `RUN_SECOND_IDENTITY_MMLPHUMAN_PREFLIGHT_WITHOUT_TRAINING` and publish the actual readiness classification.

## 8. Risks

- The controller may collapse into an endpoint classifier.
- Top-2 pair prediction may fail.
- Fallback thresholds may over-select single endpoints.
- Seven of ten dual-support pairs still show grade-2 ghosting/double outlines; two pairs retain severe alpha=`0.5` contamination.
- Active Gaussian count is approximately `2x`, render time approximately `2.625864x`, and peak VRAM approximately `2.007490x` the baseline.
- Confirmed evidence remains single-identity and closed-wardrobe.
- Current evaluation remains view-transductive rather than strict held-view/held-pose.
- A second public dataset is missing.
- Paper claims are not frozen and `PAPER_FINAL=0`.
- Cached cloud refs do not replace live cloud verification.

## 9. Last Update

- Date: `2026-07-23` (Asia/Shanghai).
- Refresh task: `CANONDRESSGS-PROJECT-CONTROL-REFRESH-002`.
- Inputs: live Git worktree/origin metadata, formal reports and summaries, artifact fingerprints, subject00 audit/manifests, and local path checks.
- Cloud result: `CLOUD_MIRROR_NOT_VERIFIED_DNS_UNREACHABLE`.
- This refresh performed zero training, backward calls, optimizer steps, evaluation runs, renders, metric recomputation, checkpoint writes, or `PAPER_FINAL` creation.
