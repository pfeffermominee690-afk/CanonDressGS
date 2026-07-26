# AAAI27 Unified Paper Smoke Acceptance

Task: `AAAI27-UNIFIED-PAPER-SMOKE-ACCEPTANCE-001`

Artifact class: **SMOKE ONLY — NOT PAPER RESULTS** (`NOT_FOR_PAPER_NUMBERS`)

## Frozen scope

This acceptance run is isolated from the 51 executable entries in
`paper_protocol/experiment_registry.yaml`. The formal registry, its 300-step budgets,
seeds 0/1/2, frozen asset manifest, formal paper output root, and `artifacts/paper_ready`
remain read-only. A8 remains historical evidence and non-executable.

The checked-in smoke registry contains exactly ten seed-0 entries: B0 through B5,
Ours, A1 rank 3, A5 legacy endpoint supervision, and A7 single reference. B0/B1/B2
create neither optimizer nor training checkpoint. Six trainable adapter checks run five
optimizer steps. Ours runs 0→10, persists complete state, resumes 10→20, and is compared
against a separate uninterrupted 0→20 control.

## Engineering acceptance contract

- All execution uses `CUDA_VISIBLE_DEVICES=-1`; the physical cloud GPU is recorded but
  inaccessible to the test process.
- The 19 frozen assets must pass their existing fingerprints before an attempt is
  created.
- Every attempt is append-only and uses the isolated
  `AAAI27-SEEN-OUTFIT-PAPER-SMOKE` root.
- Every output carries `SMOKE ONLY — NOT PAPER RESULTS` and
  `NOT_FOR_PAPER_NUMBERS` provenance.
- Prediction forward receives reference-derived features only. Target RGB/mask,
  teacher targets, outfit IDs, O07, and O06 do not enter prediction forward.
- Base Gaussians, image backbone, MMLP-Human, explicit basis, and renderer witnesses
  must retain identical hashes and zero gradients.
- The evaluator must read persisted raw episode JSONL, require 20 unique seen
  episodes and 80 cross-outfit swaps, and return all 27 registered metrics as finite
  numbers or explicit N/A.
- Formal aggregation of one seed must reject with
  `MISSING_REQUIRED_PAPER_SEEDS`. Smoke aggregation is limited to
  episode → outfit macro → seed and may not synthesize three-seed statistics.
- Tables 1–4 and Figures 1–6 are exported only under `artifacts/smoke_only` and carry
  visible smoke watermarks.
- Final `SMOKE_ACCEPTED` requires actual inspection of all eight preregistered visual
  categories. Numeric or visual quality is not a smoke pass threshold.

## Reproducible commands

The exact deployed commit and absolute cloud paths are written into every attempt's
`provenance/run_provenance.json`. The authorized run command is:

```bash
CUDA_VISIBLE_DEVICES=-1 /root/autodl-tmp/conda_envs/mmlphuman/bin/python \
  tools/paper/run_unified_paper_smoke_acceptance.py run-all \
  --repo-root /root/autodl-tmp/canondressgs_work/worktrees/canondressgs_aaai27_unified_smoke_acceptance \
  --output-root /root/autodl-tmp/canondressgs_work/outputs/pipeline_full \
  --asset-root /root/autodl-tmp/canondressgs_work/outputs \
  --artifact-root /root/autodl-tmp/canondressgs_work/outputs/pipeline_full/AAAI27-SEEN-OUTFIT-PAPER-SMOKE/artifacts/smoke_only \
  --expected-commit RUN_COMMIT
```

## Final evidence

The run-specific results, attempt paths, exact resume differences, evaluator counts,
table/figure manifests, visual observations, registry fingerprints, and final decision
are frozen in `paper_protocol/smoke/smoke_acceptance_report.json` after the cloud run.

## Executed result

- Run commit: `8b820ba4d05e0ca4317c0c076054cc0275b5799e`.
- Export-only mapping fix commit: `5edc6d24d028ad128d6e7e825f58249d07e9e5af`.
- Environment: Python 3.10.20, PyTorch 2.4.1+cu121, physical RTX 4090;
  `CUDA_VISIBLE_DEVICES=-1`, so all tests and smoke optimization executed on CPU.
- Frozen assets: 19/19 PASS.
- Tests at the run commit: 166/166 PASS (141 prior + 25 new). The export-source
  placeholder check increased the smoke suite to 26/26 PASS.
- Smoke matrix: 10/10 `SMOKE_ACCEPTED`; optimizer steps were
  B0/B1/B2=0, B3/B4/B5/A1/A5/A7=5, Ours=20.
- All eight registered adapter types executed a real PyTorch forward. All trainable
  parameter groups received finite nonzero gradients. Target-forward leakage was
  false. Frozen component gradient count and maximum parameter change were both zero.
- Ours stopped at step 10, persisted the complete checkpoint, resumed through step
  20 without repeating a step, and matched the uninterrupted control with maximum
  absolute difference 0 for model, optimizer, RNG, scheduler, schedule position,
  LayerNorm/Linear parameters, coefficients, residual, RGB, and alpha.
- Every evaluator run consumed persisted raw outputs and enforced 20 unique seen
  episodes plus 80 cross-outfit swaps. All 270 metric slots were finite or legal N/A
  (261 finite, nine B0 N/A).
- Formal one-seed aggregation rejected with `MISSING_REQUIRED_PAPER_SEEDS`. Smoke
  aggregation produced only episode → five-outfit macro → seed-0 rows.
- Tables 1–4 and Figures 1–6 were exported under `artifacts/smoke_only`. Export
  attempt_001 is preserved; export attempt_002 is final because it explicitly marks
  unrun Figure 4/5 columns `NOT_RUN` instead of repeating the only executed source.
- Eight preregistered visual categories and all six exported figures were actually
  opened. They were complete, non-black, correctly ordered, and visibly watermarked.
- Formal registry Linux SHA256 remained
  `1a01de7db498164089e8f55a8bca90d909a219ec8afb06da4e97bb9e0cb3ab84`; Git blob
  remained `62071643367f46da1f01bf0de4e9a599552e7e4d`.
- Historical formal-output metadata fingerprint remained
  `108c8f2a7e327cd76a508c16e1797cd220d169cbff44091fc9cfefbd2911c74c`.

Final decision: **SMOKE PASS**. This is engineering authorization to launch the
frozen 51-run paper matrix; it is not a paper result and authorizes no scientific
claim. The next unique task is `LAUNCH_FROZEN_PAPER_EXPERIMENT_BATCHES`.
