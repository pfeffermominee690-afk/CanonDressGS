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

Current document state before execution: `PENDING_SMOKE_EXECUTION`.
