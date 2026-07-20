# Unified paper smoke artifacts

**SMOKE ONLY — NOT PAPER RESULTS** (`NOT_FOR_PAPER_NUMBERS`)

- The top-level `tables/` and `figures/` are the preserved first export.
- `exports/attempt_002/` is the final accepted export. It was generated with zero
  optimizer steps after manual inspection found that the first export repeated the
  sole A1/A5 smoke image in columns whose variants were not run.
- In attempt_002, Figure 4 contains real smoke imagery only in the A1 K=3 column and
  Figure 5 only in the executed A5 column. Other columns are explicit `NOT_RUN`
  placeholders.
- `visual_review/` contains the eight images that were actually opened for manual
  acceptance.

These files must never be copied to `artifacts/paper_ready` or used as paper numbers.
