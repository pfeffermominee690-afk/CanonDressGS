# CanonDressGS Multi-Source Evidence Freeze Report (2026-07-26)

## Decision

`CANONDRESSGS_MULTI_SOURCE_EVIDENCE_BUNDLE_FROZEN_FOR_EXTERNAL_BASELINE_AUDIT`

Eight requested immutable commits exist locally. Each task-matched primary summary is unique, all frozen classifications and requested numeric checks match, and the committed paper snapshot has no conflict with the frozen claim boundaries. The dirty paper worktree was excluded from extraction.

## Remote Availability

- Origin live query: `PASS`; all eight source commits are reachable after publishing the freeze branch base: `True`.
- Cloud live query: unavailable because `canondress-cloud` could not be resolved from this machine. Historical local cloud remote-tracking refs reach seven of eight commits; Pure Endpoint is not confirmed. This is reported as provenance availability metadata, not rewritten as PASS.

## Scope

The bundle does not audit external methods, access external baseline repositories, train models, run renderer inference, mutate data, modify the paper, merge scientific branches, or cherry-pick scientific commits. It exports 53 text files directly from Git objects.

## Scientific Adjudication

- Pure Endpoint: `PURE_ENDPOINT_CORE_METHOD_SUPPORTED`; 24 runs, 7,200 optimizer steps, 144 checkpoints, exact clean endpoint realization, and the hard-lookup relation retained.
- Geometry Causal: geometry main effect `0.9959405426885113`; sufficient and necessary in 10/10 pairs under the registered decomposition.
- Dual-Support: `DUAL_SUPPORT_ALL_PAIR_PASS`; all ten pairs, improved aggregate LPIPS/IoU/Boundary F, zero identity contamination, externally specified pair and weight.
- Headroom: `TEACHER_SPAN_AT_LOCAL_OPTIMUM`; refinement improves 0/5 garments and is negative scope evidence.
- LOO: `LOO_BASIS_CAPACITY_LIMITED`; capacity pass 0/5 and garment success 0/5. The exact macro LPIPS values use the explicitly recorded cross-source paper-restructure registry.

## Paper Consistency

The committed paper source discloses clean hard-lookup functional equivalence, does not claim Teacher as an upper bound, limits positive evidence to subject02 and five seen garments, keeps Dual-Support external, and preserves Headroom/LOO as negative results. Conflict count: 0.

`PAPER_FINAL=false`. Next task: `RUN_EXTERNAL_BASELINE_FEASIBILITY_AUDIT_FROM_FROZEN_MULTI_SOURCE_BUNDLE`.
