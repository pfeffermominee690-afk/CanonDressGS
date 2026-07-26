# Subject00 Attempt 001 Native-Landscape Registration Audit

- Task: `AAAI27-SUBJECT00-ATTEMPT001-NATIVE-LANDSCAPE-REGISTRATION-AUDIT-001`
- Protocol SHA-256: `f8bda763ee845d7e3fd6ece32d7b4a7e8150308cd69df0aa80f15cd69aa53e76`
- Outputs audited: `48/48`
- Dominant exact resolution: `1349x1166` (`34` outputs, `20/24` raw cells)
- Machine registration pass candidates: `37`
- Machine-pass cell coverage: `21/24`
- Complete reliable single-resolution cohort: `false`
- Registration feasibility: `PARTIALLY_FEASIBLE`
- Dataset salvage classification: `PARTIAL_SALVAGE_TARGETED_RERUN_CANDIDATE`
- Final classification: `SUBJECT00_NATIVE_LANDSCAPE_PARTIAL_SALVAGE_TARGETED_GAPS_IDENTIFIED`

The audit used frozen official person masks expanded by 23 px, SIFT background matching, deterministic RANSAC, and the preregistered similarity/structure gates. Machine-pass candidates remain pending human visual review. No output is accepted or written as a Teacher target.

Human review must use the request-level panels and preserve all final decision fields as null until the user adjudicates them. `PAPER_FINAL=false`.
