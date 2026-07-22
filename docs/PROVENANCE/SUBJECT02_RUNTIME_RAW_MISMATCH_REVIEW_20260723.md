# Subject02 Runtime Raw Mismatch Review

Date: 2026-07-23

The frozen 15-file closure was compared among the cloud dirty candidate, Windows dirty candidate, and failed V1 clean snapshot. Every file records raw and normalized SHA256, CRLF/LF counts, final-newline state, BOM state, actual mode, executable bit, symlink state, Git blob, byte counts, parser result, and pairwise classification in `subject02_runtime_raw_mismatch_review.json`.

## Result

- `LINE_ENDING_ONLY`: 14
- `FILE_MODE_ONLY`: 1
- `RUNTIME_RELEVANT_CONTENT_DIFFERENCE`: 0
- `UNKNOWN_DIFFERENCE`: 0
- Semantic parse failures: 0

Cloud-vs-Windows and cloud-vs-failed differences are limited to newline representation and non-runtime file-mode metadata. The normalized closure SHA256 is `6999a663a826a2db8ae093ea3e560aace526ef66e3f2fa404cbaffb520e22c89` in all three semantic candidates.

The cloud Linux dirty runtime closure is designated `AUTHORITATIVE_PROSPECTIVE_RUNTIME_CANDIDATE` because the formal run occurred on cloud Linux, its checkpoint strict-load and inference contract pass, the normalized source closure is shared, and the cloud bytes can be preserved exactly. This designation does not mean `HISTORICALLY_EXACT_TRAINING_SOURCE`.

V1 remains a `FAILED_PROVENANCE_SNAPSHOT_CANDIDATE`. Its newline-normalized raw tree cannot be relabeled as the accepted baseline even though its single canary passed.
