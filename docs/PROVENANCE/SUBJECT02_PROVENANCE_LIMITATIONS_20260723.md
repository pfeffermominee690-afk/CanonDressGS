# Subject02 Provenance Limitations

Date: 2026-07-23

The highest permitted and assigned classification is `SUBJECT02_SOURCE_SNAPSHOT_CREATED_WITH_LIMITATIONS`.

## What is established

- V2 exactly preserves the current authoritative cloud candidate's 15 runtime-file bytes.
- V2 fresh checkouts reproduce the raw and LF-normalized closure hashes.
- No runtime-relevant or unknown mismatch was found.
- The formal checkpoint strictly loads and the frozen dirty-vs-clean 50-frame regression is exact.
- The archived scalar regression, fresh-process determinism, and template/LBS/body-forward contracts pass.
- V2 is suitable as a clean prospective baseline for new subject00 experiments.

## What is not established

- The historical subject02 800k training did not archive a source snapshot, source commit, dirty diff, runtime closure hash, or exact launch environment.
- V2 therefore cannot be called `SUBJECT02_SOURCE_PROVENANCE_EXACT` or the historically exact training source.
- Historical RGB/alpha render tensors are unavailable, so historical render exactness is not claimed.
- Template placement provenance and the exact LBS-generation launch command are unavailable.
- Full POSIX write-permission bits are not representable in Git; only executable-bit and symlink contracts are sealed.
- Raw sources contain pre-existing trailing spaces, so default `git diff --check` fails unless raw-preservation whitespace is scoped out.
- CUDA renderer exactness is established only in the frozen current cloud environment.

## Failed V1

Branch `research/mmlphuman-subject02-runtime-provenance-20260723`, HEAD `dce43c0e4b9f4f3e129b797589ce901824045382`, tree `c6baa60e5720fff909f60c3fc8c1770737ca3b1e` remains a preserved `FAILED_PROVENANCE_SNAPSHOT_CANDIDATE`. It must not be deleted, rewritten, force-pushed, or used as the accepted subject00 baseline.

No new level-A/B historical evidence was discovered. Reclassification above the limited-provenance level is prohibited.
