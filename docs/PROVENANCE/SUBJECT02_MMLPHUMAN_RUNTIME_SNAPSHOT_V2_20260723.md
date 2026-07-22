# Subject02 MMLPHuman Runtime Snapshot V2

Date: 2026-07-23
Classification: `SUBJECT02_SOURCE_SNAPSHOT_CREATED_WITH_LIMITATIONS`

## Decision

The branch `research/mmlphuman-subject02-runtime-snapshot-v2-20260723` is accepted as the clean, frozen prospective runtime baseline for future subject00 work. It is not evidence that this tree is the historically exact source used for the subject02 800k training run.

The authoritative prospective candidate was the read-only cloud Linux dirty worktree at base commit `6668509284c85fcb0f93cd7365ec8f39390ff251`. The first V2 source-closure commit is `3382078fbdd77bba8bb9df54452647456a4c9b00`.

## Frozen runtime closure

- Files: 15
- Candidate raw-byte closure SHA256: `dafe40e736b37a7023ab5db06dd3f82a8c8797c475f496619491861281b99415`
- V2 fresh-checkout raw-byte closure SHA256: `dafe40e736b37a7023ab5db06dd3f82a8c8797c475f496619491861281b99415`
- Candidate/V2 LF-normalized closure SHA256: `6999a663a826a2db8ae093ea3e560aace526ef66e3f2fa404cbaffb520e22c89`
- Git blob closure SHA256: `7894519f91a634acd8e61e1dd7289b14008324a07fd125591276d0432f460d25`
- Raw-byte parity: 15/15
- Git executable-bit parity: 15/15
- Symlink-contract parity: 15/15

The snapshot uses path-specific `.gitattributes` entries with `-text` and Git operations with `core.autocrlf=false`. The attributes exist only to prevent newline conversion; they are not part of the runtime-closure hash and do not change Python semantics.

Git does not encode group/other write permission bits. The source worktree's non-executable `0666` files therefore check out as non-executable `0644`; executable-bit and symlink semantics are exact, but full POSIX permission-bit identity is not claimed.

## Checkpoint and runtime regression

- Formal checkpoint SHA256: `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- Strict load: PASS
- Top-level keys: 42
- `checkpoint_version`: 2
- Iteration: 100000
- Gaussians: 200000
- Frozen frames: 0 through 49, camera 18
- Frozen frame-manifest SHA256: `9e17a2a0eeb4010f9c4b74d6030b219c2463c796186a74483e1da2bdab339731`
- Dirty-vs-V2 RGB: 50/50 exact
- Dirty-vs-V2 alpha: 50/50 exact
- Dirty-vs-V2 depth auxiliary: 50/50 exact
- Dirty-vs-V2 visible-Gaussian counts: 50/50 exact
- V2 fresh-process determinism: PASS across two processes
- Aggregate RGB SHA256: `d87acf29e44f361482556e8d26b2de465ccf3ea329b885b45153f3090734a6e0`
- Aggregate alpha SHA256: `abd9acca882e523a170f37e2b3c35bf8b0c90b6b014b97084fb206b0780fc57a`

The archived scalar regression passed the tolerances frozen before the earlier execution: L1 absolute tolerance `5e-5`, PSNR absolute tolerance `0.05 dB`. Observed L1 was `0.007752127721905708` and PSNR was `24.582090644836427`. Their absolute deltas from archived evidence were `1.5944018959998876e-05` and `0.016353340148924644`. Historical RGB/alpha tensors were not archived, so exact parity to the historical formal render is not claimed.

## Template and LBS

- Repo/data template SHA256: `d8050cacca15a118fb1eee753fce1f6933fb1889cac758aa37d0962867dd735f`
- Template: 96,380 vertices, 192,744 faces
- LBS SHA256: `61af875b0c9c9f8c8c5bab678e14bc262105435270be4384cd6fc465633cc5a7`
- LBS grid contract: `[128, 128, 128, 55]`, finite
- SMPLX forward-vertex SHA256: `3702056324f5e3e90f0a973a314f5c0de3c72fb99474a307609e32968da1197b`
- Dirty-vs-V2 template/LBS/body-forward contract: PASS

The template placement operation and exact LBS launch command were not historically recorded. The external PointInterpolant binary is a 169,474,576-byte build artifact with SHA256 `ff516f19b4a6735ec95b8d5734c61dfb4908052d7495dc812f9d29159219ca2b`; it is not added to Git.

## Safety accounting

Training steps, backward calls, optimizer creation, optimizer steps, scheduler steps, checkpoint writes, subject02 data mutations, formal-output mutations, and subject00 mutations were all zero. `PAPER_FINAL=0`.

The original Windows and cloud dirty repositories were treated as read-only. Their before/after evidence is retained outside Git under `manual_review_attempt_002`.

## Validation caveat

The unmodified raw sources contain pre-existing trailing spaces. Default `git diff --check` therefore reports them. Rewriting those bytes would invalidate the raw snapshot. The external evidence retains that failure log and a scoped `core.whitespace=-trailing-space,cr-at-eol` conflict-marker check that passes while preserving raw bytes. This exception is a provenance limitation, not a claim that ordinary whitespace quality checks passed.

The capture tool also embeds abbreviated object IDs in binary diff evidence. Addition of V2 objects made Git expand those display prefixes from 7 to 8 characters without changing the original cloud worktree. A process-only `core.abbrev=7` recheck reproduced the exact before fingerprint `00a6226705972ac875fb819775c1e5bb566c7321882853fd8e8b472a6734867f`; no repository configuration was changed.

## Future use

Accepted future baseline: true. Future subject00 reports must cite this branch, the consuming checkout HEAD, the raw closure hash above, this limited-provenance classification, and the continuing historical subject02 limitation.
