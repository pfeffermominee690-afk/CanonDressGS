# Subject00 Formal Storage Migration Adjudication

Task: `AAAI27-SUBJECT00-FORMAL-STORAGE-MIGRATION-ADJUDICATION-001`

Date: 2026-07-25

## Adjudication

Final classification: `SUBJECT00_STORAGE_MIGRATION_PLAN_READY`.

The exact authorized pip cache was safely removed, but the formal storage gate remains blocked. No formal attempt, `EXECUTION_HEAD`, optimizer step, checkpoint, metric, scientific result, or scientific classification was created. The recommended next action is a separately authorized migration of AvatarReX raw data and its audit reports to the existing private `E:` staging tree, followed by byte-exact verification and only then retirement of the three exact cloud source paths. This task did not execute that migration or retire those sources.

`NEXT_TASK=EXECUTE_USER_SELECTED_SUBJECT00_STORAGE_RECLAMATION_PLAN`

## Source And Hard Guards

| Field | Value |
| --- | --- |
| Source branch | `research/mmlphuman-subject00-formal-storage-provisioned-run-20260725` |
| Source HEAD | `f18eb0641afa2070ebeede3140f5c342c13944e9` |
| Audit branch | `research/mmlphuman-subject00-storage-migration-adjudication-20260725` |
| Windows worktree | `E:\model_train\canondressgs_subject00_storage_migration_adjudication` |
| Cloud worktree | `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_subject00_storage_migration_adjudication` |
| Canonical root | `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001` |
| Rejected alias | `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001` |
| Canonical root / rejected alias | both absent |
| `attempt_001` / `attempt_002` | both absent |
| Scientific attempt count | 0 |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB total, 0 MiB used, 0% utilization |
| GPU compute / active training processes | 0 / 0 |

The contract was reread from `paper_protocol/second_identity/subject00_formal_resource_budget.json`, SHA-256 `93d10b0f651f30f6c08951e8108347fca430b0d563224c8a61d01a90efd4e43d`. Its required free capacity remains `32212254720` bytes. The safety target was reread from `subject00_formal_storage_gate_recheck.json` and remains `40802189312` bytes. Neither value was modified.

## Exact Pip Cache Deletion

Target: `/root/autodl-tmp/pip_cache`

Before deletion it contained 847 files and 1469 child directories, occupied `3142000640` allocated bytes, and had `3139890084` apparent file bytes. Its only top-level entries were `http-v2` (`3141996544` allocated bytes) and `selfcheck` (`4096` allocated bytes). The canonical file-list metadata SHA-256 was `51a54f021f3954bd0286f89981c30397fa694eaeac077352a39ccd603efd9dde`. The mtime range was 2026-07-11 13:52:46.663172 through 15:41:32.548487 UTC.

| Required protection check | Result | Evidence |
| --- | --- | --- |
| Exact realpath | PASS | exact target equality |
| Not a symlink | PASS | zero root/descendant symlinks |
| No filesystem crossing | PASS | device 2304 throughout, no child mount |
| Outside Git/worktrees | PASS | no repository boundary or worktree prefix |
| Outside scientific output | PASS | independent cache root |
| Outside datasets | PASS | independent cache root |
| No checkpoint | PASS | no checkpoint names/extensions |
| No user source | PASS | no source extensions; sampled bodies were packages |
| No final/handoff/seal/manifest/Figure reference | PASS | protected-metadata and repository search found none |
| Pip cache content only | PASS | 423 pip HTTP metadata/body pairs plus one selfcheck file |

There were zero symlinks, hardlinks, or open descriptors in the complete `/proc/*/fd` scan. `lsof` and `fuser` are not installed. Only after all ten checks passed, the exact authorized command ran:

```bash
rm -rf --one-file-system /root/autodl-tmp/pip_cache
```

The path is absent. Free capacity moved from `14548701184` to `17690701824` bytes, exactly matching the `3142000640` allocated-byte inventory. No wildcard and no `find -delete` was used.

## Live Capacity After Cache Cleanup

| Quantity | Bytes |
| --- | ---: |
| Task-entry initial free capacity, before audit worktree overhead | 14588919808 |
| Free capacity immediately before pip deletion | 14548701184 |
| Reclaimed pip cache | 3142000640 |
| Observed free capacity after deletion | 17690701824 |
| Contract requirement | 32212254720 |
| Contract shortfall | 14521552896 |
| Safety target | 40802189312 |
| Safety shortfall | 23111487488 |
| 45 GiB recommended target | 48318382080 |
| Shortfall to 45 GiB | 30627680256 |

Training remains unauthorized.

## Exact Large Candidate Decisions

All five roots are on `/dev/md0` XFS, owned by `root:root`, outside Git payload tracking, contain no symlink/hardlink crossing, and had no open process descriptor. XFS birth times were not exposed; the JSON inventory records both mtime and ctime ranges without treating ctime as creation time.

| Class | Exact path | Allocated bytes | Files | Child dirs | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| B | `/root/autodl-tmp/canondressgs_work/data/subject02/images` | 35376238592 | 74640 | 24 | Current paper/Figure evidence; do not move/delete |
| A | `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets` | 29884325888 | 1634 | 123 | Required Subject00 formal assets; do not move/delete |
| A | `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00` | 26707615744 | 119412 | 50 | Required raw Subject00; do not move/delete |
| C | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full` | 23738216448 | 27306 | 3344 | Sealed attempts/checkpoints/renders/provenance; do not move/delete |
| D | `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1` | 19262648320 | 60834 | 48 | Private Windows migration allowed only after full verification |

The exact path inventory records metadata-list hashes, mtime/ctime ranges, tracked and external references, Windows/cloud copy checks, rebuildability, cost, and license boundaries. Subject00 raw and derived roots have confirmed formal/provenance references. `pipeline_full` contains internal manifests, closures, checkpoints, selected renders and Figure-source material. AvatarReX is not a Subject00 dependency and has no current paper/Figure reference, but its manifests, final summary and provenance reports remain binding.

## Pipeline Output

`pipeline_full` has `23674179275` apparent file bytes:

| Role | Files | Apparent bytes |
| --- | ---: | ---: |
| Checkpoints/tensors | 380 | 16983954025 |
| Scientific renders/images | 24444 | 5202527305 |
| Contracts/manifests/reports/metadata | 2280 | 291973803 |
| Other scientific payload | 146 | 1195429467 |
| Logs | 56 | 294675 |

Its 40 first-level entries sum to `23738212352` allocated bytes; the remaining `4096` bytes are the root directory block. The pipeline audit enumerates all 40 entries individually. A content hash audit found 1219 duplicate-image groups and `2518641032` logical bytes beyond one copy, but these are deliberate attempt/step/oracle/Figure variants, not disposable cache. No temporary render-cache directory was found.

Safe pipeline migratable subset: `0` bytes.

Safe pipeline explicit-delete subset: `0` bytes.

The requested 30 GiB pipeline target is impossible without violating class C provenance protection.

## AvatarReX

The complete staging root occupies `31963389952` allocated bytes:

| Component | Allocated bytes | Detail |
| --- | ---: | --- |
| Extracted `avatarrex_lbn1` | 19262648320 | 30416 RGB, 30416 masks, calibration and SMPL parameters |
| Source archive | 12569784320 | logical size `12569755256`, SHA-256 `531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1` |
| Reports | 130957312 | 19 integrity, manifest, audit, fingerprint and provenance files |

There are no derived templates, LBS assets, generated assets, caches, incomplete downloads, nested archives, checkpoints, symlinks, hardlinks, or zero-byte extracted files. The extracted content-tree SHA-256 is `00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15`; its path-size tree SHA-256 is `a016e63be8472649d05d6cde7c4e7bfc5da74d828715df03e1e611ee17683aaa`.

The exact archive already exists at `E:\data_pre\avatarrex_second_dataset_staging\downloads\avatarrex_lbn1.7z` with the same logical bytes and SHA-256. The cloud archive is therefore class F, but its historical final summary says `retained=true`; any future deletion requires live rehashing and a new superseding deletion record. AvatarReX is restricted to non-commercial research use and cannot be redistributed, so only private Windows storage is accepted.

## Destination Audit

No Plan B cloud target exists:

* `/root/autodl-pub` exposes `19732321591296` free bytes but is read-only public `fuse.autofs` storage and is license-incompatible.
* `/root/autodl-nas`, `/data`, and `/workspace` do not exist.
* `/mnt` and `/` are the same ephemeral 30 GB overlay with `24823889920` free bytes.
* `/dev/shm` is 45 GB ephemeral RAM.
* `/dev/sda2` contributes only individual read-only NVIDIA file binds; no writable target directory is mounted.

Windows `E:` is healthy NTFS with `354805608448` total and `74059694080` free bytes by `Get-Volume`. The exact allowed empty-directory write test passed and the test path was removed. `E:\canondressgs_archive` does not exist. The recommended destination is the already existing private `E:\data_pre\avatarrex_second_dataset_staging` so the verified archive is not duplicated. WSL Ubuntu has `rsync 3.2.7`; its Windows OpenSSH bridge to `canondress-cloud` passed.

## Plans

### Plan A: Platform Expansion

The exact addition needed to reach 45 GiB is `30627680256` bytes. A practical rounded minimum is 32 GiB (`34359738368` bytes). The recommendation is at least 50 GiB (`53687091200` bytes), which would produce `71377793024` free bytes without moving existing data. Expansion has not occurred and must be observed through `df`, not inferred from a platform request.

### Plan B: Other Cloud Persistent Storage

Not generated because no real writable persistent target was found.

### Plan C: Private Windows Migration

Use WSL `rsync -rlt --partial --append-verify` over Windows OpenSSH. Transfer `avatarrex_lbn1` plus all 19 reports into an incoming directory on `E:`. The expected new destination allocation is `19393605632` bytes and expected file count is 60853. Recheck that `E:` has at least `27983540224` free bytes, which is transfer size plus an 8 GiB reserve.

After both rsync commands exit zero, independently verify file count, directory count, apparent bytes, complete content-tree SHA, path-size tree SHA, calibration SHA, SMPL SHA, every report SHA, and both archive hashes. Publish the incoming destination only after all checks pass. Cloud source retirement additionally requires a committed migration seal, a deletion record superseding `retained=true`, and explicit user authorization for each exact path. Failed or interrupted transfers resume; they never authorize source deletion.

Projected full staging release after verified migration and source retirement: `31963389952` bytes.

Projected free capacity: `49654091776` bytes.

### Plan D: Hybrid Alternative

A 20 GiB platform expansion plus future deletion of only the live-reverified duplicate cloud archive would produce `51735322624` free bytes. It is viable only after expansion is actually observed and the duplicate-archive deletion conditions are satisfied.

## Reclamation Combinations

| Combination | Total reclaimed from pre-cache baseline | Projected free | Contract margin | Safety margin | Safety + 5 GiB | 45 GiB |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1. Pip only | 3142000640 | 17690701824 | -14521552896 | -23111487488 | FAIL | FAIL |
| 2. Pip + pipeline safe subset | 3142000640 | 17690701824 | -14521552896 | -23111487488 | FAIL | FAIL |
| 3. Pip + Avatar extracted/reports | 22535606272 | 37084307456 | 4872052736 | -3717881856 | FAIL | FAIL |
| 4. Pipeline safe + full Avatar safe subset after realized pip deletion | 35105390592 | 49654091776 | 17441837056 | 8851902464 | PASS | PASS |
| 5. Recommended Plan C | 35105390592 | 49654091776 | 17441837056 | 8851902464 | PASS | PASS |

The recommended state exceeds safety plus 5 GiB by `3483193344` bytes and exceeds 45 GiB by `1335709696` bytes.

## Mutation Audit

Authorized mutations performed:

1. Created the audit branch and Windows/cloud worktrees from the exact source HEAD.
2. Deleted only `/root/autodl-tmp/pip_cache` after all protection checks passed.
3. Created and immediately removed only `E:\.subject00_storage_adjudication_write_test_20260725` as an empty write test.
4. Created the audit documents and JSON records in the audit branch.

Mutations not performed: large directory deletion, move, rename, compression, tar creation, upload, download, symlink or mount change, bind mount, training, attempt creation, `EXECUTION_HEAD` freeze, storage-gate modification, output-root creation, scientific-data modification, or paper-body modification.

## Protected Paths

Do not touch without a new explicit task:

* `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00`
* `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets`
* `/root/autodl-tmp/canondressgs_work/data/subject02/images`
* `/root/autodl-tmp/canondressgs_work/outputs`
* `/root/autodl-tmp/canondressgs_work/worktrees`
* `/root/autodl-tmp/canondressgs_work/git`
* `/root/autodl-tmp/conda_envs`
* all Figure Bank sources, sealed attempts, formal checkpoints, unique teacher endpoints, generation responses and provenance

The three AvatarReX source paths listed in Plan C also remain protected until a user-selected execution task satisfies every migration and deletion condition.
