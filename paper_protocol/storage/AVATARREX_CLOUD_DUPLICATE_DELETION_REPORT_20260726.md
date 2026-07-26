# AvatarReX Cloud Duplicate Deletion Authorization

Task ID: `AAAI27-SUBJECT00-FORMAL-BASE-ARCHIVE-RECLAIM-AND-EXECUTION-001`

This report authorizes deletion of one verified cloud duplicate only:
`/root/autodl-tmp/avatarrex_lbn1.7z`.

Deletion is not yet executed in this pre-delete evidence commit.

## User Authorization

- Authorization token: `AUTHORIZE_DELETE_VERIFIED_CLOUD_AVATARREX_ARCHIVE_AND_START_SUBJECT00_FORMAL_BASE_101245`
- Authorized delete path: `/root/autodl-tmp/avatarrex_lbn1.7z`
- Exact deletion command draft: `rm -- /root/autodl-tmp/avatarrex_lbn1.7z`
- `deletion_authorized=true`
- `deletion_executed=false`

## Source Gate

- Required source branch: `research/subject00-formal-base-execution-preflight-20260726`
- Required source HEAD: `e5c98482160b2625a9ac6e8af388bc5350eced46`
- Source worktree: clean
- Origin sync: pass
- Cloud bare and cloud preflight worktree sync: pass
- Frozen preflight classification: `SUBJECT00_FORMAL_BASE_EXECUTION_CONTRACT_READY_PENDING_USER_AUTHORIZATION`
- Prior `TRAINING_AUTHORIZED=false`
- Prior `OPTIMIZER_STEPS=0`
- Prior `OUTPUT_ROOT_CREATED=false`

## Windows Mirror

- Primary mirror: `E:\canondressgs_archive\cloud_datasets\AvatarReX\payload\root\autodl-tmp\datasets\avatarrex_second_dataset_staging\downloads\avatarrex_lbn1.7z`
- Primary manifest: `E:\canondressgs_archive\cloud_datasets\AvatarReX\metadata\root\autodl-tmp\datasets\avatarrex_second_dataset_staging\downloads\avatarrex_lbn1.7z.archive\source_manifest.json`
- Secondary local copy: `E:\data_pre\avatarrex_second_dataset_staging\downloads\avatarrex_lbn1.7z`
- Expected bytes: `12569755256`
- Expected SHA-256: `531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1`
- Primary mirror result: pass, byte exact
- Secondary local copy result: pass, byte exact

## Cloud Duplicate

- Cloud host/container: `autodl-container-ef19489c10-464381bb`
- User: `root`
- Target path: `/root/autodl-tmp/avatarrex_lbn1.7z`
- Realpath: `/root/autodl-tmp/avatarrex_lbn1.7z`
- Actual bytes: `12569755256`
- Actual SHA-256: `531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1`
- 7z test: `PASS_EVERYTHING_IS_OK`
- Archive contents reported by `7z t`: 48 folders, 60834 files, uncompressed size `19135049684`, compressed size `12569755256`

## Active Use Gate

- Process scan: empty for `avatarrex`, `avatarrex_lbn1`, `PLAN_B_CANARY_001`, `7z`, and `7za`
- `lsof`: unavailable on the container
- `fuser`: unavailable on the container
- Staging observation window: `120` seconds
- `/root/autodl-tmp/datasets/avatarrex_lbn1_staging`: 12 files and `10467169` logical bytes before and after
- `/root/autodl-tmp/datasets/avatarrex_lbn1_staging/PLAN_B_CANARY_001`: 12 files and `10467130` logical bytes before and after
- `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging`: 1 file and `1310046` logical bytes before and after
- Result: `PASS_NO_ACTIVE_PROCESS_AND_STAGING_IMMUTABLE`

## Storage Projection

- Filesystem: `/dev/md0`, type `xfs`, mounted at `/root/autodl-tmp`
- Free bytes before deletion: `45871964160`
- Duplicate archive bytes: `12569755256`
- Projected free bytes after deletion: `58441719416`
- Projected peak write bytes: `14231374378`
- Formal storage gate: `32212254720`
- Operational margin bytes: `2147483648`
- Required projected minimum free bytes after deletion: `34359738368`
- Projected minimum free bytes after deletion: `44210345038`
- Result: `PASS_PROJECTED_AFTER_DELETION_EXCEEDS_REQUIRED_GATE`

## Restore Procedure

Future re-upload must restore from the byte-exact Windows mirror to
`/root/autodl-tmp/avatarrex_lbn1.7z`, then repeat:

- `stat -c '%s' -- /root/autodl-tmp/avatarrex_lbn1.7z`
- `sha256sum -- /root/autodl-tmp/avatarrex_lbn1.7z`
- `7z t -- /root/autodl-tmp/avatarrex_lbn1.7z`

The expected size is `12569755256` and the expected SHA-256 is
`531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1`.

## Prohibitions

This authorization does not allow deletion of the Windows mirror, staging
directories, other cloud files, any output root, dataset content, paper files,
or unrelated archives. It also does not allow full AvatarReX extraction, Full
Avatar O03, GS-VTON, Teacher Endpoint, or process termination.

Final classification:
`SUBJECT00_AVATARREX_CLOUD_DUPLICATE_DELETION_AUTHORIZED_PENDING_EXECUTION`

## Execution Update

- Authorization commit: `29d00f839ccdf383debcabb19ecfbf5b72af012d`
- Final realpath before delete: `/root/autodl-tmp/avatarrex_lbn1.7z`
- Final bytes before delete: `12569755256`
- Final SHA-256 before delete: `531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1`
- Immediate process scan before delete: empty
- Executed command: `rm -- /root/autodl-tmp/avatarrex_lbn1.7z`
- Delete exit status: `0`
- Post-delete target check: absent
- Free bytes immediately before delete: `45871788032`
- Free bytes immediately after delete: `58441551872`
- Launch storage recheck free bytes: `58441494528`
- Projected minimum free bytes after deletion and peak write: `44210120150`
- Required projected minimum free bytes after deletion: `34359738368`
- Storage result: `PASS_AFTER_DELETION_STORAGE_RECHECK`
- Staging result: unchanged

Updated final classification:
`SUBJECT00_AVATARREX_CLOUD_DUPLICATE_DELETED_STORAGE_GATE_PASS`
