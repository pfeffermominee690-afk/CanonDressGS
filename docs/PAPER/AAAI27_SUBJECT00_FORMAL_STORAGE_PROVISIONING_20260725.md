# Subject00 Formal Storage Provisioning Audit (2026-07-25)

## Scope and Decision

Task `AAAI27-SUBJECT00-FORMAL-STORAGE-PROVISION-AND-RUN-001` audited the canonical output filesystem before any scientific attempt. The canonical root, rejected alias, `attempt_001`, and `attempt_002` were all absent. Scientific attempt count remained zero.

The formal contract was reread from `paper_protocol/second_identity/subject00_formal_resource_budget.json` (SHA256 `93d10b0f651f30f6c08951e8108347fca430b0d563224c8a61d01a90efd4e43d`). It requires `32212254720` free bytes. The task safety target is `40802189312` bytes.

The canonical parent is on `/dev/md0`, mounted at `/root/autodl-tmp` as XFS. The immutable pre-cleanup audit observed `14716096512` free bytes. No authorized cleanup on that filesystem was available. The recheck therefore remained `14716096512` bytes, with a contract margin of `-17496158208` and a safety margin of `-26086092800` bytes.

Final classification: `SUBJECT00_FORMAL_STORAGE_PROVISIONING_INSUFFICIENT`.

## Inventory

Largest non-overlapping target-volume entries:

| Path | Bytes | Classification | Automatic action |
| --- | ---: | --- | --- |
| `/root/autodl-tmp/canondressgs_work/data/subject02/images` | 35376238592 | Dataset | Prohibited |
| `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets` | 29884325888 | Derived assets and provenance | Prohibited |
| `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00` | 26707615744 | Raw dataset | Prohibited |
| `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full` | 23738216448 | Scientific output | Prohibited |
| `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1` | 19262648320 | Dataset | Prohibited |
| `/root/autodl-tmp/canondressgs_work/outputs/LOO-BASIS-ADAPTATION-001` | 14057574400 | Scientific attempts | Prohibited |
| `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/downloads` | 12569784320 | Dataset source archive | Prohibited |
| `/root/autodl-tmp/conda_envs/mmlphuman` | 8146964480 | Formal environment | Prohibited |
| `/root/autodl-tmp/conda_envs/lerobot-v21-diffusion` | 6880169984 | Conda environment | Prohibited |
| `/root/autodl-tmp/conda_envs/lerobot-diffusion` | 6608756736 | Conda environment | Prohibited |
| `/root/autodl-tmp/canondressgs_work/outputs/COEFFICIENT-HEADROOM-001` | 6169628672 | Scientific output | Prohibited |
| `/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k` | 5838528512 | Checkpoints and output | Prohibited |
| `/root/autodl-tmp/canondressgs_work/worktrees` | 4346535936 | 87 Git worktrees | Prohibited |
| `/root/autodl-tmp/pip_cache` | 3142000640 | Apparently stale pip HTTP cache | User adjudication required |

The full ranked top-20 adjudication list is in `paper_protocol/reviewer_risk/subject00_formal_storage_cleanup_plan.json`.

Ten files exceed 1 GiB. Every one is a dataset archive, model weight, checkpoint, or optimizer-state file and is protected. The largest is `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/downloads/avatarrex_lbn1.7z` at `12569755256` bytes.

The link audit found 4581 symbolic links, dominated by protected Conda environments, and 26280 hardlinked file paths, dominated by Git objects shared across protected worktrees. No link target was selected for cleanup. `lsof +L1` could not run because `lsof` is not installed. `fuser` is also unavailable. Process inspection found no formal training process, and the GPU compute-process query was empty.

## Cleanup Adjudication

No files were deleted.

The exact allowlisted pip cache `/root/.cache/pip` contains `3421220864` bytes, but it is on the container `overlay` filesystem. `/root/.nv/ComputeCache` (`1368064` bytes), `/tmp` (`428531712` bytes), and the Conda tarball cache are also on `overlay`. The exact allowed Conda dry run reported 73 tarballs totaling 110.1 MB plus one index cache, but deleting them cannot increase `/dev/md0` free space. Torch extensions and Triton caches were absent.

The target-volume path `/root/autodl-tmp/pip_cache` contains `3142000640` bytes and is not the cache reported by the formal environment's `pip cache dir`, which is `/root/.cache/pip`. It is not a Git worktree, contains no symlink or hardlink candidate, and its exact path was not found in the audited repository. It was nevertheless not deleted because the task's automatic allowlist names `/root/.cache/pip`, not this separate path. Even explicit deletion would leave the contract short by `14354157568` bytes at the audited baseline.

Five target-volume Python cache directories totaling 204800 allocated bytes were found outside the broad output, dataset, environment, model, and provenance exclusions. One is inside the protected `AnimatableGaussians` Git repository; four are under an unknown research patch-staging directory. They were not deleted because protection takes precedence and their provenance was not disposable.

## Gate and Execution State

| Field | Value |
| --- | ---: |
| Before cleanup free bytes | 14716096512 |
| Deleted cache bytes | 0 |
| After cleanup free bytes | 14716096512 |
| Contract required bytes | 32212254720 |
| Safety target bytes | 40802189312 |
| Contract margin bytes | -17496158208 |
| Safety margin bytes | -26086092800 |

The contract and safety target both failed. Per the sealed governance, the 22-item formal preflight was not rerun, no `EXECUTION_HEAD` was frozen, no canonical output root was created, and no optimizer step, checkpoint, metric, visual review, scientific classification, or downstream manifest was produced.

## Required User Action

Provision at least `26086092800` additional free bytes on `/dev/md0` relative to this recheck before another formal run. Moving protected data to separately verified durable storage is preferable to deletion. The only low-risk deletion candidate identified in this task is `/root/autodl-tmp/pip_cache`, subject to explicit path authorization, but it is insufficient by itself.

`PAPER_FINAL` remains `false`.
