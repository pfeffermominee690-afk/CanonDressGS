# Subject00 Archive Restore Guide

## Archive Roots

- AvatarReX: `E:\canondressgs_archive\cloud_datasets\AvatarReX`
- Checkpoints: `E:\canondressgs_archive\cloud_checkpoints`
- Master manifest: `E:\canondressgs_archive\archive_master_manifest.json`

Payloads preserve the original cloud path below `payload\root\autodl-tmp`. Metadata for each source is below the parallel `metadata\root\autodl-tmp` mapping.

## Required Restore Procedure

1. Read the item's `source_manifest.json` and `destination_verification.json`.
2. Confirm the manifest SHA recorded in `archive_master_manifest.json`.
3. Restore the payload to the manifest's exact `source_path` without renaming roots.
4. Preserve directory layout and the three checkpoint `checkpoints/last` symlinks.
5. Run `verify-manifest` against the restored path.
6. Require exact file count, logical bytes, and all per-file SHA256 values before use.

Tool:

```text
python tools/second_identity/execute_subject00_combined_reclamation.py verify-manifest \
  --manifest <source_manifest.json> \
  --destination <restored-source-path> \
  --output <restore-verification.json>
```

## Protected AvatarReX Exception

The following file is already restored and must remain on cloud:

`/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1/smpl_params.npz`

- Bytes: 1,309,966
- SHA256: `6ed3b3877d3895999e2636990bd417783328d695412b45d0679773e34b54613b`
- Classification: `SEALED_PROVENANCE_KEEP`

When restoring the full `avatarrex_lbn1` tree, first verify this existing file against the archive. Do not overwrite a mismatched protected copy. The full tree manifest contains the same expected bytes and SHA.

## Checkpoint Symlinks

- `diffusion_piper_50k/checkpoints/last -> 050000`
- `diffusion_30k/checkpoints/last -> 030000`
- `irregular_diffusion_30k/checkpoints/last -> 030000`

The byte manifest covers regular files. Verify the symlink mappings separately after restore.

## Restore Acceptance

A restore is accepted only when:

- no missing or extra regular files exist;
- file count and total logical bytes match;
- every regular-file SHA256 matches;
- expected symlinks match;
- the restored tree is not in active use during verification.
