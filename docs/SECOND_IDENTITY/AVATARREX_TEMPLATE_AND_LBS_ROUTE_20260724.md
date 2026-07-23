# AvatarReX Template And LBS Route

Task: `AAAI27-AVATARREX-BASE-AVATAR-PREFLIGHT-001`

## Template

The primary route is the official lbn1 preprocessed archive indexed by `PREPROCESSED_DATASET.md` (Drive file `1RDM3v5P4XF6Sp88EusDvokw-yHg6Je0C`, viewer metadata title `avatarrex_lbn1.7z`, `2271523272` bytes). No download occurred. Archive SHA, members, template topology, vertex/face counts, position maps, and canonical LBS presence remain unknown and must be frozen during controlled acquisition.

If that asset fails validation, the secondary route is the official AnimatableGaussians reconstruction pipeline: PointInterpolant/canonical LBS preparation, the lbn1 template config, `main_template.py`, and position-map generation. It is a 150000-iteration learned reconstruction and was not run. The deterministic SMPL-X body surface is canary-only because MMLP explicitly recommends a loose-clothing template for `avatarrex_lbn1`; it is not equivalent evidence.

Expected accepted MMLP path: `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1/gaussian/template.ply`.

## LBS

The recommended route is MMLP-Human's own `script/gen_weight_volume.py` with the pinned neutral SMPL-X model and pinned PointInterpolant build. It produces the exact runtime schema `grid/bbox_min/bbox_max/grid_dims/grid_resolution` at `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1/gaussian/lbs_weights_grid.npz` and queries weights at the actual initialized Gaussian positions.

The official AnimatableGaussians `cano_weight_volume.npz` schema is not directly interchangeable. Surface-attached or closest-surface weights require a new runtime/checkpoint contract and are not the selected primary route. Current MMLP training has a fixed initialized Gaussian point set; no clone/split/densification path was found. Checkpoints serialize materialized per-Gaussian weights, so any future topology change requires explicit rebinding and schema versioning.

No template or LBS asset was generated in this task.
