# Full Dressable Dataset Contract v1

The normative machine-readable contract is `schemas/canondressgs_full_dataset_v1.schema.json`; semantic requirements below are additional and mandatory.

## Layout

Recommended assets are `ROOT/Oxx/{rgb,foreground_mask,clothing_mask}/<condition_id>.png`. RGB must be PNG `RGB`; masks must be PNG `L`, with foreground at least as inclusive as clothing. The JSON manifest can use relative or absolute paths.

Top-level fields are `schema_version`, `dataset_kind`, `conditions`, `splits`, and `outfits`. Official data uses `dataset_kind=official_12x200`; Gate 4 conversion uses `regression` and cannot satisfy official evidence claims.

Each condition contains the source frame/camera IDs; pose/global transform; K/w2c/c2w; output dimensions/background; explicit conventions; and SHA256 fingerprints of poses, cameras and reviewed selection sources. `export_condition_index.py` rejects a selection other than 200 unique IDs and unresolved/duplicate source IDs.

For 1536×1024 assets, K must originate from the selected source camera. The exporter scales source K by `sx=1536/source_width`, `sy=1024/source_height` (entire first and second rows respectively). Source dimensions are mandatory CLI arguments. No fixed 1536×1024 K is defined by this contract.

## Loader returns

Training returns references `[K,3,H,W]`, both reference masks `[K,1,H,W]`, pose `[K,165]`, R `[K,3,3]`, Th `[K,3]`, K `[K,3,3]`, w2c `[K,4,4]`, camera dictionaries, target RGB/masks and matching target state/camera, optional teacher descriptor, and outfit metadata.

Inference returns the same references plus target pose/R/Th/K/w2c/camera. It rejects target RGB/masks, teacher/anchor targets and any cloth/clothing embedding. `outfit_id` is metadata/routing only and is forbidden as model conditioning.

## Commands

```powershell
python tools/export_condition_index.py --poses-json POSES.json --cameras-json CAMERAS.json --selection-json conditions_200.json --source-width SOURCE_W --source-height SOURCE_H --width 1536 --height 1024 --output condition_index_v1.json
python tools/build_full_dressable_manifest.py --condition-index condition_index_v1.json --dataset-root DATASET_ROOT --output full_dataset_v1.json
python tools/check_full_dressable_dataset.py --manifest full_dataset_v1.json --projection-json actual_posed_anchor_uv.json --real-model-smoke-command python REAL_SMOKE_TOOL.py
```
