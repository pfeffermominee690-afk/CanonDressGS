# Subject00 / Subject02 Schema Comparison V2 — 2026-07-23

## Result

`PASS`. Subject00 is loader-compatible with the clean V2 prospective runtime baseline. The only data-contract difference that changes execution is subject00's 296 official image/mask gaps; the frozen loader already excludes a pair unless both files exist, and the new 60,000-entry availability manifest makes that behavior auditable.

The source baseline is `b1d304614b5a9322035b45ebeb7fa0166adaa362`, classified `SUBJECT02_SOURCE_SNAPSHOT_CREATED_WITH_LIMITATIONS`. It is a **clean frozen prospective runtime baseline**, not a historically exact source recovery for the subject02 formal MMLP-Human run.

## Contract comparison

| Item | Classification | Evidence |
|---|---|---|
| Images | `EXACT_COMPATIBLE` | `images/camXX/########.jpg` |
| Masks | `EXACT_COMPATIBLE` | `masks/camXX/########.jpg` |
| Camera IDs | `EXACT_COMPATIBLE` | 24 JSON entries, insertion order exactly `cam00` through `cam23` |
| Frame IDs | `EXACT_COMPATIBLE` | zero-based, eight-digit filenames; 2,500 SMPL frames |
| Resolution | `EXACT_COMPATIBLE` | `imgSize=[1330,1150]` (W,H) |
| Calibration | `EXACT_COMPATIBLE` | same keys and shapes: K/R 9, T 3, distCoeff 5, imgSize 2 |
| Extrinsics | `EXACT_COMPATIBLE` | loader constructs world-to-camera matrix `[R|T]` |
| SMPL parameters | `EXACT_COMPATIBLE` | identical NPZ keys and per-frame dimensions |
| Frame mapping | `EXACT_COMPATIBLE` | `smpl_params[*][frame_id]`; camera skips do not compact frame IDs |
| Missing pairs | `COMPATIBLE_WITH_AVAILABILITY_MANIFEST` | 296 paired gaps in subject00; none are image-only or mask-only |
| Canonical asset assumptions | `REQUIRES_DETERMINISTIC_PREPROCESSING` | beta-dependent template, LBS and initial points are absent |
| Loader paths | `EXACT_COMPATIBLE` | `ThumanDataset` paths match both datasets |

## Missing-file audit

- The theoretical grid is 24 × 2,500 = 60,000 pairs.
- Both official lists contain 296 unique, well-formed entries and zero duplicates.
- The image and mask lists identify the same 296 `(frame_id,camera_id)` pairs.
- Filesystem scanning independently finds the same 296 absent images and masks.
- Valid pairs: 59,704; invalid pairs: 296; image-only gaps: 0; mask-only gaps: 0; unknown list entries: 0.
- Manifest SHA256: `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`.

## Loader implications

`ThumanDataset.__init__` pre-enumerates only pairs for which both image and mask exist. `__getitem__` therefore cannot select an official gap while the raw tree remains immutable. It indexes pose data with the original frame ID and calibration with the original camera ID; no renumbering or pose shift occurs. A direct invalid-pair constructor produced length zero, the full dataset produced exactly 59,704 indices, and the enumerated set exactly matched the manifest.

One unrelated frozen-code limitation remains: `resize_image` references an undefined variable when `image_scaling != 1`. This preflight and its draft config freeze `image_scaling=1`; the V2 runtime closure was not modified.

Machine record: `paper_protocol/second_identity/subject00_subject02_schema_comparison_v2.json`.
