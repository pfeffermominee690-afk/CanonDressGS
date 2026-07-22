# Subject00 Template Contract Limitation — 2026-07-23

## Permanent boundary

`LOOSE_CLOTHING_TEMPLATE_AVAILABLE=false`

`TEMPLATE_MODE=SUBJECT_SPECIFIC_SMPLX_BODY_SURFACE_FALLBACK`

The subject00 mesh is neither a recovered loose-clothing template nor scan-derived clothing geometry. It does not reproduce subject02's placement operation or initialization contract.

## Contract comparison

| Field | subject00 candidate | subject02 formal reference |
|---|---|---|
| Template type | subject-specific SMPL-X body surface | loose-clothing template |
| Vertices | 10,475 | 96,380 |
| Faces | 20,908 | 192,744 |
| Vertex ratio | 0.10868 | 1.0 |
| Face ratio | 0.10848 | 1.0 |
| LBS | candidate `[128,128,128,55]`, blocked | archived `[128,128,128,55]` |
| Body model | neutral SMPL-X, subject00 beta | neutral SMPL-X, subject02 beta |
| Gaussian target count | 200,000 (not initialized here) | 200,000 |
| Loader | ThumanDataset compatible | ThumanDataset |
| Cameras | 24 raw, strict 18/6 split | 24; historical camera18 view-transductive |
| Poses | strict 1,130/125 with 1,245 buffer | historical configured frame contract |
| Coordinate system | model-native, no axis/scale conversion | model-native runtime |
| Render contract | not tested; LBS precondition failed | archived V2 canary PASS |
| Expected fidelity | body-surface portability baseline | clothing-aware initialization |
| Provenance | fully generated in this attempt | placement/generation history not sealed |

Static runtime inspection finds no hard-coded 96,380/192,744 assumption. Open3D loads arbitrary triangle topology; Gaussian initialization samples a configured number of surface points; LBS uses the model's 55 joints; checkpoint architecture is tied to Gaussian/control/feature counts rather than template vertex count. Therefore the template alone is classified `BODY_SURFACE_FALLBACK_RUNTIME_COMPATIBLE`.

Even if a later LBS repair and short canary pass, this experiment first tests **pipeline portability**. It does not establish cross-identity garment generalization, identical-initialization fairness, template-independent performance or loose-clothing reconstruction parity.
