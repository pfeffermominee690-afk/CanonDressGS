# Subject00 Template Determinism — 2026-07-23

## Result

`PASS`. Two clean processes independently generated the same subject-specific SMPL-X body-surface fallback. No subject02 template was read as an input or copied.

- Shape source: subject00 `smpl_params.npz:betas[0]`.
- Stored betas shape: one deterministic subject vector; finite and constant by construction.
- Shape-vector SHA256 (float32 raw array): `1624c77963cb379496b04631a8c7a11ab618d5978466b04ceea157d382b2f140`.
- Canonical transform: identity global orientation, zero translation, body-pose entries 2/5 at ±25 degrees, flat-hand/model-default facial parameters.
- Canonical-pose SHA256: `eaa1b9ce87f9db58a4f74dc56315d9d7c07b3c6bcac9afa6e131cd399343241f`.
- Model: neutral SMPL-X SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.

## Run comparison

| Field | run_a | run_b | Comparison |
|---|---|---|---|
| Vertices | 10,475 float32 | 10,475 float32 | bitwise exact |
| Faces | 20,908 int64 | 20,908 int64 | bitwise exact |
| Vertex-array SHA | `1f924c7e…9ad97` | `1f924c7e…9ad97` | exact |
| Face-array SHA | `2cb81d8e…ce2b` | `2cb81d8e…ce2b` | exact |
| Deterministic PLY SHA | `f10a3b51…0031` | `f10a3b51…0031` | exact |
| Vertex max/mean abs | — | — | 0 / 0 |
| Bounding box max abs | — | — | 0 |

Both runs report finite vertices/normals, legal face indices, zero degenerate faces, zero duplicate vertices/faces, three connected components, 32 boundary edges, 31,346 manifold edges, zero nonmanifold edges and `watertight=false`. Surface area is `1.7719255686` in SMPL-X native squared units. The three-component/nonwatertight facts are recorded characteristics of this native fallback, not altered or repaired.

## Protocol-gate erratum

Before the formal runs, an initial audit process produced the same sealed arrays but was marked failed because the new validator had added an unauthorized “connected components must equal one” hard gate. The formal derived plan required this field to be recorded, not forced to one. That output was preserved under `attempt_history/protocol_gate_erratum_001_run_a`; the audit-only gate was corrected, while shape, pose, model, dtype, topology and writer remained unchanged. The two formal clean runs were then executed from scratch.

Runtime classification: `BODY_SURFACE_FALLBACK_RUNTIME_COMPATIBLE`. This template passed, but it was not published because the downstream LBS gates failed.
