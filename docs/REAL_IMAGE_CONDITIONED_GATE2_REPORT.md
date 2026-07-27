# CanonDressGS Gate 2 Real Image-conditioned Interface Report

Last updated: 2026-07-15

Status: **IMPLEMENTED-NOT-REAL-VERIFIED**

## 1. Scope and status

This patch implements the local interfaces required for the real Gate 2 chain:

```text
real episode with pose/Rh/Th
-> real MMLPHuman anchor deformation
-> per-reference official gsplat ED depth/alpha
-> depth-aware anchor feature sampling
-> canonical xyz-only offsets
-> transactional target pose/Rh/Th rendering
-> one-batch rendering loss and backward
```

- **REAL-VERIFIED (inherited, not rerun locally):** accepted 100k backbone, Gate 0 zero-offset/manual render, Gate 1 deformation/projection/depth visibility.
- **SYNTHETIC-VERIFIED:** Rh/Th parsing, state transaction, base-anchor selection rules, depth argument propagation, teacherless forward, target transaction, xyz-only gating, gradients and regressions.
- **IMPLEMENTED-NOT-REAL-VERIFIED:** real checkpoint loading through this new training path, 10k-anchor adapter construction, K-view ED rendering, target gsplat render and one-batch backward.
- **Not claimed:** Gate 2 has not passed until the cloud command in Section 11 succeeds and its artifacts are archived.

## 2. Modified and added files

### Modified

| File | Change |
|---|---|
| `scene/dressable_dataset.py` | Strict Rh/Th schema, inline/NPY loading, episode tensors, validation and fingerprint coverage |
| `scene/image_conditioned_dressable_model.py` | Canonical anchor buffer, depth/alpha/tolerance propagation, projection output and target state transaction |
| `scene/dressable_gaussian_model.py` | Explicit xyz/scaling/opacity channel gates |
| `train_dressable.py` | Base-anchor selection, fingerprint validation, real adapter factory, K-view ED geometry, teacherless checkpoint/evaluation support |
| `tools/build_synthetic_image_conditioned_manifest.py` | Explicit synthetic inline Rh/Th |

### Added

| File | Purpose |
|---|---|
| `utils/mmlphuman_state_utils.py` | Shared pose/Rh/Th/cache context manager |
| `configs/canon_dress_gs_mvp_real.yaml` | Strict xyz-only real MVP template |
| `tools/check_real_image_conditioned_interfaces.py` | 21 CPU synthetic/mock Gate 2 interface checks |
| `tools/check_real_image_conditioned_one_batch.py` | Cloud real checkpoint one-batch acceptance entry |
| `docs/REAL_IMAGE_CONDITIONED_GATE2_REPORT.md` | This report |

Backups of modified pre-existing files are under `backups/gate2_image_conditioned_patch_20260715/`.

## 3. Rh/Th frame and episode contract

Each strict frame now requires both `Rh` and `Th`. Each value may be a JSON inline array or a `.npy` path:

```text
Rh: float Tensor[3,3]
Th: float Tensor[3]
reference_Rh: float Tensor[K,3,3]
reference_Th: float Tensor[K,3]
target_Rh: float Tensor[3,3]
target_Th: float Tensor[3]
```

Missing only one transform always fails. Missing both transforms also fails by default. Legacy synthetic fallback requires the explicit constructor/config option `allow_missing_rh_th=true`; the real MVP config fixes it to `false`. Shape, floating dtype, finite values and K consistency are validated. The manifest fingerprint includes resolved Rh/Th numeric contents, so changing a sidecar changes the fingerprint.

The real MVP config has no `image_height` or `image_width`; RGB, masks and the original camera K remain at native resolution.

## 4. Canonical anchor source

`resolve_image_conditioned_canonical_anchors()` applies this order:

1. With a required real base, use `base_model.xyz_vt.detach().clone()`.
2. Validate finite `Tensor[A,3]` and require `A=10000`.
3. If a teacher exists, require identical shape, ordering and values with `atol=1e-6, rtol=0`.
4. A teacher is optional when the real base exists.
5. Without a real base, synthetic/teacher mode may use `dataset.anchor_xyz`.
6. If neither source exists, fail explicitly.

The selected anchors are registered on `ImageConditionedDressableModel` as a buffer and are used by checkpoint save/load and evaluation. They are never written back into the dataset or base model.

## 5. State transaction

`mmlphuman_state_transaction(base_model, pose, Rh, Th)` validates exact shapes `[165]`, `[3,3]` and `[3]`, then:

1. Saves `_smpl_poses`, `smpl_poses_cuda`, `_Rh`, `_Th` and the original `cache_dict` object/content.
2. Uses the public `smpl_poses`, `Rh` and `Th` setters.
3. Explicitly clears the active cache after all setters.
4. Restores the original internal state references and original cache object/content in `finally`.

It does not use `.data`, replace Parameters or update frozen tensors. The same transaction is used for each reference ED render and for target RGB/alpha rendering. Strict real mode fails if the base lacks complete MMLPHuman state; old mock rendering remains available without being labeled real.

## 6. Reference ED geometry

`prepare_real_reference_geometry()` runs under `torch.no_grad()` and, for each reference view:

1. Installs that view's pose/Rh/Th through the state transaction.
2. Builds the unchanged MMLPHuman world-to-camera camera.
3. Calls the existing `render_mmlphuman_expected_depth()` with `render_mode="ED"`, `canonical_overrides=None` and the same background used for target rendering.
4. Collects depth and alpha into `Tensor[K,1,H,W]`.
5. Requires finite tensors and exact image dimensions.
6. Verifies that base pose/cache objects are restored.

The per-episode deformation closure captures only episode Rh/Th and calls the verified adapter as:

```python
adapter.deform_anchors(anchors, pose, reference_Rh[view_index], reference_Th[view_index])
```

No episode transforms are stored on the long-lived adapter.

## 7. Depth-aware model interface

All local image-conditioning entry points now propagate:

```text
surface_depth_maps
surface_alpha_maps
depth_abs_tolerance
depth_rel_tolerance
depth_alpha_threshold
require_depth_visibility
```

`forward_episode()` returns `projection`, `anchor_visibility` and `reference_geometry_diagnostics`. When depth is required but absent, the existing projector fail-fast behavior is preserved. Gate 1 ED mathematics, pixel/grid convention and visibility formula were not changed.

## 8. xyz-only channel gate

The real MVP config sets:

```yaml
enable_delta_xyz: true
enable_delta_scaling: false
enable_delta_opacity: false
```

Disabled anchor and Gaussian outputs are generated with `torch.zeros_like`, making them exactly zero and disconnecting their output rows from the rendering gradient. The three-head architecture remains present for later experiments. Default values are all `true`, so existing tests and old configurations retain their behavior.

The output layer remains zero-initialized to preserve zero-offset initialization. The real acceptance script therefore performs the configured one-step run, then reuses the same episode for staged connectivity checking. If the first connectivity backward is still blocked by zero-initialized intermediate heads, it performs at most one additional in-memory warmup update and a final backward. No new data are sampled, and the saved/round-tripped Gate checkpoint remains the configured one-step checkpoint. This checks nonzero finite gradients in encoder, aggregator, HyperNetwork and Anchor MLP without weakening zero-offset initialization.

## 9. Local verification

Python used for tensor tests:

```text
D:\miniconda3\envs\torch_env\python.exe
PyTorch 2.5.1
```

Results:

| Check | Result |
|---|---|
| `py_compile` for all modified/added Python files | PASS |
| Existing synthetic/mock scripts | 17/17 exit 0 |
| Existing PASS lines | 292/292 |
| New Gate 2 interface checks | 21/21 PASS |

The 21 new checks cover inline/NPY Rh/Th, missing/invalid transforms, collate, base/teacher anchor rules, normal/exception cache restoration, per-view transforms, ED stack shape, no-depth failure, model propagation, teacherless forward, target transaction, frozen base, trainable gradients, xyz-only channels and old global-only compatibility.

These are synthetic/mock results, not a real Gate 2 result.

## 10. Required cloud assets

1. `/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth`
2. Fixed `debug_episode_v1.json` with native-resolution RGB/masks, pose `[165]`, camera, Rh and Th for at least K references plus one disjoint target.
3. Every file referenced by that manifest.
4. Verified subject02 `gaussian/lbs_weights_grid.npz` path.
5. CUDA environment containing the accepted MMLPHuman, gsplat ED and PyTorch3D dependencies.

The manifest and LBS paths in `configs/canon_dress_gs_mvp_real.yaml` are explicit cloud placeholders and must be checked before execution.

## 11. Cloud sync list and first command

Sync these files without overwriting unrelated cloud changes:

```text
scene/dressable_dataset.py
scene/image_conditioned_dressable_model.py
scene/dressable_gaussian_model.py
train_dressable.py
utils/mmlphuman_state_utils.py
tools/build_synthetic_image_conditioned_manifest.py
tools/check_real_image_conditioned_interfaces.py
tools/check_real_image_conditioned_one_batch.py
configs/canon_dress_gs_mvp_real.yaml
docs/REAL_IMAGE_CONDITIONED_GATE2_REPORT.md
```

First run the CPU/synthetic interface check in the cloud environment:

```bash
python tools/check_real_image_conditioned_interfaces.py
```

Then run the first real acceptance command after replacing the manifest/LBS placeholders:

```bash
python tools/check_real_image_conditioned_one_batch.py \
  --config configs/canon_dress_gs_mvp_real.yaml \
  --manifest /root/autodl-tmp/canondressgs_work/data_dressable/debug_episode_v1.json \
  --output_dir outputs/gates/gate_2/GATE-ONEBATCH-REAL-001 \
  --device cuda
```

Success requires `diagnostics.json`, a saved checkpoint, a rendered image, finite loss/gradients, zero base gradients, exact zero disabled channels and restored reference/target state. Only then may Gate 2 be changed to **REAL-VERIFIED**.
