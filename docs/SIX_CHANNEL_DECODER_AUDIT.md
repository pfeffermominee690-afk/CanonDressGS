# Six-channel Decoder Audit

Audit baseline: `ed5fa150f1d10aab5bcc9fc96c9969aa684f517a`. Gate 4-C checkpoint: `GATE4-REAL-OVERFIT100-001/checkpoint_step_000100.pth`.

## Existing structure

- Shared trunk: `AnchorClothingMLP.hidden_layers`, two Linear layers for `num_layers=3`, each width 128, with SiLU.
- Input has 27 anchor features plus a historical 64-dimensional clothing segment. FiLM image inference uses only the anchor columns of layer 0; the local 64→27 zero-initialized adapter is added to anchor features.
- FiLM is injected after every hidden Linear and before SiLU. `ClothingFiLMGenerator` has a 64→128 SiLU trunk and separate zero-initialized gamma/beta heads for both 128-wide hidden layers.
- Historical output is `anchor_clothing_mlp.output_layer`, shape `[7,128]`, bias `[7]`. Rows 0:3 are raw `delta_xyz`; rows 3:6 and 6:7 are historical scaling/opacity outputs.
- Gate 4-C xyz is not bounded in `AnchorClothingMLP`; `enable_delta_xyz` gates it later in `DressableGaussianModel`. The new bounded Module-2 path uses explicit `max_abs=0.05`; the unchanged legacy path is used for exact Gate 4-C comparison.
- Current gate is applied after prediction and before interpolation. Gate 4 uses one precomputed reference-only `[A]` tensor for every legacy channel.
- Current legacy interpolation gathers `[N,K,C]`, multiplies normalized weights and sums K. Module 1 already supplies typed six-channel interpolation but lacked the standalone validation contract now provided by `interpolate_anchor_field`.
- Historical scaling/opacity outputs exist inside the seven-row legacy layer, not as independent heads. Minimal compatible adaptation keeps that layer and its keys unchanged, treats rows 0:3 as `xyz_head`, ignores rows 3:7 in Module 2, and adds five independent zero-initialized heads.

## Gate 4-C decoder state keys

`dressable_model.anchor_clothing_mlp.hidden_layers.{0,1}.{weight,bias}`; `output_layer.{weight,bias}`; `local_feature_adapter.{weight,bias}`; FiLM `trunk.0.{weight,bias}`, `gamma_heads.{0,1}.{weight,bias}`, and `beta_heads.{0,1}.{weight,bias}`. Exact observed shapes are `[128,91]`, `[128]`, `[128,128]`, `[128]`, `[7,128]`, `[7]`, `[27,64]`, `[27]`, with FiLM heads `[128,128]`/`[128]`.

Legacy loading inserts deterministic zero tensors for missing new-head keys before strict state loading. It never renames or reinitializes `output_layer`.
