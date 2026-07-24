# LOO F2 Producer Consumer Interface Audit

## Producer

`FrozenF2ReferenceFeatureExtractor.forward` returns `FrozenF2FeatureOutput` for K=1..3. Its
per-reference field is `[K,256]`; frozen mean/max aggregation yields `[1,512]`. It
contains mask pooling and valid-view semantics, no depth field, and preserves input
dtype, device, and query order. The attempted calibration K4 call returned no payload.

## Consumer

`_calibration_initialization_feature_record` now consumes `F2ReferenceFeatureSet`.
The record explicitly contains `features_per_view=[4,256]`, `set_mean=[1,256]`,
`set_max=[1,256]`, and `aggregated_feature=[1,512]`, plus condition IDs, query order,
source image/mask SHA-256 values, dtype, device, and the frozen aggregation rule.
Unknown keys, missing keys, silent squeeze, implicit aggregation, dtype drift, device
drift, and order drift are rejected.

## Field Diff

The first incompatible field was `reference_images.shape[0] / semantic view_count`. The selected
adapter extracts each calibration observation through a legal singleton producer call,
retains `[V,256]` explicitly, then applies `FROZEN_PER_REFERENCE_F2_MEAN_MAX_V1`. It does not
raise the frozen F2 K limit or modify the backbone. Source records audited:
`20/20`; missing features: `0`;
schema drift: `0`.
