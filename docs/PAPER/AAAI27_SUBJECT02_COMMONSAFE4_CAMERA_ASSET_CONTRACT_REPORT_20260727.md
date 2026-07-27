# Subject02 CommonSafe4 camera asset contract audit

Task: `AAAI27-SUBJECT00-AUTH-REVIEW-FREEZE-SUBJECT02-CAMERA-CONTRACT-001`

## Outcome

The unique classification is `SUBJECT02_CAMERA_CONVENTION_OR_ASSET_REGISTRY_AMBIGUOUS`. Four formal Subject02 target
cameras exist and all have O01/O03/O04 supervision, but no authoritative
Subject02 slot-to-condition/camera binding exists. The raw 24-camera calibration
and the four-camera formal target registry are expressed in different asset
frames, and no frozen raw-to-formal transform or 24-camera-to-condition mapping
was found. Therefore no `right -> slot06`, nearest-yaw, or renderer-only
substitution is authorized.

The authoritative Subject00 review freeze has been rebound from the superseded
PDF `f782724f47d542ff429f21cf1b418c37b3d9107dbd10495874a7a3c0593ec4b9` to the unique 12-page PDF
`1859c41eaa43f81ecfed8e459251871780401070ccd65e4c6f49a9a67012b874` at review HEAD `8de156e2eac0932db5ec70830eedae5381cf22cc`. Human
decision content is unchanged.

## Formal target cameras

| condition | camera | frozen label | yaw (deg) | pitch (deg) | garment coverage |
|---|---|---:|---:|---:|---:|
| cond_000000 | camera_000000 | front | 180.000000 | -3.000000 | 3/3 |
| cond_000318 | camera_000318 | back | 0.000005 | -3.000000 | 3/3 |
| cond_000017 | camera_000017 | left | 270.000003 | -3.000000 | 3/3 |
| cond_000347 | camera_000347 | right | 90.000001 | -3.000000 | 3/3 |

## Requested garment × condition coverage

| garment | condition | camera | frozen label | loader index | eligible |
|---|---|---|---|---:|---:|
| O01 | cond_000000 | camera_000000 | front | 0 | true |
| O01 | cond_000318 | camera_000318 | back | 1 | true |
| O01 | cond_000017 | camera_000017 | left | 2 | true |
| O01 | cond_000347 | camera_000347 | right | 3 | true |
| O03 | cond_000000 | camera_000000 | front | 8 | true |
| O03 | cond_000318 | camera_000318 | back | 9 | true |
| O03 | cond_000017 | camera_000017 | left | 10 | true |
| O03 | cond_000347 | camera_000347 | right | 11 | true |
| O04 | cond_000000 | camera_000000 | front | 12 | true |
| O04 | cond_000318 | camera_000318 | back | 13 | true |
| O04 | cond_000017 | camera_000017 | left | 14 | true |
| O04 | cond_000347 | camera_000347 | right | 15 | true |

All 24 selected RGB/mask assets were independently rehashed against the formal
manifest. The result was `PASS_24_OF_24`.

## Underlying raw cameras

| camera | raw-world yaw (deg) | raw-world pitch (deg) | raw frames/masks | formal garment targets |
|---|---:|---:|---:|---:|
| cam00 | 25.552844 | -27.691038 | 3110 | 0 |
| cam01 | 15.147892 | -17.086389 | 3110 | 0 |
| cam02 | 4.252691 | -4.682942 | 3110 | 0 |
| cam03 | 356.654215 | 5.268222 | 3110 | 0 |
| cam04 | 345.730982 | 18.018327 | 3110 | 0 |
| cam05 | 336.168268 | 29.603741 | 3110 | 0 |
| cam06 | 323.244528 | 40.088193 | 3110 | 0 |
| cam07 | 306.173894 | 48.062297 | 3110 | 0 |
| cam08 | 281.442131 | 54.862006 | 3110 | 0 |
| cam09 | 256.868911 | 55.350741 | 3110 | 0 |
| cam10 | 235.265970 | 50.831484 | 3110 | 0 |
| cam11 | 215.400755 | 43.152908 | 3110 | 0 |
| cam12 | 196.490362 | 33.262112 | 3110 | 0 |
| cam13 | 187.680381 | 21.010095 | 3110 | 0 |
| cam14 | 178.612720 | 9.338459 | 3110 | 0 |
| cam15 | 169.216453 | -3.000220 | 3110 | 0 |
| cam16 | 160.607238 | -12.660812 | 3110 | 0 |
| cam17 | 150.689467 | -24.600511 | 3110 | 0 |
| cam18 | 137.919791 | -34.608245 | 3110 | 0 |
| cam19 | 119.868580 | -43.709564 | 3110 | 0 |
| cam20 | 100.452913 | -48.341026 | 3110 | 0 |
| cam21 | 77.812083 | -49.541193 | 3110 | 0 |
| cam22 | 57.683273 | -45.713591 | 3110 | 0 |
| cam23 | 39.667658 | -37.613114 | 3110 | 0 |

These 24 cameras each contain 3,110 raw JPEGs and 3,110 masks. They are not
formal garment targets. No explicitly registered render-only camera asset was
found; arbitrary renderer capability is not counted as an asset.

## Coordinate and proxy finding

The audit convention is OpenCV-style camera `+Z` forward. `w2c` maps world to
camera, `c2w = inverse(w2c)`, world `+Y` is the audit up axis, yaw is
`atan2(optical_axis_world.x, optical_axis_world.z) mod 360`, and pitch is
`atan2(y, hypot(x,z))`.

Subject00 slot06 is yaw `320.537631` and pitch
`-24.087365` degrees. Subject02 formal right is
yaw `90.000001` and pitch
`-3.000000` degrees. Their direct numeric
yaw difference is `129.462369`
degrees. A label-derived front-baseline normalization gives
`39.044249` degrees, but
this is diagnostic only because the required cross-frame transform is absent.

Raw `cam06` is the closest numeric raw-calibration yaw candidate:
`cam06` /
`323.244528` degrees, a numeric yaw difference of
`2.706897` degrees from slot06, but its
pitch difference is `64.175558` degrees.
That record is not a physical back-right binding.

## Safety and next task

- `RIGHT_TO_SLOT06_MAPPING_AUTHORIZED=false`
- `TARGET_EXTENSION_AUTHORIZED=false`
- `SUBJECT02_MATCHED_EXECUTION_AUTHORIZED=false`
- `SUBJECT02_OPTIMIZER_STEPS=0`
- `SUBJECT02_NEW_OUTPUT_ROOTS_CREATED=0`
- `PAPER_MODIFICATIONS=0`

Next task: `FREEZE_SUBJECT02_RAW_TO_FORMAL_CAMERA_FRAME_TRANSFORM_AND_EXPLICIT_SLOT_CONDITION_BINDING`. It must freeze the raw-to-formal camera-frame transform
and an explicit result-independent slot/condition/camera registry before any
matched execution can be authorized.
