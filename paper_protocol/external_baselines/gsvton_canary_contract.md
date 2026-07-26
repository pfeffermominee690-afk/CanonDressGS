# GS-VTON Micro-Canary Contract

Status: `CONTRACT_READY_WITH_LICENSE_AND_DATA_PREFLIGHT_GATES`

This is a native-input, different-assumptions canary. It is not a strict apples-to-apples avatar comparison and does not establish animation or canonical-state compatibility.

## Single fixed case

- Exactly one garment, one target condition, one garment image, one official initialization path, one fixed config, and one attempt.
- Select the garment and target condition before any output exists. No sweep, automatic retry, seed selection, or result-dependent substitution is allowed.
- Use official repository commit `96964b0a6528089123cc27a3ff3e3eb46505cf6e` and preserve its native stage contracts.

## Preflight gates

1. Resolve the missing top-level GS-VTON license through institutional review or written author permission. Public source availability alone is not a license.
2. Convert the preregistered subject02 target-subject views to the official Nerfstudio/GS-VTON camera and image layout without altering RGB content.
3. Build the required vanilla static 3DGS through the official 3DGS initialization path. The CanonDressGS animatable Base Avatar is not silently treated as a compatible PLY.
4. Supply the same one garment reference used by the CanonDressGS one-reference path. CanonDressGS comparison value remains frozen at one-reference top-1 `0.95`; clean multi-reference `1.0` may not replace it.
5. Freeze masks, camera, target condition, output resolution, config, dependency-weight manifest, expected download bytes, available storage, and a 24 GB-class GPU preflight before execution.

## Execution cap and output

- Official config only; stage-2 `max_steps=10000`. Record stage-1 processing, LoRA adaptation, 3DGS initialization, and editing time separately.
- Expected output is one edited static 3DGS and target-camera RGB/alpha renders for the selected fixed pose.
- Evaluate only with `external_target_space_metric_protocol.md`; report native-input differences beside every result.
- Fail on missing output, camera mismatch, unresolved masks, non-finite render, severe identity contamination, or any unregistered retry.

## Claim boundary

Even a successful canary supports only feasibility for one subject02 garment and one fixed target condition under GS-VTON native inputs. It does not support a paper claim, an animatable-avatar claim, a canonical endpoint claim, or a five-garment result.

No download, conversion, training, or inference is authorized by this document.
