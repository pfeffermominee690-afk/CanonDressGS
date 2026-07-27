# Subject00 Resolution and Camera Registration Contract Adjudication

## Proven Facts

- Managed Image Edit can return native 1024x1536 PNG from a portrait derived canvas.
- The current contain-plus-gray-padding construction substantially reduces subject pixel scale.
- The native-resolution contract passes independently.
- The camera framing, subject scale, and registration contract fails.
- All four canaries remain failed evidence and must not be replaced or hidden.

## Mandatory V2 Gates

Every V2 candidate must return native 1024x1536 without output resize, crop, pad, rotate, stretch, re-encoding repair, or canvas embedding. The complete person and pose/camera direction must remain stable. Subject pixel scale must not be materially lower than the condition. The image must not contain large pure-gray or visibly artificial canvas regions. Background extension must be continuous with the source scene, and identity, hands, feet, and garment boundaries must remain reviewable. Only four canaries may be tested; the remaining 39 stay denied until 4/4 human Visual PASS.

## Candidate Protocols For User Selection

| Candidate | Input canvas construction | Subject size changed | Condition cropped | Camera framing changed | Model extends background | Identity drift risk | Pose drift risk | Native-resolution risk | Scientific-contract fit | Recommendation |
|---|---|---:|---:|---:|---:|---|---|---|---|---|
| A: portrait outpainting / scene extension | Keep the original condition at native subject scale and expand only the vertical scene context to a 1024x1536 edit target | No intended reduction | No | No intended change | Yes | Medium | Medium | Low to medium | Potentially compatible if registration gates pass | HIGH for controlled four-image evaluation |
| B: subject-scale-preserving derived canvas | Scale the source only enough to fit width while preserving subject pixel height; allocate extension outside the source scene without gray filler | Minimal | No | Low intended change | Yes | Medium | Low to medium | Low | Compatible in principle; requires measurable scale tolerance | HIGH for controlled four-image evaluation |
| C: original condition reference plus independent portrait scene target | Image 1 is a portrait scene target with registered camera geometry; Image 2 is the unchanged condition authority | No intended reduction | No | Depends on target registration quality | Yes | Medium to high | Medium | Low | Conditional; target geometry must be independently validated | MEDIUM |
| D: direct portrait recomposition from the condition | Ask the model to synthesize a portrait composition without a deterministic registered scene target | Uncontrolled | No deterministic crop | Likely | Yes | High | High | Medium | Weak registration guarantee | LOW |

These are design candidates only. This task does not select or execute a candidate. The sole next action is `USER_SELECT_SUBJECT00_PORTRAIT_CANARY_V2_PROTOCOL`.
