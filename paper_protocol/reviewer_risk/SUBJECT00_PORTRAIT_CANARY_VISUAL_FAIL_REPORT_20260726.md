# Subject00 Portrait Canary Visual Fail Report

- Task: `AAAI27-SUBJECT00-PORTRAIT-CANARY-VISUAL-FAIL-ADJUDICATION-001`
- Source: `research/subject00-managed-portrait-canary-20260725` at `4a1eeb2e53fc1a8b5a5bf1c3b706aac374e156dc`
- Portrait canary result: `9742c675943c2b105a5b153888e5b160b73471f2`
- Engineering result: `PASS`
- Native-resolution result: `PASS 4/4`
- Human visual result: `FAIL 4/4`
- Accepted / Teacher targets: `0 / 0`
- Formal Base: `PENDING`
- Remaining 39 authorization: `DENIED`
- PAPER_FINAL: `false`

The fixed user review rejects all four canaries. The portrait canvas contract returned native 1024x1536 PNG files, but the contained 1330x1150 condition was reduced to 960x830 at offset (32,353). The model retained the appearance of a reduced landscape scene embedded in a gray portrait canvas instead of preserving subject scale and camera registration while extending the scene.

The common visual failures are reduced subject pixel height, changed camera framing, increased apparent camera distance, large gray canvas areas, weakened identity reviewability, and weakened garment-boundary reviewability. Therefore `NATIVE_RESOLUTION_PASS` is not `CAMERA_REGISTRATION_PASS` and is not `FORMAL_DATASET_PASS`.

The historical technical result remains valid as engineering evidence. This adjudication supersedes its pending visual status with `SUBJECT00_PORTRAIT_CANARY_TECHNICAL_PASS_VISUAL_FAIL`. No image is accepted, no Teacher target is created, and no additional generation is authorized.
