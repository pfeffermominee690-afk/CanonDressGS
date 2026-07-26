# Subject00 Human Review Execution Plan

Task: `AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001`

The manifest contains 48 empty candidate queue entries and zero review events. It does not invent review results.

Each candidate requires two blinded, independent initial reviewers. Two `ACCEPT` decisions accept; any `REJECT` rejects; every other complete pair requires a human adjudicator. A VLM may not make the final decision or break a tie. Review events are append-only and SHA-chained, and candidate pixels are read-only.

Required review dimensions cover identity, face, hair, skin tone, body shape, pose, camera, garment semantics, slot semantics, mask/edge quality, hands/feet, background, lighting, artifact grade, decision, and reject reason. The strict occlusion allowlist cannot override a hard reject.
