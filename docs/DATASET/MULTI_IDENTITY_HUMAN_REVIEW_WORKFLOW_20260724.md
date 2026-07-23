# Multi-Identity Human Review Workflow

Task: `AAAI27-MULTI-IDENTITY-GENERATION-BACKEND-001`

## Candidate review

Each candidate first receives per-check automatic evidence. Automatic checks may reject or warn, but cannot formally accept. Every check stores its raw score, units, threshold, comparison, evaluator version, timestamp, and decision; a single aggregate score cannot replace those records.

Formal candidate review requires two independent people. Reviewer A and Reviewer B inspect candidate, mask, identity reference, and same-garment cross-view evidence. Neither initial reviewer can retrieve the other's initial decision. Each records ACCEPT/REJECT/MAYBE plus identity preservation, garment semantics, full-body, hands, feet, anatomy, background, blur, view, cross-view consistency, body-conforming bias, garment volume, comment, reviewer ID, and timestamp.

Two ACCEPT decisions resolve to ACCEPT. Any initial REJECT resolves to REJECT. Every other complete pair resolves to `ADJUDICATION_REQUIRED`. A third person must append an explicit ADJUDICATION review; a VLM cannot break the tie. All review output is hash-chained JSONL and append-only.

## Group review

After single-candidate resolution, each identity x garment group is reviewed across all eight distinct semantic slots. Reviewers check color, material, sleeve length, hem, front/back semantics, left/right consistency, identity, and random accessory changes. The group must contain eight distinct candidate SHA-256 values. Endpoint acceptance requires a group PASS in addition to candidate-level acceptance.

## Local tool

`tools/multi_identity_review/review_server.py` binds only to `127.0.0.1`. It serves static UI assets, a candidate evidence queue, and read-only media constrained below a configured root. It rejects absolute paths and traversal. Candidate and group decisions are appended to separate JSONL logs; no route mutates candidate pixels.

The browser UI supports candidate navigation, side-by-side candidate/mask/identity/cross-view evidence, keyboard navigation and decision shortcuts, browser-local drafts, and append-only submission. Content Security Policy restricts images, scripts, styles, and connections to the local origin. The tool has no upload or generation API client and contains no credential field.

No candidate or group reviews were performed in this task because no candidate image was generated.
