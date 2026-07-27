# Subject00 Authoritative Review and Subject02 Matched Audit

- Task: `AAAI27-SUBJECT00-AUTHORITATIVE-REVIEW-SUBJECT02-MATCHED-001`
- Source: `research/subject00-commonsafe4-review-pack-unique-root-20260727@8de156e2eac0932db5ec70830eedae5381cf22cc`
- New branch: `research/subject00-authoritative-review-subject02-matched-20260727`
- Final classification: `SUBJECT00_REVIEW_FROZEN_SUBJECT02_MATCHED_BLOCKED_BY_ASSET_CONTRACT`

## Subject00 frozen result

The authoritative 12-page review is frozen as a valid mixed/negative result.
Pure Endpoint obtains 30/36, below Reference Classifier 31/36 and
Nearest-Centroid 35/36. All six errors are O04-to-O03. The primary positive
method claim is not supported.

All three Teacher checkpoints remain technical passes. The human visual
decision is `INCONCLUSIVE_INSUFFICIENT_DEDICATED_MULTIVIEW_EVIDENCE`; a
dedicated multiview review remains pending.

## Subject02 pre-optimizer gate

The Base, O01/O03/O04 Teachers, target manifest, and historical Pure Endpoint
contracts were recovered and hashed. The passed target manifest exposes:

| Condition | Camera | Geometric cardinal direction |
|---|---|---|
| `cond_000000` | `camera_000000` | `front` |
| `cond_000318` | `camera_000318` | `back` |
| `cond_000017` | `camera_000017` | `left` |
| `cond_000347` | `camera_000347` | `right` |

The matched contract requires symbolic slots `[slot00, slot07, slot03,
slot06]`, but the target registry contains zero explicit slot binding rows.
In particular, Subject00 slot06 is cam09/back-right while the Subject02 set
contains a right camera and no authority equating the two. The historical
condition-ID runner/config is also not frozen against the new slot contract.

No positional mapping was invented. Optimizer creations, optimizer steps,
checkpoints, method runs, and baseline cells are all zero. Both reserved
Subject02 output roots remain absent.

## Claim boundary

No matched cross-identity comparison, protocol-effect estimate,
identity-effect estimate, statistical-significance statement, or new positive
paper claim is authorized. The paper body was not modified.

## Required resolution

Freeze an evidence-backed slot-to-Subject02 condition/camera mapping and an
exact matched runner/config binding, then reauthorize execution.

Next task: `USER_FREEZE_SUBJECT02_COMMONSAFE4_MATCHED_ASSET_CONTRACT_AND_REAUTHORIZE_EXECUTION`.
