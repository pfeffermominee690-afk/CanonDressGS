# Subject00 Valid-Region Protocol Comparison

| Candidate | Protocol | Classification | Recommendation |
| --- | --- | --- | --- |
| A | `REGISTERED_VALID_REGION_PORTRAIT` | `BLOCKED_BY_MISSING_END_TO_END_PIXEL_VALIDITY_SUPPORT` | `CONDITIONAL_PREFERRED_RESEARCH_DIRECTION` |
| B | `NATIVE_REGISTERED_LANDSCAPE` | `CONDITIONAL_AUDIT_REQUIRED_HOLD` | `BACKUP_ONLY_AFTER_REGISTRATION_AUDIT` |
| C | `REGISTERED_CROP_NATIVE_1024x1150` | `PIPELINE_TECHNICALLY_SUPPORTED_GENERATION_BACKEND_UNPROVEN_HOLD` | `CONDITIONAL_ON_NATIVE_BACKEND_PROOF` |
| D | `CONTINUE_VERTICAL_OUTPAINT` | `REJECT` | `REJECT` |

## Decision Boundary

No candidate is executable in this task. Candidate A is the preferred engineering direction only after an independent HxW validity mask reaches every relevant loss and evaluation metric. Candidate C becomes viable if the generation backend proves native registered `1024x1150` output. Candidate B remains a backup pending per-image registration and pipeline work. Candidate D is rejected by the fixed `0/4` V2 result.

The unique next action is `USER_SELECT_SUBJECT00_VALID_REGION_OR_NATIVE_LANDSCAPE_PROTOCOL`.
