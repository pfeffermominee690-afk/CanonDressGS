# Subject00 Backend Evidence Comparison

| Candidate | Strongest evidence | Blocking evidence |
|---|---|---|
| Codex managed image edit | 12/12 Subject02 direct edits and 15 accepted donor sheets; native 1024x1536 | exact model/revision unavailable; interactive/session-bound; raw provider response unavailable; Subject02 formal visual gate failed |
| Historical Sublyx proxy | scriptable `/v1/images/edits`; 246 accepted traceable donor records; 915-row Jay execution log | alias/revision unpinned; current availability unverified; historical 1536x1024 only; research-use boundary absent |
| Historical 78Code proxy | exact `gpt-image-2` model ID discovered; scripted diagnostics and error taxonomy | edits probe returned HTTP 500 `convert_request_failed`; no successful image; revision/license/resolution unverified |

The weighted score is diagnostic only. Mandatory gates override every total, and no backend is eligible for PRIMARY or FALLBACK.

Historical success classes are kept separate: `CONNECTIVITY_ONLY`, `IMAGE_EDIT_SUCCESS`, and `FORMAL_DATASET_GENERATION_SUCCESS`. No evidence establishes a successful Subject02 formal target dataset from any selectable reproducible backend.
