# Subject00 Backend Minimal Canary Plan

This task designs but does not authorize or execute a canary.

The canary uses two existing requests from the frozen 48-entry budget:

- `subject00_O01_slot00_cand00`: O01 front.
- `subject00_O03_slot03_cand00`: O03 left side.

Both must preserve the exact frozen inputs, prompt IDs, SHA-256 values, output count, and target resolution. Each request permits at most two retries after the initial attempt, only for technical transient errors, with 3/9 second backoff. Scientific quality rejection is never retried automatically.

Both outputs must pass two independent identity, garment, pose/camera, full-body, hand/foot, no-text, and provenance reviews. Any hard rejection or nonretryable provider error stops the canary. The remaining 46 requests require both canaries to pass, a sealed Formal Base dependency, and a new explicit user authorization.
