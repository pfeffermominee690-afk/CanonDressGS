# Subject00 Generation Request Plan

Task: `AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001`

The frozen budget is exactly `3 garments x 8 slots x 2 candidates = 48` requests. Stable IDs range from `subject00_O01_slot00_cand00` through `subject00_O04_slot07_cand01`. The canonical request-set SHA-256 is `10da088b6387e008c6e3e6f2de7403a08dea3e511e6d6fd75569214ef7bd0e91`.

Every entry binds one real Subject00 RGB/mask condition, shared calibration and SMPL-X provenance, expected view, garment-specific prompt/negative hashes, one PNG output, and a 1024x1536 portrait normalization target. Backend, model, seed, and garment donor image remain null pending backend adjudication. Each record includes planned raw-response, accepted-output, and append-only provenance paths.

All 48 entries have `authorized=false`, `executed=false`, and `response_count=0`. No request may start before explicit authorization and Formal Base adjudication. Technical retries are limited to two with 3/9-second backoff; quality rejection is never an automatic retry reason.
