# AAAI27 Geometry-Disentangled Dual-Support Micro-Pilot

**RESEARCH MICRO-PILOT — NOT PAPER FINAL**

- Source HEAD: `8c43524b7ca0ee8b3c795dbe349466f30b29354f`.
- Protocol SHA-256: `559a367b69be867dab0075b480f44384a78aae7309b3374672a949773f08b552`.
- Selected stable pair: `O01_O02`; selected unstable pairs: `O01_O03`, `O01_O08`.
- FULL reused: 72 logical / 36 unique / 0 regenerated.
- HARD renders: 72; DUAL renders: 72.
- Endpoint parity: `PASS`; maximum absolute difference `0.0`.
- Manual visual review: 24/24.
- Final classification: **DUAL_SUPPORT_MICRO_PILOT_PASS**.
- NEXT_TASK: **RUN_ALL_10_PAIR_DUAL_SUPPORT_EVALUATION** (not started).

## Artifact-grade comparison

| pair | FULL max | DUAL max | drop | FULL severe | DUAL severe | severe reduction |
|---|---|---|---|---|---|---|
| O01_O02 | 3 | 2 | 1 | 32 | 0 | 1.000 |
| O01_O03 | 3 | 3 | 0 | 64 | 32 | 0.500 |
| O01_O08 | 3 | 1 | 2 | 32 | 0 | 1.000 |

## Quantitative comparison

| pair | silhouette degradation | garment LPIPS degradation |
|---|---|---|
| O01_O02 | 0.00000 | 0.00000 |
| O01_O03 | 0.00000 | 0.00000 |
| O01_O08 | 0.00000 | 0.00000 |

## Governance

No training, backward, diagnostic optimizer, optimizer step, scheduler step, checkpoint write, teacher mutation, basis mutation, formal-output mutation, pair selection, direction selection, view selection, or threshold adjustment occurred. The legacy context optimizer was construction-only, unused, unsaved, and discarded. PAPER_FINAL=0.

The result is limited to the three preregistered seen garment pairs, two directions, three alpha values, and four frozen target views. It does not establish arbitrary garment interpolation, unseen garment generation, novel-pose behavior, novel-view behavior, or cross-identity generalization.
