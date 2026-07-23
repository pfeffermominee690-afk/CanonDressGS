# Subject00 canary optimization signal (2026-07-23)

## Gate result

`PASS`.

| Gate | Before | After | Result |
|---|---:|---:|---|
| Total loss median (first/last 48) | 0.009627176449 | 0.009059807751 | 5.893407% reduction; PASS |
| L1 median (first/last 48) | 0.009598378558 | 0.008991606068 | 6.321614% reduction; PASS |
| Train/train mean LPIPS | 0.137431931061 | 0.121086820339 | PASS |
| Train/train improved queries | — | 24/24 | required >=18/24; PASS |

LPIPS was available in the frozen environment and is the contracted primary
evaluation error. All four quadrant means improved in LPIPS, RGB MAE, PSNR,
SSIM, silhouette IoU and boundary F-score. Held-out metrics were recorded but
were not used to tune, select, extend or rerun the canary.

The numerical signal does not override the visual limitation: predictions
remain low-texture gray bodies after 384 steps. This report establishes a
bounded optimization signal, not formal reconstruction quality.
`PAPER_FINAL=0`.
