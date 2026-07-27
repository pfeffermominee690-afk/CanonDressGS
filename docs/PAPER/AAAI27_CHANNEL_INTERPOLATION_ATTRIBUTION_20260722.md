# AAAI27 Channel Interpolation Attribution

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

| variant | FULL reproduction pairs | NONE/MINOR pairs |
|---|---|---|
| XYZ_ONLY | 10 | 0 |
| SCALE_ROT_ONLY | 10 | 0 |
| OPACITY_ONLY | 10 | 0 |
| SH_ONLY | 10 | 0 |
| GEOMETRY_ALL | 10 | 0 |
| APPEARANCE_ALL | 10 | 0 |

Classification: **NO_SINGLE_CHANNEL_DOMINANT**.

At alpha=0.5 every preregistered variant equals the same six-channel midpoint because all selected and non-selected channels use weight 0.5. Midpoint reproduction counts are therefore descriptive but cannot identify a unique channel cause.

Every variant obeyed the frozen rule: selected channels used alpha, while all other channels used the fixed 0.5 teacher midpoint.
