# Subject00 body-surface representation analysis

Decision: `APPEARANCE_AND_SILHOUETTE_EMERGING`

`PAPER_FINAL=0`

The decision combines automatic representation metrics with 96/96 original-detail manual
comparison-sheet reviews. It is not based on LPIPS alone.

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| foreground RGB variance | 0.083380051112 | 0.083965063095 | 0.067105233669 | 0.097602248192 |
| foreground chroma | 0.070652318459 | 0.070006381720 | 0.059063222259 | 0.086284860969 |
| GT/pred color histogram L1 | 0.153080977093 | 0.139014316698 | 0.074711095547 | 0.288072562358 |
| silhouette IoU | 0.915549589663 | 0.915253702858 | 0.885439087387 | 0.966085398365 |
| boundary F-score | 0.803278516131 | 0.808268163788 | 0.668293680974 | 0.979827473453 |
| garment proxy undercoverage | 0.051245250758 | 0.052723457748 | 0.024950287807 | 0.082189805580 |
| garment proxy overcoverage | 0.001571124126 | 0.001594992639 | 0.000247287085 | 0.003156150060 |

Appearance and color are clearly emerging: medium-final views consistently reconstruct the
blue/white hoodie and dark lower-body appearance, and all 96 queries improve both LPIPS and RGB
MAE against step0. Silhouettes are coherent and all four strict quadrants improve.

The body-surface constraint remains visible. Loose hoodie bulk, sleeve volume, and hem offset are
underrepresented, producing mild-to-moderate body-conforming bias. Face/eye and hand/finger fine
detail remains limited. No severe component contamination or catastrophic structure failure was
found. The result therefore establishes emerging appearance and silhouette, not complete
high-resolution loose-clothing capacity.
