# AAAI27 Publication Figure Export Report

## Export Set

| Figure | PDF | Scientific role |
|---|---|---|
| 2 | `paper_draft/figures/publication/figure2_method.pdf` | Frozen offline endpoint construction and reference-conditioned inference |
| 3 | `paper_draft/figures/publication/figure3_geometry_causal.pdf` | Geometry interpolation causal attribution |
| 4 | `paper_draft/figures/publication/figure4_dual_support.pdf` | Oracle/user-specified Dual-Support extension across all 10 pairs |
| 5 | `paper_draft/figures/publication/figure5_hard_lookup_relation.pdf` | Clean closed-wardrobe hard-lookup relation |
| 6 | `paper_draft/figures/publication/figure6_endpoint_limits.pdf` | Endpoint headroom, LOO capacity, and fallback boundaries |

Each figure is also exported as a PNG review raster and editable SVG composition. The PDF is the publication asset.

## Deterministic Export

The command is:

```text
python tools/paper_figures/build_publication_figure_assets.py
```

The builder uses a fixed Matplotlib configuration, embedded TrueType fonts, fixed PDF metadata dates, and a fixed SVG hash salt. Raster source panels are loaded without crop or pixel edits. No training, inference, renderer invocation, image generation, or scientific recomputation occurs.

## Frozen Numerical Content

- Figure 3 geometry main effect: `0.9959405426885113`; sufficient `10/10`; necessary `10/10`.
- Figure 4 LPIPS: `0.085303 -> 0.056985`.
- Figure 4 IoU: `0.710052 -> 0.725290`.
- Figure 4 Boundary F: `0.300263 -> 0.379103`.
- Figure 4 severe artifact improvement: `10/10`; identity contamination: `0`.
- Figure 4 cost: approximately `2.000x` active Gaussians and `1.626x` render time.
- Figure 5 clean endpoint agreement: `60/60`; this is descriptive equivalence, not a superiority claim.
- Figure 6 teacher LPIPS: `0.03876773160882294`; refined LPIPS: `0.03898213766515255`; improved garments: `0/5`.
- Figure 6 LOO hard lookup LPIPS: `0.11343667805194854`; rank-3 basis LPIPS: `0.11355112642049789`.
- Figure 6 registered top-1 rates: clean `1.00`, mild blur `0.35`, single reference `0.95`; complete dropout safe fallback `80/80`.

## Provenance Result

All 35 copied source PNGs are registered with SHA-256, byte size, dimensions, and frozen source head. Registered JSON inputs are also hashed. Every publication transform maps its actual source paths and SHA-256 values to the output PDF SHA-256.

- Scientific source branch mutation count: `0`
- Scientific source pixel mutation count: `0`
- Subject00 use count: `0`
- New render count: `0`
- `PAPER_FINAL=false`

The authoritative machine-readable records are:

- `paper_publication_figure_source_registry.json`
- `paper_publication_figure_transform_registry.json`
- `paper_publication_figure_asset_registry.json`
- `paper_figure3_geometry_causal_registry.json`
