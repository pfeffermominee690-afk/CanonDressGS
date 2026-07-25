# AAAI27 Figure 3 Geometry-Causal Assembly

## Frozen Claim

Figure 3 reports geometry interpolation causal attribution from branch `research/continuous-control-causal-attribution-20260722` at `8c43524b7ca0ee8b3c795dbe349466f30b29354f`.

- Geometry global main-effect magnitude: `0.9959405426885113`
- Visibility global main-effect magnitude: `0.012870075845089885`
- Appearance global main-effect magnitude: `0.032429038245185914`
- Geometry sufficient: `10/10` registered subject02 garment pairs
- Geometry necessary: `10/10` registered subject02 garment pairs
- Identity and wardrobe scope: fixed identity, closed wardrobe

These values are read from `paper_protocol/reviewer_risk/factor_causal_profiles.json`; they are not recomputed by the figure builder.

## Assembly Rule

Panel A uses `O01_O02_strongest_factor.png`, selected by the deterministic rule `LEXICALLY_FIRST_REGISTERED_PAIR_O01_O02_NOT_QUALITY_PICKED`. The copied PNG is byte-identical to its frozen source. It is displayed in full with proportional resizing and no crop, retouch, sharpening, or source-pixel modification.

Panels B-D are regenerated labels and plots from registered structured data:

- Panel B compares the global main effects for geometry, visibility, and appearance.
- Panel C reports the all-pair sufficiency and necessity counts.
- Panel D states the frozen protocol scope and denominator.

The assembled PDF is `paper_draft/figures/publication/figure3_geometry_causal.pdf`. Full source and output hashes are recorded in `paper_protocol/reviewer_risk/paper_figure3_geometry_causal_registry.json` and `paper_protocol/reviewer_risk/paper_publication_figure_transform_registry.json`.

## Caption Boundary

The paper caption must identify the 10 subject02 garment pairs, frozen geometry/visibility/appearance decomposition, fixed identity, and closed-wardrobe scope. It must not imply cross-subject, unseen-garment, learned continuous-control, or automatic-routing evidence.

`PAPER_FINAL=false`.
