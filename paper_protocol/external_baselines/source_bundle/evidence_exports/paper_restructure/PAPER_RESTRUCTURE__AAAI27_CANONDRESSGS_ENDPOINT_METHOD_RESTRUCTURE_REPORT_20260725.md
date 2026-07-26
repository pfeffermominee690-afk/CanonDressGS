# CanonDressGS Endpoint-Method Restructure Report (2026-07-25)

## Outcome

The AAAI-27 draft has been rebuilt around reference-controlled closed-wardrobe endpoint realization. The prior HyperNetwork/LHM-assisted generation outline was removed from the main narrative. The draft now defines a frozen avatar, optimized Teacher Endpoints, a rank-4 endpoint coordinate system, frozen F2 set features, a 3,076-parameter coefficient head, nearest-valid-endpoint snapping, and a frozen deformation/LBS/renderer path.

The scientific scope is fixed to subject02 and O01/O02/O03/O04/O08. Raw coefficient prediction is explicitly separated from snapped endpoint realization. Dual-Support is an oracle- or user-specified extension. Headroom and leave-one-garment-out results are presented as negative scope evidence.

## Source And Structure

- Source: `research/paper-figure-bank-loo-method-freeze-20260725` at `3f67f577b85e5f74951287e9a938c11aef74ac60`.
- Target: `research/paper-endpoint-method-restructure-20260725`.
- Main entry: `paper_draft/main.tex` (unique entry by documentclass, document body, title, maketitle, and input graph).
- Previous title: `CanonDressGS: Canonical Clothing-conditioned Gaussian Avatars with LHM-assisted Priors`.
- Working title: `CanonDressGS: Reference-Controlled Garment Endpoint Selection in Canonical Gaussian Space`.
- Previous visible structure: Abstract; Introduction; Related Work; Method; Experiments; Limitations.
- Current visible structure: Abstract; Introduction; Related Work; Method; Experiments; Limitations; Conclusion.

## Evidence Alignment

- Geometry: main-effect magnitude 0.9959405426885113; sufficient and necessary in 10/10 garment pairs.
- Endpoint selection: clean top-1, macro precision, macro recall, and endpoint exact match are 1.0; identity/component contamination and severe wrong outfit are zero for CanonDressGS.
- Hard lookup: clean functional equivalence is disclosed; no superiority claim is made.
- Dual-Support: all ten pairs are included; LPIPS 0.085303 to 0.056985, IoU 0.710052 to 0.725290, and boundary F 0.300263 to 0.379103. Two pairs retain severe residual artifacts.
- Headroom: Teacher LPIPS 0.03876773160882294 versus refined 0.03898213766515255; 0/5 garments improve.
- Leave-one-out: every four-garment rank-3 basis fails the held-out capacity gate; low-dimensional LPIPS 0.11355112642049789 does not improve on hard lookup 0.113437.
- Subject00: no result is included; only the contract-authorized invisible insertion comment exists.

## Build Status

Compilation is blocked, not failed manuscript logic. The source branch contains no AAAI style/class or bibliography, and pdflatex/latexmk/bibtex are absent on Windows, WSL, and the configured cloud host. A placeholder `references.bib` with no fabricated entries was added so verified citations can be restored. No PDF or page count is claimed.

Final task classification: `PAPER_ENDPOINT_METHOD_RESTRUCTURE_COMPILE_BLOCKED`. `PAPER_FINAL=false`.
