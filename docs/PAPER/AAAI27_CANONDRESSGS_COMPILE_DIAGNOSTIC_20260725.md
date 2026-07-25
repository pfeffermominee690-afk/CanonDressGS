# CanonDressGS AAAI-27 Compile Diagnostic

Date: 2026-07-25

## Build History

1. Initial cloud build stopped because the official style transitively required missing `binhex.tex`.
2. Retry 1 stopped because the TeX Gyre `ts1-qtmr` font metric was missing.
3. Retry 2 generated a PDF but BibTeX could not locate `aaai2027.bst` from the build directory.
4. The first successful candidate produced five pages and resolved all citations, but a narrow table produced multiple underfull horizontal boxes.
5. Layout checks removed unsafe figure inclusions, moved table captions below tables, and changed the three tables to auditable full-width layouts without changing scientific values.
6. The formal Build 1 and Build 2 each started with `latexmk -C`, used the same fixed `SOURCE_DATE_EPOCH`, and completed successfully.

## Formal Build Result

- Compile-result source HEAD: `5af83d483d8f06751b79fd18e49267d594aa457b`
- Build command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- PDF SHA-256: `BED98C75B5411606D578632452014D628198024B638C410988222091AF456236`
- Final log SHA-256: `2187CA293425F2BDEE2F86393D0B5AF42DECD89585DB983839AF09CC081AB540`
- BBL SHA-256: `E54F08CEEB517A235A9832F1EBB4C7E51254FA8050BA6C3676748593CAC62BC5`
- Build 1 and Build 2 PDF, log, and BBL files are byte-identical.
- Fatal errors: 0
- Undefined references in final log: 0
- Undefined citations in final log: 0
- Missing figure files: 0
- Missing table inputs: 0
- Duplicate labels: 0
- Overfull boxes: 0
- Underfull horizontal boxes: 0
- Underfull vertical boxes: 2 (`badness 2653` on page 1 and `badness 10000` on page 4)
- Font warnings: 0
- Final absolute-path leaks: 0

Latexmk's first internal LaTeX pass reported the expected unresolved citations and cross-references before BibTeX and subsequent passes. Those transient messages are retained only in the external build stdout/stderr and are fully resolved in the submitted `main.log` and PDF.

## Figure Decision

No figure is embedded in the compiled draft. This is deliberate rather than a missing-file workaround:

- Figure 1 still requires manual adjudication.
- Figure 3 lacks an assembled core geometry-causal asset.
- Figure 2 contains a pixel-embedded candidate/final-state label.
- Figure 4 contains a pixel-embedded supplementary/candidate label.
- Figure 5 contains pixel-embedded candidate and governance labels.
- Figure 6 contains pixel-embedded attempt, classification, and pass/fail labels.

The task forbids retouching, cropping, repainting, or regenerating scientific pixels. Figure 2/4/5/6 are therefore marked `FIGURE_PUBLICATION_LABEL_REQUIRES_REGENERATION` and omitted until an authorized publication export is available. Scientific pixels were not changed.

## PDF Validation

The five rendered pages were visually inspected. The PDF has US-letter pages, no blank page, no overlap, no garbled text, no visible TODO, continuous equations (1)--(6), three legible tables, and a populated References section. All 15 fonts reported by `pdffonts` are embedded and subset Type 1 fonts. Ghostscript validation completed without error. The PDF is not encrypted and has no JavaScript.

The compile is successful, but unresolved Figure 1, Figure 3, and final Subject00 scope integration keep the draft in `PAPER_ENDPOINT_DRAFT_COMPILED_WITH_P0_GAPS`.
