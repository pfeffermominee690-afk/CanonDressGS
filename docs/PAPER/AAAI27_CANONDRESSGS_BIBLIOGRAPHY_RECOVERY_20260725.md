# CanonDressGS AAAI-27 Bibliography Recovery

Date: 2026-07-25

## Discovery

Repository and Git-history searches found no usable prior bibliography beyond the placeholder created by the endpoint-restructure task. A local user library was located at an external path and had SHA-256 `210C405F3258746264E177FC46EEB861B013BF70285A9785236499C2ED266AD9`. Its 19 entries were treated only as candidates; the file was not copied wholesale.

## Recovery Method

Seven entries were selected because they directly support the three Related Work statements. Authors, titles, venue, year, and pages were checked against the official Computer Vision Foundation Open Access records. No DOI or arXiv identifier was invented, and no metadata field was completed from model memory.

Recovered keys:

1. `Li_2024_CVPR`
2. `Qian_2024_CVPR`
3. `Kocabas_2024_CVPR`
4. `Ma_2020_CVPR`
5. `Patel_2020_CVPR`
6. `Kim_2024_CVPR`
7. `Neuberger_2020_CVPR`

The first three support the statement that Gaussian-avatar systems deform and rasterize explicit Gaussian primitives. The next three support the statement that garment-aware human methods model clothing through geometry, appearance, deformation, or combinations of these factors. The final citation supports the bounded statement that image-based virtual try-on has synthesized edited person images from garment references.

## Audit Result

- Bibliography file: `paper_draft/references.bib`
- Entry count: 7
- Unique key count: 7
- Visible citation key count: 7
- Unresolved citation TODO count: 0
- Undefined visible citation count: 0
- Fabricated citation count: 0
- Duplicate key count: 0
- Metadata-incomplete entries admitted to the PDF: 0
- BibTeX result: pass

The formal BBL contains all seven references, and PDF text extraction shows no `[?]`, undefined citation marker, or raw citation key.
