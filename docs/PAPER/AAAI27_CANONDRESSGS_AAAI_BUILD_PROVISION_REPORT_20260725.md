# CanonDressGS AAAI-27 Build Provision Report

Date: 2026-07-25

Task: `AAAI27-CANONDRESSGS-AAAI-BUILD-PROVISION-COMPILE-001`

Status: `PAPER_ENDPOINT_DRAFT_COMPILED_WITH_P0_GAPS`

## Source and Entry Point

- Source branch: `research/paper-endpoint-method-restructure-20260725`
- Source HEAD: `20e7d029958f61beb15b095d526e3d894e1571f3`
- Build branch: `research/paper-aaai-build-provisioned-compile-20260725`
- Unique manuscript entry: `paper_draft/main.tex`
- The entry contains the expected document class, document environment, title, anonymous author block, `\maketitle`, section inputs, and bibliography command.

## Official AAAI-27 Inputs

The kit was obtained from the official AAAI-27 conference site. The conference page links the AAAI-27 Author Kit, and the resolved official archive is:

`https://aaai.org/wp-content/uploads/2026/05/AuthorKit27.zip`

- Retrieval time: `2026-07-25T18:35:24+08:00`
- HTTP last-modified value: `Thu, 28 May 2026 15:54:36 GMT`
- Archive size: `5,495,535` bytes
- Archive SHA-256: `E28C6AC9BC6EB3B4E2D849547D2CEFB5162610EE39D0A12E0DC62D1126B44A7D`
- Style declaration: `aaai2027`, `2027/05/04 AAAI 2027 Submission format`
- Official style SHA-256: `391BCE82815BF698B8E382DD3AE7E30C75D7AB46DF140CB295B1266016BC8623`
- Official bibliography style SHA-256: `5DB7765BA99DE5C1E4686F9B3940A0ADD9C5E702F2164514462BEC130CCB6E3C`

Only `aaai2027.sty` and `aaai2027.bst` are retained in `paper_draft/aaai_author_kit/`. The official examples, checklist, PDFs, and Word templates remain in the external build cache. The two retained files are byte-identical to the official archive. Repository attributes explicitly disable line-ending conversion for these files so their official hashes survive Windows/Linux transfer.

The style header prohibits modification for AAAI publication use and supplies the files without warranty. The bibliography style identifies the LaTeX Project Public License. No copyright header was removed, no style macro was copied into `main.tex`, and neither official file was edited.

## Binding

- `\documentclass[letterpaper]{article}`
- `\usepackage[submission]{aaai2027}`
- `natbib` is used with the official `aaai2027.bst`.
- `paper_draft/latexmkrc` adds the author-kit directory to `TEXINPUTS` and `BSTINPUTS` using relative paths.
- `paper_draft/build_clean.sh` performs a clean followed by the official-compatible `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` flow.
- The generated review PDF is anonymous and carries the official submission disclaimer.

## Toolchain Provision

Windows had no usable `pdflatex`, `latexmk`, `bibtex`, `biber`, `kpsewhich`, Ghostscript, or `pdftotext`. The build therefore ran in the authorized cloud environment. TeX packages were installed on the cloud root filesystem, never under `/root/autodl-tmp`, and no Conda environment was changed.

The first package transaction installed the required LaTeX, publisher, font, build, Poppler, and Ghostscript packages. Two narrow dependency additions supplied `binhex.tex` and the TeX Gyre font metric required by the official style. Final root free space after `apt-get clean` is `22,701,973,504` bytes; `/root/autodl-tmp` retains `17,562,742,784` bytes free.

## Boundary

GPU use, training, optimizer execution, rendering, model inference, image generation, scientific result mutation, and Subject00 execution were all zero. The manuscript remains `PAPER_FINAL=false`.
