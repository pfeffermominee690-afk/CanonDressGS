#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
