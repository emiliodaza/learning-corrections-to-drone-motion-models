#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/report build/presentation

(
    cd report
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/report report.tex
    (cd ../build/report && BIBINPUTS=../../report: bibtex report)
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/report report.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/report report.tex
)
(
    cd presentation
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/presentation slides.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/presentation slides.tex
)
cp build/report/report.pdf report/report.pdf
cp build/presentation/slides.pdf presentation/slides.pdf
printf '\nBuilt report/report.pdf and presentation/slides.pdf\n'
