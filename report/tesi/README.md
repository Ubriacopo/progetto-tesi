# Thesis

This directory contains the LaTeX sources for *A Multimodal Knowledge Distillation Model for Emotive EEG Characterization*.

## Chapters

1. [`1-introduction.tex`](1-introduction.tex) introduces the problem and research question.
2. [`2-sota.tex`](2-sota.tex) reviews EEG foundation models, self-supervised learning, multimodal fusion, and knowledge distillation.
3. [`3-eegavi.tex`](3-eegavi.tex) describes the data pipeline, architecture, and training strategy.
4. [`4-test.tex`](4-test.tex) defines the experimental protocol and reports results.
5. [`5-conclusions.tex`](5-conclusions.tex) summarizes the findings and limitations.

`tesi.tex` is the main document. `tesi-ref.tex` is the original template reference and is not part of the thesis build.

## Build

The document is built from this directory:

```bash
latexmk -pdf tesi.tex
```

The equivalent build sequence without `latexmk` is:

```bash
pdflatex tesi.tex
bibtex tesi
pdflatex tesi.tex
pdflatex tesi.tex
```

The University of Milan thesis style and referenced images must be available locally.

## Before submission

- Replace the placeholder degree, student number, laboratory, and logo data in `tesi.tex`.
- Remove the remaining template chapters and sample text.
- Resolve empty citations and cross-references.
- Complete the experimental and conclusion chapters.
- Build from a clean directory and review the PDF.
