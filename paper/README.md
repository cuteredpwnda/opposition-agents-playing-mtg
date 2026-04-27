# Tech Report

This directory contains the LaTeX source for the project's tech report.

## Build

```bash
cd paper
pdflatex opposition_agents_mtg.tex
pdflatex opposition_agents_mtg.tex   # second pass for cross-references
```

The bibliography uses an inline `thebibliography` environment, so no `bibtex`/`biber` pass is required.

## Files

- `opposition_agents_mtg.tex` — main tech report (arXiv-compatible, self-contained, MIT-licensed).

## Citation

See the [`Citation`](../README.md#citation) section of the top-level README for a BibTeX entry.
