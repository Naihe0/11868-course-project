# PagedAttention in MiniTorch — Poster

LaTeX source for the project poster.

## Specs

- **Size**: 36 in (wide) × 42 in (tall), portrait
- **Class**: `article` with `geometry` for an exact custom paper size
- **Layout**: `tcolorbox` blocks inside `minipage` columns (no exotic poster class required)
- **Figures**: pulled from `../../../benchmarks/plots/*.png`

## Build

```bash
# from this directory
latexmk -pdf poster.tex
# or
pdflatex poster.tex
```

Required packages (all in TeX Live full / MiKTeX): `geometry`, `graphicx`,
`xcolor`, `tcolorbox` (with `skins` lib), `enumitem`, `booktabs`,
`fontawesome5`, `helvet`.
