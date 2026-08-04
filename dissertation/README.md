# Writing the HumanSLAM dissertation

Open `main.tex` in VS Code. LaTeX Workshop builds on save and opens the PDF in a
VS Code tab. The generated PDF and intermediate files are placed in `build/`.

Useful commands from this directory:

```bash
latexmk -xelatex -outdir=build main.tex
latexmk -C -outdir=build main.tex
texcount -inc -sum main.tex
chktex -q main.tex
```

Use `\autocite{citation-key}` for citations and add the corresponding BibLaTeX
record to `references.bib`. Put images in `figures/`, preferably as PDF for plots
and PNG/JPEG for raster images. Use `\label{...}` with `\cref{...}` for stable
cross-references.

Before extensive writing, replace the generic title page and margins with the
official signed cover sheet. Complete the supplied Word cover sheet, export it
as `frontmatter/cover_sheet.pdf`, and the document will include it automatically.

The COMP7039 handbook specifies 10,000 words for the main body and an abstract
of no more than 250 words. Check the included chapter files with:

```bash
texcount -inc -sum main.tex
```

The handbook recommends 12-point Arial. The template uses Liberation Sans, the
Arial-compatible font installed on this machine. If Arial is installed before
submission, change `\setmainfont{Liberation Sans}` to `\setmainfont{Arial}`.
