# Thesis report

This directory contains an empty thesis template based on the UCM TeXiS template
linked from the [2026/2027 TFG page](https://informatica.ucm.es/tfgs-2026-2027).
The cover data and chapter headings are retained. Chapter bodies, abstracts,
keyword lists and bibliography entries are empty.

## Report files

- `main.tex`: document order and template setup.
- `metadata.tex`: titles, author, degree, supervisor, academic year and submission details.
- `sections/`: chapters, the English abstract and the Spanish summary and keywords.
- `english_layout.tex`: English cover labels, academic year and caption overrides.
- `figures/`: figures cited in the text and the official crest.
- `references.bib`: empty bibliography database.
- `TeXiS/`: original template support files with their authorship and licence notices.

Edit cover information in `metadata.tex`. The cover adds "Bachelor's Degree in"
before `degreeName`. An empty `collaboratorName` hides the collaborator row on both
title pages; an empty `examSession` hides the submission-period label. The submission
date is displayed when `submissionDate` contains a value.

## Compile

### Windows with MiKTeX

1. Install [MiKTeX](https://miktex.org/download) for your user. In MiKTeX Console,
   enable automatic installation of missing packages. The first build needs an
   internet connection to download the packages used by the UCM template.
2. Install VS Code's LaTeX Workshop extension and open the repository folder.
   The included **Build thesis with MiKTeX** recipe is the default.
3. Edit `metadata.tex` or a file in `sections/` and save it. The root directives
   tell LaTeX Workshop to compile `main.tex`. Use **LaTeX Workshop: Build LaTeX
   project** from the command palette to start a build manually, then **LaTeX
   Workshop: View LaTeX PDF file** to view the output.

Alternatively, from the repository root in Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\report\build.ps1
```

This execution-policy option applies only to the current command. `build.ps1`
locates MiKTeX through PATH or its standard Windows installation directories,
then runs `texify` to repeat pdfLaTeX and BibTeX as needed. No Perl installation
or `latexmk` is needed for this recipe. The helper also works before an already
open VS Code window refreshes its PATH.

### Linux/macOS or an existing latexmk setup

Install TeX Live or MacTeX with pdfLaTeX, BibTeX, `latexmk` and the packages used
by the template. From the repository root:

```text
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

In VS Code, select **Build thesis with latexmk** through **LaTeX Workshop: Build
with recipe**. To make it the default, set `latex-workshop.latex.recipe.default`
to `Build thesis with latexmk` in the workspace settings.

For the empty template, you can also run pdfLaTeX twice from `report/`:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `report/main.pdf`. Build files and this generated PDF are ignored by
Git. Keep `main.tex` as the root document when importing the report into Overleaf.
After changing the text, inspect the cover, long headings, contents and references.
The upstream template emits non-fatal warnings about preamble `\include` commands
and its legacy `pdfborder` option.

## Add bibliography entries

BibTeX is disabled while the bibliography is empty. When you add entries to
`references.bib` and cite them in the text, remove `\chapter*{\bibname}` from
the references block in `main.tex` and uncomment its `\bibliographystyle` and
`\bibliography` commands. The normal build helper will then run BibTeX.
For manual compilation, run pdfLaTeX, `bibtex main`, then pdfLaTeX twice.

## Faculty requirements

The [faculty regulations, section V](https://informatica.ucm.es/file/normativa-tfg-actual?ver=)
require bilingual titles, abstracts and keyword lists (at most ten keywords),
contents, an introduction, results, discussion, conclusions and references.
The template provides headings for the English chapters and both language versions
of the abstract and keywords. The counted body must reach
25 pages for one student, plus five per additional student. Group contributions
need their own statements. The submitted cover must use the registered information
and include grades only once awarded.

## Template provenance

Downloaded on 2026-09-12 from the
[official LaTeX archive](https://informatica.ucm.es/file/plantilla_tfg_latex?ver).
The ZIP's SHA-256 is
`1f253123e00e418fb0a5c7ad7f649b81f3571e3c442008ade57d1d6364034a1f`.
Individual preserved-file hashes are in `template_provenance.json`.

The original `TeXiS/` files and the UCM crest are unchanged. Project root and
section files provide the empty document structure. `english_layout.tex` translates
cover labels, omits an empty collaborator row and sets the rendered academic year
from `metadata.tex`. Local contents use English Babel captions, and the bibliography
uses the upstream `TeXiS_en` style.

The original TeXiS code credits Marco Antonio Gómez-Martín and Pedro Pablo Gómez-Martín
and declares the LaTeX Project Public License, version 1.3 or later. Preserve upstream
notices and respect the crest's institutional provenance. The upstream manual and
example thesis PDFs are not copied into the project.
