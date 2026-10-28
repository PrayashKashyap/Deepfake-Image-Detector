# IEEE paper (IEEEtran, conference format)

- `main.tex` - the paper (pdfLaTeX).
- `generated/` - tables (`*.tex`) and figures, produced from the experiment result files.
  Do not edit them by hand; regenerate with (repo root, venv active):

      python -m ml.experiments.paper_assets --out paper/generated --dff-root C:/Users/ASUS/datasets/DeepFakeFace

## Compile
Overleaf: New Project -> Upload Project -> `paper_overleaf.zip`, compiler pdfLaTeX (IEEEtran is built in).
Local TeX Live / MiKTeX: `pdflatex main && pdflatex main`.

## Before submission
- Replace author names/affiliations.
- Remove every red `[PENDING ...]` marker: rerun the generator after the remaining experiments finish,
  then rewrite the two pending paragraphs in Sec. V with the measured results.
- Fig. "Grad-CAM" shows celebrity photos from DeepFakeFace/IMDB-WIKI; confirm the dataset terms and your
  venue's policy allow reproducing them (or blur faces / show only heatmaps).
