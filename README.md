# Turkish Music Emotion: random forest vs class-balanced calibrated LightGBM

Stand-in experiment comparing a default RandomForestClassifier (baseline) with a
class-balanced LightGBM classifier using isotonic calibration (improved) on the
UCI Turkish Music Emotion dataset (id 862).

## Data loading
The `ucimlrepo` API refuses to import dataset 862 ("not available for import"), so
`analysis.py` first tries `ucimlrepo` and, if that fails, downloads the official UCI
zip directly with `requests` and reads the spreadsheet with `pandas`/`openpyxl`.
The data is the same; the method is unchanged.

## Run
    pip install -r requirements.txt
    python analysis.py

Outputs `results.json` and `figures/baseline_vs_improved.png`. Seed 42; all tuning is
5-fold CV on the training split; the test split is used once per model.

## Reference

This project is motivated by the following paper:

> JPM Avila, A Hazzard, A Chamberlain (2018). *An AI-based design framework to support musicians' practices*. Proceedings of the. https://dl.acm.org/doi/abs/10.1145/3243274.3275381

Paper author: Prof. Chris Greenhalgh, University of Nottingham (chris.greenhalgh@nottingham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
