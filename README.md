# Music-Emotion Suggestions: Random Forest vs Calibrated LightGBM

Motivation: Tools that support musicians' practices, such as the design framework in "An AI-based design framework to support musicians' practices" (2018), depend on suggestions a user can trust. A suggestion model that is accurate on average but poorly calibrated, or that ignores minority emotion classes, may give misleading feedback. This experiment checks a simple, cheap alternative on a small public music-audio dataset.

Question: Can a class-balanced, calibrated gradient-boosting model give more accurate and better-calibrated music-emotion suggestions than a standard random forest on a public music-audio feature dataset?

Dataset: UCI Turkish Music Emotion (about 400 songs, audio-derived features, four emotion classes), loaded with the `ucimlrepo` package. This is a public stand-in dataset, not the paper's own data (`same_as_paper: false`).

Baseline vs improved model:
- Baseline: standard random forest classifier with default settings.
- Improved: class-balanced LightGBM classifier with a small cross-validated hyper-parameter search on the training split, and cross-validated isotonic probability calibration.

Both models use the same stratified 80/20 split (seed 42), the same scaling fitted on the training split, and the same evaluation code. The main metric is macro-F1 on the test split; accuracy and multiclass Brier score are secondary.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts are saved to `figures/`).

## Reference

This project is motivated by the following paper:

> JPM Avila, A Hazzard, A Chamberlain (2018). *An AI-based design framework to support musicians' practices*. Proceedings of the. https://dl.acm.org/doi/abs/10.1145/3243274.3275381

Paper author: Prof. Chris Greenhalgh, University of Nottingham (chris.greenhalgh@nottingham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
