Stand-in experiment on the UCI Turkish Music Emotion dataset (id 862).

Question: can a model chosen by cross-validated search give better music-emotion suggestions (macro-F1 on a held-out split) than a default random forest?

How to run: `pip install -r requirements.txt && python analysis.py`. Writes `results.json` and `figures/baseline_vs_improved.png`.

Method: one stratified 80/20 split (seed 42). The baseline is a default random forest. The improved model is chosen from a small candidate set (RBF SVM, a soft-voting ensemble of SVM + random forest + LightGBM, and a LightGBM grid), all class-balanced. Selection uses 5-fold stratified CV macro-F1 on the training split only. The chosen model is refit with seeds 42, 43 and 44, and the test set is scored once per model.

Caveats: the test split is small (about 80 songs), so differences are noisy. The seeds capture model randomness only. The paper's own method is not reproduced.

## Reference

This project is motivated by the following paper:

> JPM Avila, A Hazzard, A Chamberlain (2018). *An AI-based design framework to support musicians' practices*. Proceedings of the. https://dl.acm.org/doi/abs/10.1145/3243274.3275381

Paper author: Prof. Chris Greenhalgh, University of Nottingham (chris.greenhalgh@nottingham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
