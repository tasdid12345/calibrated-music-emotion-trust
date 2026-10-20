Stand-in experiment on the UCI Turkish Music Emotion dataset (id 862).

Question: can a model chosen by cross-validated search give better music-emotion suggestions (macro-F1 on a held-out split) than a default random forest?

How to run: `pip install -r requirements.txt && python analysis.py`. Writes `results.json` and `figures/baseline_vs_improved.png`.

Method: one stratified 80/20 split (seed 42). The baseline is a default random forest. The improved model is chosen from a small candidate set (RBF SVM, a soft-voting ensemble of SVM + random forest + LightGBM, and a LightGBM grid), all class-balanced. Selection uses 5-fold stratified CV macro-F1 on the training split only. The chosen model is refit with seeds 42, 43 and 44, and the test set is scored once per model.

Caveats: the test split is small (about 80 songs), so differences are noisy. The seeds capture model randomness only. The paper's own method is not reproduced.

## Results (latest run)

On the Turkish Music Emotion (UCI) test split (80 songs), a standard random-forest classifier reached 75.2% macro-F1 and 0.347 Brier, while my selected class-balanced ensemble reached 75.3% macro-F1 and 0.307 Brier, a negligible F1 gain within split noise.

| Metric | Value |
|---|---|
| baseline_score | 0.7518 |
| baseline_score_std | 0.0163 |
| improved_score | 0.7535 |
| improved_score_std | 0 |
| improvement | 0.001675 |
| baseline_accuracy | 0.7583 |
| improved_accuracy | 0.7625 |
| baseline_brier | 0.3466 |
| improved_brier | 0.3073 |
| brier_improvement | 0.03932 |
| best_train_cv_macro_f1 | 0.8 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Selected configuration: ensemble {'C': 10} (train CV macro-F1 0.8000). Single stratified split, so the test set is small (about 80 songs) and differences are noisy; the three seeds capture model randomness only, not split variability. The test set was not used for any selection. Brier score is a calibration proxy, not a user-trust measurement. This is a stand-in experiment; the motivating paper's model and results are not stated and are not reproduced. The ucimlrepo import of dataset 862 is refused by the service, so the script falls back to downloading the same official UCI zip directly; the data content is identical.

## Reference

This project is motivated by the following paper:

> JPM Avila, A Hazzard, A Chamberlain (2018). *An AI-based design framework to support musicians' practices*. Proceedings of the. https://dl.acm.org/doi/abs/10.1145/3243274.3275381

Paper author: Prof. Chris Greenhalgh, University of Nottingham (chris.greenhalgh@nottingham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
