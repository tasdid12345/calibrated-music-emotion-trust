"""Stand-in experiment: standard random forest (baseline) vs an improved model selected by
cross-validation on the TRAINING split only, on the UCI Turkish Music Emotion dataset.

Improved model: candidate set = RBF SVM (balanced, Platt-calibrated probabilities),
soft-voting ensemble of SVM + balanced random forest + balanced LightGBM, and a balanced
LightGBM grid. Each candidate is scored by 5-fold stratified CV macro-F1 on the training split;
the best one is refit (with seeds 42, 43, 44) and evaluated once on the held-out test split."""
import io
import json
import os
import socket
import sys
import time
import warnings
import zipfile

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 42
SEEDS = [42, 43, 44]          # both models are run with these seeds
DATASET_ID = 862              # UCI "Turkish Music Emotion"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/862/turkish+music+emotion"
ZIP_URL = "https://archive.ics.uci.edu/static/public/862/turkish+music+emotion.zip"
MAX_ROWS = 200_000            # subsample cap (data is ~400 rows, so not triggered)
TEST_SIZE = 0.2

warnings.filterwarnings("ignore")  # silence LightGBM/sklearn chatter in CI logs


def _load_via_ucimlrepo():
    """First attempt: the ucimlrepo API. Raises if the dataset is not importable."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=DATASET_ID)
    X = ds.data.features
    y = ds.data.targets.iloc[:, 0]
    return X, y


def _load_via_zip(retries=5):
    """Fallback: download the official static zip from the UCI archive directly.
    Exponential backoff on failure, 60 s socket timeout."""
    content = None
    for attempt in range(retries):
        try:
            r = requests.get(ZIP_URL, timeout=60)
            r.raise_for_status()
            content = r.content
            break
        except Exception as err:  # HTTP 429/5xx, timeouts, connection resets
            wait = 2 ** attempt
            print(f"zip download failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    if content is None:
        sys.exit("could not download the dataset")

    zf = zipfile.ZipFile(io.BytesIO(content))
    names = zf.namelist()
    data_files = [n for n in names if n.lower().endswith((".xlsx", ".xls", ".csv"))
                  and not n.startswith("__MACOSX")]
    if not data_files:
        sys.exit(f"no data file found in zip: {names}")
    name = data_files[0]
    if name.lower().endswith(".csv"):
        df = pd.read_csv(zf.open(name))
    else:
        df = pd.read_excel(io.BytesIO(zf.read(name)), engine="openpyxl")

    label_cols = [c for c in df.columns if str(c).strip().lower() == "class"]
    label_col = label_cols[0] if label_cols else df.columns[-1]
    y = df[label_col]
    X = df.drop(columns=[label_col])
    return X, y


def load_data():
    """Return (features DataFrame, label Series) for the Turkish Music Emotion dataset."""
    socket.setdefaulttimeout(60)
    try:
        X, y = _load_via_ucimlrepo()
        print("loaded via ucimlrepo")
    except Exception as err:
        print(f"ucimlrepo import unavailable ({err}); falling back to direct zip download")
        X, y = _load_via_zip()
        print("loaded via direct UCI zip download")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")  # drop non-numeric / empty columns
    return X, y


def brier(proba, y_idx, n_classes):
    """Multiclass Brier score: mean over songs of sum_k (p_k - onehot_k)^2."""
    onehot = np.eye(n_classes)[y_idx]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(model, X, y, n_classes):
    """Shared evaluation function used for both models."""
    pred = model.predict(X)
    return {
        "f1": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "brier": brier(model.predict_proba(X), y, n_classes),
    }


def mean_std(runs, key):
    vals = np.array([r[key] for r in runs])
    return float(vals.mean()), float(vals.std())


def build(name, p, seed):
    """Improved-model candidates. Each is a pipeline that standardises raw features
    inside the pipeline, so CV folds never see validation-fold statistics."""
    if name == "svc":
        clf = SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight="balanced",
                  probability=True, random_state=seed)
    elif name == "ensemble":
        clf = VotingClassifier([
            ("svc", SVC(kernel="rbf", C=p["C"], gamma="scale", class_weight="balanced",
                        probability=True, random_state=seed)),
            ("rf", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                          random_state=seed)),
            ("lgb", LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   class_weight="balanced", random_state=seed, verbose=-1)),
        ], voting="soft")
    elif name == "lgbm":
        clf = LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1,
                             n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                             num_leaves=p["num_leaves"])
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), clf)


def candidate_grid():
    out = []
    for C in [1, 10]:
        for gamma in ["scale", 0.01]:
            out.append(("svc", {"C": C, "gamma": gamma}))
    for C in [1, 10]:
        out.append(("ensemble", {"C": C}))
    for n in [100, 300]:
        for lr in [0.05, 0.1]:
            for nl in [15, 31]:
                out.append(("lgbm", {"n_estimators": n, "learning_rate": lr, "num_leaves": nl}))
    return out


def main():
    t0 = time.time()
    feats, labels = load_data()
    X = feats.to_numpy(dtype=float)
    y_raw = labels.astype(str).to_numpy()
    keep = ~np.isnan(X).any(axis=1)  # drop rows with missing features
    X, y_raw = X[keep], y_raw[keep]
    if len(X) > MAX_ROWS:  # subsample very large data
        idx = np.random.default_rng(SEED).choice(len(X), MAX_ROWS, replace=False)
        X, y_raw = X[idx], y_raw[idx]
    print(f"loaded {len(X)} songs, {X.shape[1]} features")

    # ONE stratified split; all preprocessing is fitted on the training split only
    Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
        X, y_raw, test_size=TEST_SIZE, stratify=y_raw, random_state=SEED)
    le = LabelEncoder().fit(ytr_raw)
    ytr, yte = le.transform(ytr_raw), le.transform(yte_raw)
    K = len(le.classes_)
    scaler = StandardScaler().fit(Xtr_raw)           # baseline input, unchanged
    Xtr_s, Xte_s = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
    print(f"train {len(Xtr_raw)}, test {len(Xte_raw)}, classes {list(le.classes_)}")

    # Model selection: 5-fold stratified CV macro-F1 on the TRAINING set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best = None
    for name, p in candidate_grid():
        cv_f1 = float(cross_val_score(build(name, p, SEED), Xtr_raw, ytr, cv=cv,
                                      scoring="f1_macro").mean())
        print(f"CV macro-F1 {cv_f1:.4f}  {name} {p}")
        if best is None or cv_f1 > best[2]:
            best = (name, p, cv_f1)
    best_name, best_p, best_cv = best
    print(f"selected (train CV): {best_name} {best_p}, CV macro-F1 {best_cv:.4f}")

    base_runs, imp_runs = [], []
    for s in SEEDS:
        print(f"seed {s}: fitting baseline random forest")
        rf = RandomForestClassifier(random_state=s).fit(Xtr_s, ytr)
        base_runs.append(evaluate(rf, Xte_s, yte, K))

        print(f"seed {s}: fitting selected improved model {best_name}")
        imp = build(best_name, best_p, s).fit(Xtr_raw, ytr)
        imp_runs.append(evaluate(imp, Xte_raw, yte, K))

    b_f1, b_f1_sd = mean_std(base_runs, "f1")
    i_f1, i_f1_sd = mean_std(imp_runs, "f1")
    b_acc, _ = mean_std(base_runs, "accuracy")
    i_acc, _ = mean_std(imp_runs, "accuracy")
    b_br, _ = mean_std(base_runs, "brier")
    i_br, _ = mean_std(imp_runs, "brier")
    metrics = {
        "baseline_score": b_f1, "baseline_score_std": b_f1_sd,
        "improved_score": i_f1, "improved_score_std": i_f1_sd,
        "improvement": i_f1 - b_f1,  # positive = improved model is better
        "baseline_accuracy": b_acc, "improved_accuracy": i_acc,
        "baseline_brier": b_br, "improved_brier": i_br,
        "brier_improvement": b_br - i_br,  # positive = improved is better calibrated
        "best_train_cv_macro_f1": best_cv,
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("non-finite metrics; failing")

    # Optional chart: baseline vs improved, mean +/- std over seeds
    os.makedirs("figures", exist_ok=True)
    panels = [("macro-F1 (higher better)", "f1"), ("accuracy (higher better)", "accuracy"),
              ("Brier (lower better)", "brier")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (title, key) in zip(axes, panels):
        b = [r[key] for r in base_runs]
        i = [r[key] for r in imp_runs]
        ax.bar(["baseline RF", f"improved {best_name}"], [np.mean(b), np.mean(i)],
               yerr=[np.std(b), np.std(i)], color=["grey", "tab:blue"], capsize=4)
        ax.set_title(title)
    fig.suptitle("Turkish Music Emotion, test split, mean over seeds 42-44")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)

    results = {
        "question": "Can a model selected by cross-validated search over SVM, LightGBM and a "
                    "soft-voting ensemble give more accurate music-emotion suggestions than a "
                    "standard random forest on a public music-audio feature dataset?",
        "method": "Stratified 80/20 split (seed 42); preprocessing fitted on train only. Baseline: "
                  "random forest with default settings. Improved: candidate models (RBF SVM with "
                  "balanced class weights and Platt probabilities, soft-voting ensemble of SVM + "
                  "balanced RF + balanced LightGBM, balanced LightGBM grid) compared by 5-fold "
                  "stratified CV macro-F1 on the training split only; the best is refit with seeds "
                  "42, 43, 44 and the test set is used once per model.",
        "dataset": {
            "name": "Turkish Music Emotion (UCI)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(Xtr_raw)),
            "n_test": int(len(Xte_raw)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test split)",
        "higher_is_better": True,
        "baseline": "Standard RandomForestClassifier with default settings (generic suggestion model).",
        "improved": f"Selected by training-set CV: {best_name} with params {best_p}, "
                    "class-balanced, standardised inside a pipeline.",
        "metrics": metrics,
        "notes": f"Selected configuration: {best_name} {best_p} (train CV macro-F1 {best_cv:.4f}). "
                 "Single stratified split, so the test set is small (about 80 songs) and differences "
                 "are noisy; the three seeds capture model randomness only, not split variability. "
                 "The test set was not used for any selection. Brier score is a calibration proxy, "
                 "not a user-trust measurement. This is a stand-in experiment; the motivating "
                 "paper's model and results are not stated and are not reproduced. The ucimlrepo "
                 "import of dataset 862 is refused by the service, so the script falls back to "
                 "downloading the same official UCI zip directly; the data content is identical.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"done in {time.time() - t0:.1f}s; wrote results.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
