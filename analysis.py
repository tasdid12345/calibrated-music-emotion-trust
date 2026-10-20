"""Stand-in experiment: standard random forest (baseline) vs class-balanced,
isotonic-calibrated LightGBM (improved) on the UCI Turkish Music Emotion dataset."""
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    Xtr, Xte, ytr_raw, yte_raw = train_test_split(
        X, y_raw, test_size=TEST_SIZE, stratify=y_raw, random_state=SEED)
    le = LabelEncoder().fit(ytr_raw)
    ytr, yte = le.transform(ytr_raw), le.transform(yte_raw)
    K = len(le.classes_)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    print(f"train {len(Xtr)}, test {len(Xte)}, classes {list(le.classes_)}")

    # Small hyper-parameter search with CV on the TRAINING set only (test set untouched)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31]}
    search = GridSearchCV(LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
                          grid, scoring="f1_macro", cv=cv)
    search.fit(Xtr, ytr)
    print(f"best LightGBM params (train CV): {search.best_params_}")

    base_runs, imp_runs = [], []
    for s in SEEDS:
        print(f"seed {s}: fitting baseline random forest")
        rf = RandomForestClassifier(random_state=s).fit(Xtr, ytr)
        base_runs.append(evaluate(rf, Xte, yte, K))

        print(f"seed {s}: fitting balanced LightGBM + isotonic calibration")
        lgb = LGBMClassifier(class_weight="balanced", random_state=s, verbose=-1,
                             **search.best_params_)
        cal = CalibratedClassifierCV(lgb, method="isotonic", cv=cv).fit(Xtr, ytr)
        imp_runs.append(evaluate(cal, Xte, yte, K))

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
        ax.bar(["baseline RF", "improved LGBM"], [np.mean(b), np.mean(i)],
               yerr=[np.std(b), np.std(i)], color=["grey", "tab:blue"], capsize=4)
        ax.set_title(title)
    fig.suptitle("Turkish Music Emotion, test split, mean over seeds 42-44")
    fig.tight_layout()
    fig.savefig("figures/baseline_vs_improved.png", dpi=120)

    results = {
        "question": "Can a class-balanced, calibrated gradient-boosting model give more accurate "
                    "and better-calibrated music-emotion suggestions than a standard random forest "
                    "on a public music-audio feature dataset?",
        "method": "Stratified 80/20 split (seed 42); preprocessing fitted on train only; baseline "
                  "random forest with default settings vs class-balanced LightGBM with isotonic "
                  "calibration (hyper-parameters chosen by 5-fold CV on train). Both run with seeds "
                  "42, 43, 44; test set used once per model.",
        "dataset": {
            "name": "Turkish Music Emotion (UCI)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test split)",
        "higher_is_better": True,
        "baseline": "Standard RandomForestClassifier with default settings (generic suggestion model).",
        "improved": "Class-balanced LightGBM classifier (class_weight='balanced', small CV grid on "
                    "train) with cross-validated isotonic probability calibration.",
        "metrics": metrics,
        "notes": "Single stratified split, so the test set is small (about 80 songs) and differences "
                 "are noisy; the three seeds capture model randomness only, not split variability. "
                 "Isotonic calibration is fitted by internal CV on a small training set and can be "
                 "noisy. Brier score is a calibration proxy for trust, not a user-trust measurement. "
                 "This is a stand-in experiment; the motivating paper's model and results are not "
                 "stated and are not reproduced. The ucimlrepo import of dataset 862 is refused by "
                 "the service ('not available for import'), so the script falls back to downloading "
                 "the same official UCI zip directly; the data content is identical.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"done in {time.time() - t0:.1f}s; wrote results.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
