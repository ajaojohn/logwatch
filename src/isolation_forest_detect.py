import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.ensemble import IsolationForest

from data import load_splits

# --- Training parameters ---
N_ESTIMATORS = 500  # number of trees in the forest
MAX_SAMPLES = 256  # subsample size per tree
MAX_FEATURES = 1.0  # fraction of features per tree
RANDOM_STATE = 42

FEATURES = ["ct_state_ttl", "sbytes", "tcprtt", "sttl", "dttl",
            "rate", "sload", "dload", "sinpkt", "dinpkt"]


def print_metrics(labels, pred_anomaly):
    tp = int(((pred_anomaly == 1) & (labels == 1)).sum())
    fp = int(((pred_anomaly == 1) & (labels == 0)).sum())
    fn = int(((pred_anomaly == 0) & (labels == 1)).sum())
    tn = int(((pred_anomaly == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={prec:.4f} Recall={rec:.4f}")


def run_general():
    train, test = load_splits()
    drop = ["id", "label", "attack_cat"]
    X_train = train.drop(columns=drop).select_dtypes(include="number")
    X_test = test.drop(columns=drop).select_dtypes(include="number")

    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=0.5,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
    )
    iso_forest.fit(X_train)

    preds = iso_forest.predict(X_test)
    pred_anomaly = (preds == -1).astype(int)
    print_metrics(test["label"], pred_anomaly)


def run_feature_selection():
    train, test = load_splits()
    X_train = train[FEATURES]
    X_test = test[FEATURES]

    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=0.5,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
    )
    iso_forest.fit(X_train)

    preds = iso_forest.predict(X_test)
    pred_anomaly = (preds == -1).astype(int)
    print_metrics(test["label"], pred_anomaly)


def run_normal_only():
    train, test = load_splits()
    drop = ["id", "label", "attack_cat"]
    normal = train[train["label"] == 0]
    X_train = normal.drop(columns=drop).select_dtypes(include="number")
    X_test = test.drop(columns=drop).select_dtypes(include="number")

    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=0.01,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
    )
    iso_forest.fit(X_train)

    preds = iso_forest.predict(X_test)
    pred_anomaly = (preds == -1).astype(int)
    print_metrics(test["label"], pred_anomaly)


def main():
    print("=== General (all features, mixed training) ===")
    run_general()
    print("\n=== Feature selection ===")
    run_feature_selection()
    print("\n=== Normal-only training ===")
    run_normal_only()


if __name__ == "__main__":
    main()
