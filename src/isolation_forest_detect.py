from sklearn.ensemble import IsolationForest

from data import load_splits

# --- Training parameters ---
N_ESTIMATORS = 500  # number of trees in the forest
MAX_SAMPLES = 256  # subsample size per tree
MAX_FEATURES = 1.0  # fraction of features per tree
RANDOM_STATE = 42

FEATURES = [
    "ct_state_ttl",
    "sbytes",
    "tcprtt",
    "sttl",
    "dttl",
    "rate",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
]


def print_metrics(labels, pred_anomaly):
    tp = int(((pred_anomaly == 1) & (labels == 1)).sum())
    fp = int(((pred_anomaly == 1) & (labels == 0)).sum())
    fn = int(((pred_anomaly == 0) & (labels == 1)).sum())
    tn = int(((pred_anomaly == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={prec:.4f} Recall={rec:.4f}")


def run(X_train, X_test, y_test, contamination, label):
    print(f"=== {label} ===")
    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=contamination,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
    )
    iso_forest.fit(X_train)

    preds = iso_forest.predict(X_test)
    pred_anomaly = (preds == -1).astype(int)
    print_metrics(y_test, pred_anomaly)


def main():
    train, test = load_splits()
    y_test = test["label"]
    drop = ["id", "label", "attack_cat"]

    X_train_all = train.drop(columns=drop).select_dtypes(include="number")
    X_test_all = test.drop(columns=drop).select_dtypes(include="number")
    run(X_train_all, X_test_all, y_test, 0.5, "General (all features, mixed training)")

    missing = [f for f in FEATURES if f not in train.columns or f not in test.columns]
    if missing:
        raise KeyError(f"FEATURES not found in splits: {missing}")
    run(train[FEATURES], test[FEATURES], y_test, 0.5, "Feature selection")

    normal = train[train["label"] == 0]
    X_train_normal = normal.drop(columns=drop).select_dtypes(include="number")
    run(X_train_normal, X_test_all, y_test, 0.01, "Normal-only training")


if __name__ == "__main__":
    main()
