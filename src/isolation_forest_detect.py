from sklearn.ensemble import IsolationForest

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


def run(X_train, X_test, y_test, contamination) -> dict:
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

    tp = int(((pred_anomaly == 1) & (y_test == 1)).sum())
    fp = int(((pred_anomaly == 1) & (y_test == 0)).sum())
    fn = int(((pred_anomaly == 0) & (y_test == 1)).sum())
    tn = int(((pred_anomaly == 0) & (y_test == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": prec,
        "Recall": rec,
    }
