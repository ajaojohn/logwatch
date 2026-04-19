import pandas as pd

from data import load_files, load_splits
from feature_analysis import compare_means, feature_distribution
from isolation_forest_detect import FEATURES, run
from rule_based_detect import RULES, eval_rule


def _configure_pandas_display() -> None:
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:.4g}".format)


def _section(title: str) -> None:
    bar = "#" * 72
    print(f"\n\n{bar}\n#   {title}\n{bar}")


def _run_feature_analysis(df: pd.DataFrame, category: str, top_k: int = 4) -> None:
    ranking = compare_means(df, category)
    print(f"\n=== {category}: top {top_k} feature ranking ===")
    print(ranking.head(top_k))

    for feat in ranking.head(top_k).index:
        print(f"\n=== {category}: {feat} distribution ===")
        print(feature_distribution(df, feat, category))


def _run_rule_experiments(df: pd.DataFrame) -> None:
    for rule in RULES:
        eval_rule(df, rule)


def _run_iso_forest_experiments() -> None:
    train, test = load_splits()
    y_test = test["label"]
    drop = ["id", "label", "attack_cat"]

    missing = [f for f in FEATURES if f not in train.columns or f not in test.columns]
    if missing:
        raise KeyError(f"FEATURES not found in splits: {missing}")

    X_train_all = train.drop(columns=drop).select_dtypes(include="number")
    X_test_all = test.drop(columns=drop).select_dtypes(include="number")

    normal = train[train["label"] == 0]
    X_train_normal = normal.drop(columns=drop).select_dtypes(include="number")

    results = {
        "All features, mixed training (c=0.5)": run(
            X_train_all, X_test_all, y_test, 0.5
        ),
        "Feature selection (c=0.5)": run(
            train[FEATURES], test[FEATURES], y_test, 0.5
        ),
        "Normal-only training (c=0.01)": run(
            X_train_normal, X_test_all, y_test, 0.01
        ),
    }

    print()
    print(pd.DataFrame.from_dict(results, orient="index"))


def run_experiments() -> None:
    _configure_pandas_display()
    df = load_files(2)

    _section("FEATURE ANALYSIS")
    _run_feature_analysis(df, "Reconnaissance")
    _run_feature_analysis(df, "DoS")

    _section("RULE-BASED DETECTION")
    _run_rule_experiments(df)

    _section("ISOLATION FOREST")
    _run_iso_forest_experiments()


if __name__ == "__main__":
    run_experiments()
