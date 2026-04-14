import pandas as pd

from data import load_files


def compare_means(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Per-feature mean for `category` vs. other attacks vs. Normal.

    Good rule candidates: features where `category` differs from both
    `other_attacks` (specific) and `normal` (separates from benign).
    """
    numeric = df.select_dtypes(include="number").drop(
        columns=["Label"], errors="ignore"
    )

    cat = df["attack_cat"]

    target_mask = cat == category
    other_mask = (df["Label"] == 1) & (cat != category)
    normal_mask = cat == "Normal"

    out = pd.DataFrame(
        {
            "normal": numeric[normal_mask].mean(),
            "other_attacks": numeric[other_mask].mean(),
            category: numeric[target_mask].mean(),
        }
    )

    # replace(0, pd.NA) to avoid division-by-zero on always-zero features.
    out["ratio_vs_other"] = out[category] / out["other_attacks"].replace(0, pd.NA)
    out["ratio_vs_normal"] = out[category] / out["normal"].replace(0, pd.NA)

    return out.sort_values("ratio_vs_other", ascending=False, na_position="last")


def feature_distribution(df: pd.DataFrame, feature: str, category: str) -> pd.DataFrame:
    """Per-group percentiles for `feature`, to pick rule thresholds.

    A clean threshold exists when `category`'s low percentiles sit above
    `normal`/`other_attacks`'s high percentiles (or vice versa).
    """
    cat = df["attack_cat"]
    col = df[feature]

    groups = {
        "normal": col[cat == "Normal"],
        "other_attacks": col[(df["Label"] == 1) & (cat != category)],
        category: col[cat == category],
    }

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = {}
    for label, s in groups.items():
        stats = s.quantile(qs).to_dict()
        stats["min"] = s.min()
        stats["max"] = s.max()
        stats["mean"] = s.mean()
        stats["n"] = len(s)
        rows[label] = stats

    cols = ["n", "min"] + [f"p{int(q * 100)}" for q in qs] + ["max", "mean"]
    out = pd.DataFrame(rows).T
    out.columns = [
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "min",
        "max",
        "mean",
        "n",
    ]
    return out[cols]


def main():
    df = load_files(2)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:.4g}".format)

    print("\n=== Reconnaissance: feature ranking ===")
    print(compare_means(df, "Reconnaissance"))

    # Top behavioural candidates from compare_means (skipping sttl/dttl —
    # likely OS/testbed artifacts).
    for feat in ["tcprtt", "ackdat", "synack", "ct_state_ttl"]:
        print(f"\n=== Reconnaissance: {feat} distribution ===")
        print(feature_distribution(df, feat, "Reconnaissance"))

    print("\n=== DoS: feature ranking ===")
    print(compare_means(df, "DoS"))


if __name__ == "__main__":
    main()
