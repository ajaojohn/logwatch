# Scratchpad for testing and experimenting with rule-based detection on UNSW-NB15.
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from data import load_files


@dataclass
class Rule:
    name: str  # attack_cat label
    predicate: Callable[[pd.DataFrame], pd.Series]


def eval_rule(df, rule: Rule):
    print(f"\n----- Rule: {rule.name} -----")

    rule_mask = rule.predicate(df)
    flagged = df[rule_mask]
    n_flagged = len(flagged)
    n_total = len(df)
    print(f"Flagged: {n_flagged} / {n_total} ({n_flagged / n_total:.2%})")

    def metrics(truth_mask, label):
        tp = int((rule_mask & truth_mask).sum())
        fp = int((rule_mask & ~truth_mask).sum())
        fn = int((~rule_mask & truth_mask).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  [{label}] TP={tp} FP={fp} FN={fn}  P={prec:.4f} R={rec:.4f}")

    cat = df["attack_cat"]

    metrics(cat == rule.name, f"cat={rule.name}")

    flagged_counts = cat[rule_mask].value_counts(dropna=False)
    total_counts = cat.value_counts(dropna=False)
    breakdown = pd.DataFrame(
        {
            "flagged": flagged_counts,
            "total": total_counts.reindex(flagged_counts.index),
        }
    )
    breakdown["pct"] = (breakdown["flagged"] / breakdown["total"] * 100).round(2)
    print(breakdown)


RULES = [
    Rule(
        "Reconnaissance",
        lambda d: (
            (d["ct_state_ttl"] >= 1) & (d["sbytes"] <= 600) & (d["tcprtt"] > 0.01)
        ),
    ),
    Rule(
        "DoS",
        lambda d: (d["ct_state_ttl"] >= 1) & (d["Spkts"] > 20) & (d["sbytes"] > 9000),
    ),
]


def main():
    df = load_files(2)

    for rule in RULES:
        eval_rule(df, rule)


if __name__ == "__main__":
    main()
