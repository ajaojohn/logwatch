from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "UNSW-NB15"
SPLITS_DIR = DATA_DIR / "Training and Testing Sets"
CACHE_DIR = DATA_DIR / ".cache"


def load_files(count: int) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"shards_{count}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    files = [DATA_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, count + 1)]
    dfs = [pd.read_csv(f, header=None, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    features = pd.read_csv(DATA_DIR / "NUSW-NB15_features.csv", encoding="latin-1")
    df.columns = features["Name"].values

    # NaN attack_cat means benign; strip handles stray whitespace like "Reconnaissance ".
    df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLITS_DIR / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(SPLITS_DIR / "UNSW_NB15_testing-set.csv")
    return train, test
