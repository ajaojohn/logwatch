# logwatch

LogWatch — a network intrusion detection system that combines rule-based, ML, and LLM-powered analysis on real attack data.

Built on the UNSW-NB15 dataset (~2.5M labeled network events), LogWatch runs three detection layers:

- **Rule-based detectors** — explicit signatures for known attack patterns
- **Isolation Forest** — unsupervised anomaly detection to catch what rules miss
- **LLM triage** — Claude API classifies alert severity and surfaces context

Results surface in a minimal Flask dashboard: alert table sorted by severity, per-method detection counts, and an overlap chart comparing what each layer caught.

**Stack:** Python, scikit-learn, Flask, SQLite

## Dataset setup

LogWatch uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) from UNSW Canberra. The dataset is not included in this repo — you need to download it separately.

### Download

1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Follow the download link to the Cloudstor/UNSW mirror
3. Grab the **CSV Files** directory (the raw shards) and the **Training and Testing Sets** directory

### Directory layout

Place the files under `data/UNSW-NB15/` so the tree looks like this:

```
data/
└── UNSW-NB15/
    ├── NUSW-NB15_features.csv         # column schema for the raw shards
    ├── NUSW-NB15_GT.csv                # ground-truth labels
    ├── UNSW-NB15_LIST_EVENTS.csv       # attack event listing
    ├── UNSW-NB15_1.csv                 # raw shard 1
    ├── UNSW-NB15_2.csv                 # raw shard 2
    ├── UNSW-NB15_3.csv                 # raw shard 3
    ├── UNSW-NB15_4.csv                 # raw shard 4
    └── Training and Testing Sets/
        ├── UNSW_NB15_training-set.csv
        └── UNSW_NB15_testing-set.csv
```

The loader in `src/data.py` resolves paths relative to the repo root, so this is the only supported layout. A `.cache/` folder will be created here on first run to store pickled shards for faster reloads.

### Loading

- `load_files(n)` concatenates the first `n` raw shards (`UNSW-NB15_1.csv` … `UNSW-NB15_n.csv`), attaches column names from `NUSW-NB15_features.csv`, and normalizes `attack_cat` (NaN → `"Normal"`).
- `load_splits()` returns the pre-split training and testing sets used for ML evaluation.

## Quickstart

Uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv venv
uv pip install -r requirements.txt

# after placing the dataset under data/UNSW-NB15/
uv run src/rule_based_detect.py
uv run src/isolation_forest_detect.py
```
