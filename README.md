# logwatch

LogWatch — a network intrusion detection system that combines rule-based, ML, and LLM-powered analysis on real attack data.

Built on the UNSW-NB15 dataset (~2.5M labeled network events), LogWatch runs three detection layers:

Rule-based detectors — explicit signatures for known attack patterns
Isolation Forest — unsupervised anomaly detection to catch what rules miss
LLM triage — Claude API classifies alert severity and surfaces context

Results surface in a minimal Flask dashboard: alert table sorted by severity, per-method detection counts, and an overlap chart comparing what each layer caught.

Stack: Python, scikit-learn, Flask, SQLite
