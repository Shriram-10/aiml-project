# Football Match Outcome Prediction

```cmd
git clone https://github.com/Shriram-10/aiml-project
cd aiml-project
```

What you'll find here
- `main.py`: the full pipeline (read data, make features, train models, evaluate, save models and images).
- `predict.py`: a small helper to load the saved model and produce predictions from a prepared CSV.
- `data/`: where input CSVs go (`data/Datasets/` for per-season files or a single `final_dataset.csv`).
- `models/` and `figures/`: populated by running the pipeline.
- `REPORT_FINAL.md` / `REPORT_FINAL.html`: a readable project report with results and interpretation.

Dataset 
This project uses historical English league season results CSVs. A commonly used public source is the "EPL Match Results" dataset on Kaggle (example):

- Kaggle (season-by-season results): https://www.kaggle.com/datasets/saife245/english-premier-league

Quick notes before running
- The code expects columns like `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`. If your files are named differently, rename them or tweak `main.py`.
- `MW` (matchweek) is optional but helps create a few extra plots.
- The pipeline saves `feature_columns.pkl` and `scaler.pkl`; keep those if you want reproducible predictions.

Exact run steps (preserved)
1. Create and activate a virtual environment (Windows cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare the data: either place per-season files in `./data/Datasets/` or put a consolidated `final_dataset.csv` in `./data/`.

3. Run the pipeline:

```cmd
python main.py
```

Troubleshooting (quick)
- If imports fail, re-run `pip install -r requirements.txt`. Add `shap` / `umap-learn` only if you're generating those plots.
- If long GridSearch jobs hang on Windows, stop the run and re-run; the code uses single-threaded CV by default to avoid this.
- `SettingWithCopyWarning` from pandas is noisy but non-fatal; it indicates some in-place assignment that can be cleaned up later.




