# 🏀 March Madness 2026 — Prediction Dashboard

Interactive Streamlit dashboard that predicts NCAA March Madness matchup outcomes using a
trained machine learning model. Built for the Kaggle "March Machine Learning Mania 2026"
competition, as the final deliverable of the *Machine Learning* course (Universidad Europea).

> **Scope of this repository:** this repo contains the **dashboard / model-serving app** only.
> Feature engineering and model training (forward-chaining cross-validation, model selection,
> etc.) were done in **Databricks notebooks**, which are not included here — see
> [Project background](#project-background) below.

## Features

- **Match prediction** — pick any two teams and get a live win-probability prediction, with the
  key statistical factors driving the model's decision.
- **Title favorites** — estimated championship probability for the top 20 contenders.
- **Model accuracy (past seasons)** — backtests the model against real 2023–2025 tournament
  results to show how well its predictions held up.
- **Model metrics** — log loss, AUC and a plain-English explanation of what each metric means.

Both the men's and women's tournaments are supported, each served from its own trained model.

## Screenshots

![Match prediction view](docs/screenshots/match_prediction.png)
![Title favorites view](docs/screenshots/title_favorites.png)

*(add your own screenshots to `docs/screenshots/` — see that folder for naming suggestions)*

## Project structure

```
.
├── dashboard.py           # Streamlit app entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml        # app theme
├── models/                # trained model bundles (not tracked in git — see models/README.md)
│   └── README.md
├── data/                  # teams/seeds/results data (not tracked in git — see data/README.md)
│   └── README.md
└── docs/
    └── screenshots/
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Add the model and data files

This repo ships code only — the trained model bundles and data tables are too large for GitHub
(one file alone is ~240 MB). Place them locally as:

```
models/bundle_men.pkl
models/bundle_women.pkl
data/data_men.pkl
data/data_women.pkl
```

See [`models/README.md`](models/README.md) and [`data/README.md`](data/README.md) for details on
what each file contains and how to regenerate them.

### 3. Run the app

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Project background

This dashboard is the presentation layer of a larger pipeline:

1. **Data & feature engineering** (Databricks) — team stats, ratings, seeds and head-to-head
   history are assembled into model-ready features from the Kaggle competition data.
2. **Model training & selection** (Databricks) — several candidate models are evaluated with
   forward-only cross-validation (never trained on future seasons) and the best performer is
   exported as a `ModelBundle`.
3. **Serving (this repo)** — the Streamlit app loads the exported bundles and turns model output
   into an interactive, non-technical dashboard.

| | Men's tournament | Women's tournament |
|---|---|---|
| CV Log Loss | 0.5668 | 0.4414 |
| CV AUC | 0.7707 | n/a |
| Test Log Loss | 0.5504 | 0.4103 |
| Test AUC | 0.7997 | 0.8981 |

## Tech stack

- [Streamlit](https://streamlit.io/) — app framework
- [Plotly](https://plotly.com/python/) — charts
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [scikit-learn](https://scikit-learn.org/) — trained model

## License

MIT — see [LICENSE](LICENSE).
