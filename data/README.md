# data/

This folder holds the raw/derived tables the dashboard uses for team lists, seeds and historical
matchups. They are **not tracked in Git** (see root `.gitignore`) — `data_men.pkl` alone is ~240 MB,
well over GitHub's 100 MB file limit.

Place these two files here before running the app:

| File | Contents |
|---|---|
| `data_men.pkl`   | Teams, seeds and tournament results used by the men's dashboard views |
| `data_women.pkl` | Teams, seeds and tournament results used by the women's dashboard views |

Source data comes from the Kaggle March Madness competition datasets, processed by the Databricks
notebooks used for training. If you don't have access to those, the dashboard still runs on the
model bundles alone — pages that rely on historical tournament data (e.g. "Model accuracy") will
be skipped with a warning.
