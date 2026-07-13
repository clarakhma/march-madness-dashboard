# models/

This folder holds the trained model bundles consumed by the dashboard. They are **not tracked in
Git** (see root `.gitignore`) because of their size (~25 MB and ~17 MB).

`dashboard.py` downloads them automatically on first run from the repo's
[`data-v1` GitHub Release](https://github.com/clarakhma/march-madness-dashboard/releases/tag/data-v1)
and caches them here — no manual step needed for a normal `streamlit run`.

| File | Contents |
|---|---|
| `bundle_men.pkl`   | Trained classifier + feature list + metrics for the men's tournament |
| `bundle_women.pkl` | Trained classifier + feature list + metrics for the women's tournament |

Both bundles are produced by the feature engineering / model selection pipeline that runs in
Databricks (forward-chaining cross-validation over the Kaggle March Madness dataset). If you'd
rather regenerate them yourself, re-run that pipeline and copy the `.pkl` outputs from your
Databricks workspace / DBFS into this folder — the app will skip the download if the files
already exist.
