# models/

This folder holds the trained model bundles consumed by the dashboard. They are **not tracked in
Git** (see root `.gitignore`) because of their size (~25 MB and ~17 MB).

Place these two files here before running the app:

| File | Contents |
|---|---|
| `bundle_men.pkl`   | Trained classifier + feature list + metrics for the men's tournament |
| `bundle_women.pkl` | Trained classifier + feature list + metrics for the women's tournament |

Both bundles are produced by the feature engineering / model selection pipeline that runs in
Databricks (forward-chaining cross-validation over the Kaggle March Madness dataset). Re-run that
pipeline, or copy the `.pkl` outputs from your Databricks workspace / DBFS into this folder.
