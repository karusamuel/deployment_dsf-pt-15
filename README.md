# Churn Radar: from notebook to product

## Data

The public [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset has
7,043 customers of a fictional US telecom provider, with their services, account details and whether they churned.
The notebook downloads it directly from GitHub. No private data is used.

## Start here: [`index.ipynb`](index.ipynb)

The index notebook is the working record of how the model was built and tested, and it produces the files the app needs:

1. Load the public data and check it (types, blanks, duplicates, target balance)
2. Explore which customers leave
3. Split off a 20% test set, untouched until the final test
4. Baselines: no model (AUC 0.50) and a "month-to-month = at risk" rule (AUC 0.73)
5. Compare logistic regression, random forest and gradient boosting with 5-fold CV; tune regularisation
6. Pick the call-list threshold from out-of-fold predictions
7. Final test, run once: ROC AUC 0.84, gain chart, calibration
8. Error analysis by segment (contract, internet, gender, senior) and the worst misses
9. Exact per-customer explanations (`src/model.py`), checked to add up to the model's output
10. Save `models/churn_v1.joblib` and `data/sample.csv`, reload, and test what the app runs

## Repo layout

```
.
├── index.ipynb              # start here: build and test the model, save files
├── app.py                   # entry point, st.navigation
├── pages/
│   ├── predict.py           # upload or sample, ranked list, reasons, what-if, download
│   ├── explore.py           # compare segments, overall risk drivers
│   └── about.py             # data, model-quality charts, limits, feedback
├── src/
│   ├── features.py          # same prep as training + input validation
│   ├── model.py             # load, predict, explain
│   ├── evaluation.py        # gain, threshold, calibration & segment tables from saved test predictions
│   ├── charts.py            # Altair charts used by the pages
│   └── loaders.py           # @st.cache_resource / @st.cache_data loaders
├── models/churn_v1.joblib   # pipeline + baseline + metadata + test predictions (20 KB)
├── data/sample.csv          # 300 unlabelled test customers for "Try it"
├── requirements.txt         # pinned app dependencies
├── requirements-dev.txt     # + Jupyter for the notebook
└── .streamlit/config.toml   # theme; hides stack traces from users
```

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

jupyter lab index.ipynb      # optional: retrain and regenerate models/ and data/
streamlit run app.py         # http://localhost:8501
```

## Deploy to Streamlit Community Cloud

1. Pin versions in `requirements.txt` (already done). scikit-learn must match the version the model was trained with.
2. Push to GitHub. The code, `models/` and `data/sample.csv` are all small enough.
3. Sign in at [share.streamlit.io](https://share.streamlit.io), pick this repo, branch `main`, file `app.py`.
   Under Advanced settings, choose Python 3.12.
4. No secrets are needed. If you add any, put them in the app's Secrets settings and read them with `st.secrets`. Never commit `.streamlit/secrets.toml`.
5. Every `git push` redeploys the app. Check the logs when something breaks.

