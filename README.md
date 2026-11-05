# 📡 Churn Radar: from notebook to product

The worked example from the **Capstone · Deployment** deck: a churn model built two ways.

| Before (model demo) | After (this app, for a retention manager) |
|---|---|
| Type 12 numbers for one customer | Upload this month's customer list, or use sample data |
| Click Predict | See a ranked table of who is most at risk |
| See “Churn: 1” | Open a customer to see the top 3 reasons |
| …then what? | Try a what-if offer and watch the risk change |
| | Download the at-risk list for the call team |

**Live app:** _add your Streamlit Community Cloud URL here after deploying_

## Data

Public **[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)** dataset:
7,043 customers of a fictional US telecom provider, with their services, account details and whether they churned.
The notebook downloads it directly from GitHub; nothing private is used.

## Start here: [`index.ipynb`](index.ipynb)

The index notebook walks through the whole path and produces the files the app needs:

1. Think product first (user, job, decision)
2. Load the public data
3. Clean & explore
4. One feature pipeline, shared by training and the app (`src/features.py`)
5. Train & evaluate (ROC AUC ≈ 0.84 on a held-out 20%)
6. Explain each prediction in plain words (`src/model.py`)
7. Save `models/churn_v1.joblib` and `data/sample.csv`
8. Smoke-test exactly what the app runs
9. Deploy, and the definition-of-done checklist

## Repo layout

```
.
├── index.ipynb              # start here: data → model → saved files
├── app.py                   # entry point, st.navigation
├── pages/
│   ├── predict.py           # upload/sample → ranked list, reasons, what-if, download
│   ├── explore.py           # compare segments, overall risk drivers
│   └── about.py             # data, metrics, limits, feedback
├── src/
│   ├── features.py          # same prep as training + input validation
│   ├── model.py             # load, predict, explain
│   └── loaders.py           # @st.cache_resource / @st.cache_data loaders
├── models/churn_v1.joblib   # pipeline + baseline + metadata (3 KB)
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

1. **Pin versions:** done in `requirements.txt`. scikit-learn must match the version the model was trained with.
2. **Push to GitHub:** code, `models/` and `data/sample.csv` are all small.
3. **Connect:** sign in at [share.streamlit.io](https://share.streamlit.io), pick this repo, branch `main`, file `app.py`.
   Under *Advanced settings*, choose **Python 3.12**.
4. **Secrets:** none needed. If you add any, use the app's *Secrets* settings and `st.secrets`. Never commit `.streamlit/secrets.toml`.
5. **Share, iterate:** every `git push` redeploys. Check the logs when something breaks.

## Limits

- Scores are associations, not causes. The what-if shows how the *model* responds, not a guaranteed customer reaction.
- Price discounts aren't offered as a what-if: in this data, bill size mostly reflects which services a customer has, so the model can't isolate the effect of price.
- Fictional, single-snapshot data. Retrain on real data before using it for real decisions.
- Gender and senior-citizen status are model inputs. Review fairness before using scores to decide who gets offers.
