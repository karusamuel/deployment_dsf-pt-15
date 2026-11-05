"""Orientation and trust: what the app does, on what data, and its limits."""
import streamlit as st

from src.features import REQUIRED_COLUMNS
from src.loaders import get_model, get_sample

bundle = get_model()
meta = bundle["meta"]

st.title("ℹ️ About & limits")
st.markdown(
    "**Churn Radar** helps a retention manager decide *who to call this month*. "
    "It scores each customer's chance of leaving, explains the score, and lets you test offers."
)

st.subheader("Data")
st.markdown(
    f"- Trained on the public **[IBM Telco Customer Churn]({meta['data_url']})** dataset "
    f"({meta['n_rows']:,} customers of a fictional US telecom provider).\n"
    f"- Model `{meta['version']}`, trained {meta['trained_on']} with scikit-learn {meta['sklearn']}."
)

st.subheader("How good is it?")
c1, c2, c3, c4 = st.columns(4)
m = meta["metrics"]
c1.metric("ROC AUC", f"{m['roc_auc']:.2f}", help="1.0 = perfect ranking, 0.5 = coin flip")
c2.metric("Recall @ 0.6", f"{m['recall']:.0%}", help="Share of real churners that land on the call list")
c3.metric("Precision @ 0.6", f"{m['precision']:.0%}", help="Share of the call list that would really churn")
c4.metric("Test customers", f"{m['n_test']:,}")

st.subheader("Limits — read before you act")
st.markdown(
    "- Scores are **associations, not causes**. A what-if shows how the *model* responds, "
    "not a guaranteed customer reaction.\n"
    "- The data is a single historical snapshot from a fictional company; retrain on your own data before real use.\n"
    "- Logistic regression keeps the explanations exact and simple, at the cost of some accuracy.\n"
    "- Gender and senior-citizen status are model inputs; review fairness before using scores to decide who gets offers."
)

st.subheader("Expected file format")
st.code(", ".join(REQUIRED_COLUMNS), language=None)
st.download_button("⬇️ Download sample CSV", get_sample().to_csv(index=False), "sample.csv", "text/csv")

st.subheader("Feedback")
with st.form("feedback"):
    rating = st.feedback("stars")
    comment = st.text_area("What would make this more useful?")
    if st.form_submit_button("Send"):
        # In a real deployment, write this to a sheet or database (credentials via st.secrets).
        st.success("Thanks — feedback received.")
