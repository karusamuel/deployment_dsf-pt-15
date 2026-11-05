"""Orientation and trust: what the app does, on what data, and its limits."""
import pandas as pd
import streamlit as st

from src import charts
from src import evaluation as ev
from src.features import REQUIRED_COLUMNS
from src.loaders import get_model, get_sample

bundle = get_model()
meta = bundle["meta"]

st.title("About & limits")
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
c2.metric(f"Recall @ {m['threshold']:.1f}", f"{m['recall']:.0%}", help="Share of real churners that land on the call list")
c3.metric(f"Precision @ {m['threshold']:.1f}", f"{m['precision']:.0%}", help="Share of the call list that would really churn")
c4.metric("Test customers", f"{m['n_test']:,}")

test = ev.test_frame(bundle)
if test is None:
    st.info("Retrain with `index.ipynb` to see the evaluation charts for this model.")
else:
    threshold = st.session_state.get("call_threshold", m["threshold"])
    st.caption(f"All charts use the {len(test):,} held-out test customers the model never saw in training, "
               f"except *Model comparison* (cross-validation on the training set).")
    t_gain, t_thr, t_cmp, t_cal, t_seg = st.tabs(
        ["Who we reach", "Choosing the threshold", "Model comparison", "Calibration", "By segment"])

    with t_gain:
        gain = ev.gain_curve(test["churn"], test["risk"])
        st.altair_chart(charts.gain_chart(gain), width="stretch")
        st.caption("Calling customers in the model's order reaches far more churners than calling at random. "
                   "Hover to read any point.")

    with t_thr:
        table = ev.threshold_table(test["churn"], test["risk"])
        now = ev.threshold_table(test["churn"], test["risk"], [threshold]).iloc[0]
        st.altair_chart(charts.threshold_chart(table, threshold), width="stretch")
        st.markdown(
            f"At the current threshold of **{threshold:.0%}** you'd call **{now['share_called']:.0%}** of customers; "
            f"**{now['precision']:.0%}** of them would really leave (vs {test['churn'].mean():.0%} if calling at random), "
            f"and you'd reach **{now['recall']:.0%}** of everyone who leaves. "
            "Lower the threshold to reach more churners with a longer list; raise it for a shorter, surer list. "
            "Change it with the slider on **Score customers**."
        )
        with st.expander("Show as table"):
            st.dataframe(table, hide_index=True, column_config={
                "threshold": st.column_config.NumberColumn("Threshold", format="percent"),
                "share_called": st.column_config.NumberColumn("Customers called", format="percent"),
                "precision": st.column_config.NumberColumn("Called who'd really leave", format="percent"),
                "recall": st.column_config.NumberColumn("Churners reached", format="percent"),
            })

    with t_cmp:
        cv = pd.DataFrame(meta["evaluation"]["cv"])
        st.altair_chart(charts.model_comparison(cv, meta["evaluation"]["chosen"]), width="stretch")
        st.caption("The three real models rank customers about equally well (gaps smaller than fold-to-fold noise, grey bars), all far above "
                   "the simple \"month-to-month = at risk\" rule. Logistic regression (blue) was chosen because "
                   "its explanations are exact.")

    with t_cal:
        st.altair_chart(charts.calibration_chart(ev.calibration_bins(test["churn"], test["risk"])), width="stretch")
        st.caption("Points below the grey line mean the score overstates the real chance of leaving. "
                   "The model gives churners extra weight in training, so scores run high. "
                   "Use scores to rank and compare customers, not as exact probabilities.")

    with t_seg:
        labels = {"Contract": "Contract", "InternetService": "Internet service",
                  "SeniorCitizen": "Senior citizen", "gender": "Gender"}
        col = st.segmented_control("Segment by", list(labels), format_func=labels.get, default="Contract")
        seg = ev.segment_table(test, col or "Contract", threshold)
        st.altair_chart(charts.segment_chart(seg), width="stretch")
        st.caption("Where orange is far above blue, people in that group are called more often than they really "
                   "leave; far below means churners there are being missed.")
        with st.expander("Show as table"):
            st.dataframe(seg, hide_index=True, column_config={
                "segment": "Segment", "customers": "Customers",
                "churn_rate": st.column_config.NumberColumn("Really churned", format="percent"),
                "share_flagged": st.column_config.NumberColumn("Put on call list", format="percent"),
                "recall": st.column_config.NumberColumn("Churners reached", format="percent"),
                "roc_auc": st.column_config.NumberColumn("ROC AUC (within segment)", format="%.2f"),
            })

st.subheader("Limits")
st.markdown(
    "- Scores reflect associations, not causes. A what-if shows how the *model* responds, "
    "not a guaranteed customer reaction.\n"
    "- Scores rank customers well but run high as probabilities (see *Calibration* above).\n"
    "- The data is a single historical snapshot from a fictional company; retrain on your own data before real use.\n"
    "- Logistic regression keeps the explanations exact and simple, at the cost of some accuracy.\n"
    "- Gender and senior-citizen status are model inputs; review fairness before using scores to decide who gets offers."
)

st.subheader("Expected file format")
st.code(", ".join(REQUIRED_COLUMNS), language=None)
st.download_button("Download sample CSV", get_sample().to_csv(index=False), "sample.csv", "text/csv")

st.subheader("Feedback")
with st.form("feedback"):
    rating = st.feedback("stars")
    comment = st.text_area("What would make this more useful?")
    if st.form_submit_button("Send"):
        # In a real deployment, write this to a sheet or database (credentials via st.secrets).
        st.success("Thanks, your feedback was received.")
