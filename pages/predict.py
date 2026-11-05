"""Score a customer list, explain each score, try what-ifs and export the at-risk list."""
import altair as alt
import pandas as pd
import streamlit as st

from src import charts
from src.features import LABELS, validate
from src.loaders import get_model, get_sample
from src.model import contributions, predict_risk, risk_band, top_reasons

st.title("Churn Radar")
st.markdown(
    "**Find the customers most likely to leave this month, see why, "
    "and hand your call team a prioritised list.**"
)
with st.expander("How to use"):
    st.markdown(
        "1. Upload this month's customer export (CSV, same columns as the "
        "[sample](https://github.com/IBM/telco-customer-churn-on-icp4d)), or click **Try with sample data**.\n"
        "2. Review the ranked table, highest risk first.\n"
        "3. Pick a customer to see the top reasons and test a retention offer.\n"
        "4. Download the at-risk list."
    )

bundle = get_model()

# ---- Inputs ------------------------------------------------------------------
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
file = c1.file_uploader("Upload customers (CSV)", type="csv")
if c2.button("Try with sample data", type="primary", width="stretch"):
    st.session_state.df = get_sample()
    st.session_state.source = "sample data"
if file is not None:
    try:
        st.session_state.df = pd.read_csv(file)
        st.session_state.source = file.name
    except Exception:
        st.error("We couldn't read that file. Please upload a comma-separated CSV with a header row.")
        st.stop()

if "df" not in st.session_state:
    st.info("Upload a CSV or click **Try with sample data** to get started.")
    st.stop()

df = st.session_state.df.copy()
problems = validate(df)
if problems:
    st.error("This file can't be scored yet:\n\n- " + "\n- ".join(problems))
    st.caption("Tip: download the sample from the About page to see the expected columns.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Call-list threshold (churn risk)", 0.1, 0.9,
                          st.session_state.get("call_threshold", 0.6), 0.05,
                          help="Customers at or above this risk go on the download list.")
    st.session_state.call_threshold = threshold   # widget state is dropped on page change; the About page reads this
    st.caption("The 'Choosing the threshold' tab on About & limits shows the trade-off.")

# ---- Score -------------------------------------------------------------------
try:
    with st.spinner("Scoring customers..."):
        df["risk"] = predict_risk(bundle, df)
        contrib = contributions(bundle, df)
        df["band"] = [risk_band(p, high=threshold) for p in df["risk"]]
        df["top_reason"] = [top_reasons(bundle, df.loc[i], contrib.loc[i], 1)[0] for i in df.index]
except Exception as exc:  # never show a stack trace to users
    st.error(f"Something went wrong while scoring: {exc}. Check that the values look like the sample file.")
    st.stop()

df = df.sort_values("risk", ascending=False)
at_risk = df[df["risk"] >= threshold]

st.caption(f"Scored **{len(df):,}** customers from *{st.session_state.source}*.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(df):,}")
m2.metric("On call list", f"{len(at_risk):,}", f"{len(at_risk) / len(df):.0%} of base", delta_color="off")
m3.metric("Expected churners", f"{df['risk'].sum():,.0f}")
m4.metric("Monthly revenue at risk", f"${(df['risk'] * df['MonthlyCharges']).sum():,.0f}")

tab_rank, tab_one = st.tabs(["Ranked list", "One customer: why & what-if"])

with tab_rank:
    st.altair_chart(charts.risk_histogram(df["risk"], threshold), width="stretch")
    st.dataframe(
        df[["customerID", "risk", "band", "top_reason", "Contract", "tenure", "MonthlyCharges"]],
        hide_index=True,
        width="stretch",
        column_config={
            "customerID": "Customer",
            "risk": st.column_config.ProgressColumn("Churn risk", format="percent", min_value=0, max_value=1),
            "band": "Band",
            "top_reason": "Main reason",
            "tenure": st.column_config.NumberColumn("Tenure (months)"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly bill", format="$%.2f"),
        },
    )
    st.download_button(
        f"Download call list ({len(at_risk):,} customers)",
        at_risk.drop(columns=["band"]).to_csv(index=False),
        "at_risk.csv",
        "text/csv",
        disabled=at_risk.empty,
    )

with tab_one:
    cid = st.selectbox("Customer", df["customerID"], help="Sorted by risk, highest first.")
    row = df[df["customerID"] == cid].iloc[0]
    idx = df.index[df["customerID"] == cid][0]

    left, right = st.columns(2)
    with left:
        st.metric("Churn risk", f"{row['risk']:.0%}", row["band"], delta_color="off")
        st.markdown("**Top 3 reasons**")
        for r in top_reasons(bundle, row, contrib.loc[idx], 3):
            st.markdown(f"- {r}")
        drivers = (
            contrib.loc[idx].rename(index=LABELS).sort_values(key=abs, ascending=False).head(8)
            .rename("push").rename_axis("factor").reset_index()
        )
        drivers["direction"] = drivers["push"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
        chart = (
            alt.Chart(drivers)
            .mark_bar()
            .encode(
                x=alt.X("push:Q", title="Push on risk vs. average customer"),
                y=alt.Y("factor:N", sort="-x", title=None),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                              range=["#d1495b", "#2e86ab"]), title=None),
                tooltip=["factor", alt.Tooltip("push:Q", format=".2f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")

    with right:
        st.markdown("**What-if: try a retention offer**")
        contracts = ["Month-to-month", "One year", "Two year"]
        new_contract = st.selectbox("Contract", contracts, index=contracts.index(row["Contract"]))
        support = st.toggle("Add free tech support", value=row["TechSupport"] == "Yes",
                            disabled=row["InternetService"] == "No")
        security = st.toggle("Add free online security", value=row["OnlineSecurity"] == "Yes",
                             disabled=row["InternetService"] == "No")
        autopay = st.toggle("Switch to automatic card payment",
                            value=row["PaymentMethod"] == "Credit card (automatic)")

        scenario = df.loc[[idx]].copy()
        scenario["Contract"] = new_contract
        if row["InternetService"] != "No":
            scenario["TechSupport"] = "Yes" if support else "No"
            scenario["OnlineSecurity"] = "Yes" if security else "No"
        if autopay:
            scenario["PaymentMethod"] = "Credit card (automatic)"
        new_risk = predict_risk(bundle, scenario)[0]

        a, b = st.columns(2)
        a.metric("Risk now", f"{row['risk']:.0%}")
        b.metric("Risk with offer", f"{new_risk:.0%}", f"{(new_risk - row['risk']) * 100:+.0f} pts",
                 delta_color="inverse")
        st.altair_chart(charts.what_if(row["risk"], new_risk, threshold), width="stretch")
        st.caption(f"Expected monthly revenue kept: ${(row['risk'] - new_risk) * row['MonthlyCharges']:,.2f}. "
                   "This shows how the model responds, not a guaranteed customer reaction. "
                   "Price discounts aren't offered here: in this data, bill size mostly reflects which "
                   "services a customer has, so the model can't isolate the effect of price.")

st.divider()
st.markdown("Was this list useful?")
st.feedback("thumbs", key="feedback_predict")
