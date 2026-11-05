"""Compare segments and see what drives churn across the whole list."""
import altair as alt
import pandas as pd
import streamlit as st

from src.features import LABELS
from src.loaders import get_model, get_sample
from src.model import contributions, predict_risk

st.title("Explore")
st.markdown("See which customer segments carry the most risk, and what drives it overall.")

bundle = get_model()
df = st.session_state.get("df", get_sample()).copy()
source = st.session_state.get("source", "sample data")
st.caption(f"Using *{source}* ({len(df):,} customers). Upload your own on the **Score customers** page.")

try:
    df["risk"] = predict_risk(bundle, df)
    contrib = contributions(bundle, df)
except Exception:
    st.error("This data couldn't be scored. Go to **Score customers** to see what needs fixing.")
    st.stop()

df["tenure_band"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 1000],
                           labels=["0-6 mo", "7-12 mo", "1-2 yr", "2-4 yr", "4+ yr"])

segment = st.selectbox("Compare risk by", ["Contract", "tenure_band", "InternetService", "PaymentMethod",
                                           "TechSupport", "SeniorCitizen"],
                       format_func=lambda c: "Tenure" if c == "tenure_band" else LABELS.get(c, c))
seg = (df.groupby(segment, observed=True)
         .agg(customers=("risk", "size"), avg_risk=("risk", "mean"),
              revenue_at_risk=("MonthlyCharges", lambda s: (s * df.loc[s.index, "risk"]).sum()))
         .reset_index())

c1, c2 = st.columns(2)
c1.altair_chart(
    alt.Chart(seg).mark_bar(color="#d1495b").encode(
        x=alt.X(f"{segment}:N", title=None, sort="-y"),
        y=alt.Y("avg_risk:Q", title="Average churn risk", axis=alt.Axis(format="%")),
        tooltip=[segment, "customers", alt.Tooltip("avg_risk:Q", format=".0%")],
    ).properties(title="Average risk", height=300),
    width="stretch",
)
c2.altair_chart(
    alt.Chart(seg).mark_bar(color="#2e86ab").encode(
        x=alt.X(f"{segment}:N", title=None, sort="-y"),
        y=alt.Y("revenue_at_risk:Q", title="Monthly revenue at risk ($)"),
        tooltip=[segment, "customers", alt.Tooltip("revenue_at_risk:Q", format="$,.0f")],
    ).properties(title="Revenue at risk", height=300),
    width="stretch",
)

st.subheader("What drives risk across this list")
imp = (contrib.clip(lower=0).mean().rename(index=LABELS).sort_values(ascending=False)
       .head(10).rename("avg_push").rename_axis("factor").reset_index())
st.altair_chart(
    alt.Chart(imp).mark_bar(color="#d1495b").encode(
        x=alt.X("avg_push:Q", title="Average upward push on risk"),
        y=alt.Y("factor:N", sort="-x", title=None),
    ).properties(height=320),
    width="stretch",
)

st.subheader("Risk distribution")
st.altair_chart(
    alt.Chart(df).mark_bar(color="#888").encode(
        x=alt.X("risk:Q", bin=alt.Bin(step=0.05), title="Churn risk", axis=alt.Axis(format="%")),
        y=alt.Y("count():Q", title="Customers"),
    ).properties(height=220),
    width="stretch",
)
