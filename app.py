"""Churn Radar: entry point. Run with `streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="Churn Radar", page_icon=":material/radar:", layout="wide")

nav = st.navigation(
    [
        st.Page("pages/predict.py", title="Score customers", icon=":material/target:", default=True),
        st.Page("pages/explore.py", title="Explore", icon=":material/bar_chart:"),
        st.Page("pages/about.py", title="About & limits", icon=":material/info:"),
    ]
)
nav.run()
