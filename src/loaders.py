"""Streamlit-cached loaders, shared by every page."""
import pandas as pd
import streamlit as st

from src.model import SAMPLE_PATH, load_model


@st.cache_resource  # load once, not on every rerun
def get_model() -> dict:
    return load_model()


@st.cache_data
def get_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)
