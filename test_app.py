import streamlit as st 
import pandas as pd 

st.title("Test application")



if st.button("Test Button") :
    
    st.balloons()
    
    #  .
    #  ./
    #  ./../../

df = pd.read_csv("./data/sample.csv")

st.dataframe(df.head())

st.bar_chart(df["gender"])
