import streamlit as st
import pandas as pd
from apputil import *

st.write('''
# Week 9: Group Estimate Model

Enter a category combination to get a predicted estimate.
''')

# Sample data: tips dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

X = df[["day", "time"]]
y = df["tip"]

gm = GroupEstimate(estimate='mean')
gm.fit(X, y)

day = st.selectbox("Day:", df["day"].unique())
time = st.selectbox("Time:", df["time"].unique())

if st.button("Predict"):
    result = gm.predict([[day, time]])
    if pd.isna(result[0]):
        st.write("No data available for this combination.")
    else:
        st.write(f"Estimated tip for **{day} {time}**: **${result[0]:.2f}**")
