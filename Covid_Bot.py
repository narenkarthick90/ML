import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("covid_19_data.csv")

st.title("🌍 COVID-19 Real-Time Dashboard")
country = st.selectbox("Select a Country", df["Country/Region"].unique())
filtered = df[df["Country/Region"] == country]

st.metric("Total Cases", filtered["Confirmed"].sum())
st.metric("Deaths", filtered["Deaths"].sum())
st.metric("Recovered", filtered["Recovered"].sum())

fig = px.line(filtered, x="ObservationDate", y=["Confirmed", "Deaths", "Recovered"], title=f"COVID Trend in {country}")
st.plotly_chart(fig)