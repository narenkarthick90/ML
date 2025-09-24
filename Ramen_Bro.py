import streamlit as st
import pandas as pd
import pickle
import openpyxl

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# Load the ramen dataset
url = "https://raw.githubusercontent.com/narenkarthick90/Ramen_Bro/master/instant_data.csv"
df = pd.read_csv(url, encoding="latin1")

# Clean the data
df['Stars'] = pd.to_numeric(df['Stars'], errors='coerce')
df = df.dropna(subset=['Stars'])

# Select features for training
X = df[['Brand', 'Variety', 'Country']]
y = df['Stars']

# Fit encoder and model
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_encoded, y)

# Streamlit UI
st.title("🍜 Ramen Rating Predictor")

# Dropdowns
brand = st.selectbox("Select Brand", sorted(df['Brand'].unique()))
variety = st.selectbox("Select Variety", sorted(df['Variety'].unique()))
country = st.selectbox("Select Country", sorted(df['Country'].unique()))

# Predict button
if st.button("Predict Rating"):
    # Prepare input for model
    input_df = pd.DataFrame([[brand, variety, country]],
                            columns=['Brand', 'Variety', 'Country'])
    input_encoded = encoder.transform(input_df)
    prediction = model.predict(input_encoded)[0]

    st.success(f"Predicted Rating: ⭐ {prediction:.2f}")
