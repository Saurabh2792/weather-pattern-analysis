import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load and prep data
df = pd.read_csv('data/dataset.csv')
X = df[['Temperature_C', 'Humidity_pct', 'Wind_Speed_kmh']]
y = df['Rain']

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Dashboard Interface
st.title("Weather Pattern & Rain Predictor")
st.write("Input current weather metrics to predict the likelihood of rain.")

# Inputs
temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=15.0)

# Prediction Logic
if st.button("Predict Weather"):
    prediction = model.predict([[temp, humidity, wind]])
    if prediction[0] == 1:
        st.error("🌧️ Prediction: Rain is expected.")
    else:
        st.success("☀️ Prediction: Clear skies expected.")

st.subheader("Historical Weather Data")
st.dataframe(df)