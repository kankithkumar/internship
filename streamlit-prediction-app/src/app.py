
import streamlit as st
import pandas as pd
import joblib
import json
import os
from datetime import datetime
import ephem

# ---------- Load files ----------
model = joblib.load("model/xgb_model.pkl")
encoder = joblib.load("model/encoder.pkl")
with open("data/feature_medians.json") as f:
    feature_medians = json.load(f)

# ---------- Helper Functions ----------
def moon_phase(date):
    moon = ephem.Moon()
    moon.compute(date)
    return moon.phase

def preprocess_input(user_input):
    user_input['Nak_Tithi'] = f"{user_input['Nakshatra']}_{user_input['Tithi']}"
    user_input['Yoga_Karna'] = f"{user_input['Yoga']}_{user_input['Karna']}"
    user_input['Date'] = pd.to_datetime(user_input['Date'])
    user_input['DayOfWeek'] = user_input['Date'].dayofweek
    user_input['LunarPhase'] = moon_phase(user_input['Date'])
    user_input['NearFullMoon'] = int(user_input['LunarPhase'] > 95)
    user_input['NearNewMoon'] = int(user_input['LunarPhase'] < 5)

    for col in feature_medians.keys():
        if col not in user_input:
            user_input[col] = feature_medians[col]
    return user_input

# ---------- Streamlit App ----------
st.set_page_config(page_title="Astro Prediction for Nifty50", page_icon="🔮", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .css-1d391kg { padding: 2rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔮 Astro-Based Outcome Predictor")
st.subheader("Predict Market Outcome Based on Panchang Inputs")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        nakshatra = st.selectbox("🌙 Nakshatra", options=[
    'Anuradha', 'Ardra', 'Ashlesha', 'Ashwini', 'Bharani', 'Chitra', 'Dhanishta', 'Hasta',
    'Jyeshtha', 'Krittika', 'Magha', 'Moola', 'Mrigashira', 'Punarvasu', 'Pushya', 'Purva Ashadha',
    'Purva Bhadrapada', 'Purva Phalguni', 'Revati', 'Rohini', 'Shatabhisha', 'Shravana',
    'Swati', 'Uttara Ashadha', 'Uttara Bhadrapada', 'Uttara Phalguni', 'Vishakha'
])
        karna = st.selectbox("🪔 Karna", options=[
    'Balava', 'Bava', 'Chatushpada', 'Garija', 'Kaulava', 'Kimstughna', 'Naga', 
    'Sakuna', 'Taitula', 'Vanija', 'Vishti'
])
    with col2:
        tithi = st.selectbox("📆 Tithi", options=[
    'Amavasya', 'Ashtami', 'Chaturdasi', 'Chaturthi', 'Dasami', 'Dwadasi', 'Dwitiya',
    'Ekadashi', 'Navami', 'Panchami', 'Pratipat', 'Pournimasya', 'Sapthami', 'Shashthi',
    'Tritiya','Trayodasi'
])

        yoga = st.selectbox("🧘 Yoga", options=[
    'Atiganda', 'Ayushman', 'Brahma', 'Dhriti', 'Dhruva', 'Ganda', 'Harshana', 'Indra',
    'Parigha', 'Priti','Sadhya','Siddha','Siddhi', 'Siva', 'Sobhana', 'Soola','Soubhagya', 'Subha',
    'Sukarman',  'Sukla', 'Vaidhriti', 'Vajra', 'Variyan','Vishkambha','Vriddhi','Vyaghata','Vyatipata'
])

    date = st.date_input("📅 Select a Date", value=datetime.today())

# ---------- Prediction ----------
if st.button("🔍 Predict Outcome"):
    user_input = {
        'Nakshatra': nakshatra,
        'Tithi': tithi,
        'Karna': karna,
        'Yoga': yoga,
        'Date': date,
   
    }

    processed_input = preprocess_input(user_input)
    
    user_df = pd.DataFrame([processed_input])

    # Encode categorical + interaction columns
    categorical_cols = ['Nakshatra', 'Tithi', 'Karna', 'Yoga']
    interaction_cols = ['Nak_Tithi', 'Yoga_Karna']
    feature_cols = categorical_cols + interaction_cols + [
        'DayOfWeek', 'LunarPhase', 'NearFullMoon', 'NearNewMoon'
    ] + list(feature_medians.keys())

    user_df[categorical_cols + interaction_cols] = encoder.transform(user_df[categorical_cols + interaction_cols])
    user_df = user_df[feature_cols]
    prediction = model.predict(user_df.values)[0]
    label = '📈 Positive' if (prediction) == 1 else '📉 Negative'
    st.success(f"✅ **Predicted Outcome**: {label}")

