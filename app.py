# !pip install -q streamlit
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import plotly as px
from datetime import datetime
# !npm install localtunnel
# !streamlit run app.py &>/content/logs.txt &
# !npx localtunnel --port 8501

import os

# Set the timezone environment variable
os.environ['TZ'] = 'Asia/Rangoon'

# This updates the C-level timezone settings that Python uses
import time
time.tzset()


# --- CONFIG & STYLING ---
st.set_page_config(page_title="Elevator AEP Predictor", layout="wide")
st.title("Elevator Position Predictor")
st.markdown("Using the **Asymptotic Equipartition Property (AEP)** to predict floor locations.")

# --- DATA ENGINE ---
@st.cache_data
def load_and_process():
    df = pd.read_csv('elevator_24h_30days.csv')
    time_cols = ['5min', '10min', '15min', '20min', '25min', '30min', 
                 '35min', '40min', '45min', '50min', '55min', '60min']
    
    model_rows = []
    for hour in range(24):
        hour_data = df[df['Hour'] == hour]
        for col in time_cols:
            counts = hour_data[col].value_counts(normalize=True)
            typical_floor = counts.idxmax()
            confidence = counts.max()
            
            # Simplified Entropy calculation for the UI
            probs = counts.values
            entropy = -np.sum(probs * np.log2(probs))
            
            model_rows.append({
                'Hour': hour,
                'Minute': int(col.replace('min', '')),
                'Typical_Floor': typical_floor,
                'Confidence': confidence,
                'Entropy': entropy
            })
    return pd.DataFrame(model_rows), df

model_df, raw_df = load_and_process()

# --- SIDEBAR / INPUTS ---
st.sidebar.header("Current Time Settings")
sim_mode = st.sidebar.checkbox("Simulation Mode (Set Custom Time)")

if sim_mode:
    target_hour = st.sidebar.slider("Hour", 0, 23, 9)
    target_minute = st.sidebar.slider("Minute", 0, 59, 15)
else:
    now = datetime.now()
    target_hour, target_minute = now.hour, now.minute

# --- PREDICTION LOGIC ---
rounded_min = 5 * round(target_minute / 5)
if rounded_min == 0: rounded_min = 5
if rounded_min > 60: rounded_min = 60

result = model_df[(model_df['Hour'] == target_hour) & (model_df['Minute'] == rounded_min)].iloc[0]

# --- UI LAYOUT ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Floor", f"Floor {int(result['Typical_Floor'])}")
with col2:
    st.metric("Typicality (Confidence)", f"{result['Confidence']*100:.1f}%")
with col3:
    st.metric("Hour Entropy (H)", f"{result['Entropy']:.2f} bits")

# --- VISUALIZATION ---
st.divider()
st.subheader("Daily Typical Path")
# Step chart of the typical path
fig = px.line(model_df, x='Hour', y='Typical_Floor', hover_data=['Minute', 'Confidence'],
              title="Most Typical Floor Throughout the Day", markers=True, line_shape="hv")
fig.update_yaxes(tick0=0, dtick=1)
st.plotly_chart(fig, use_container_width=True)

# Advice Box
st.info(f"**Live Advice:** At {target_hour}:{target_minute:02d}, the elevator is " + 
        ("likely idling. You should probably walk." if result['Entropy'] < 0.5 else "moving frequently. It's worth waiting!"))

# --- DATA SOURCE TABLE ---
# with st.expander("View Underlying Typicality Model"):
#     st.dataframe(model_df)

