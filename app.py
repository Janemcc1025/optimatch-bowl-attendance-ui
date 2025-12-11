import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------
# LOAD MODEL + FEATURE COLUMNS
# -----------------------------------
model = joblib.load("attendance_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.title("Optimatch Bowl Attendance Predictor")
st.write("""
This tool uses CSMG's Gradient Boosted Regression model to estimate attendance 
for future bowl matchups based on fanbase size, travel distance, brand power, 
team strength, and bowl-specific factors.
""")

# -----------------------------------
# USER INPUTS
# -----------------------------------
st.header("Matchup Inputs")

team1 = st.text_input("Team 1 Name")
team2 = st.text_input("Team 2 Name")

team1_ap = st.number_input("Team 1 AP Ranking (0 if unranked)", 0, 50, 0)
team2_ap = st.number_input("Team 2 AP Ranking (0 if unranked)", 0, 50, 0)

team1_wins = st.number_input("Team 1 Wins", 0, 15, 0)
team2_wins = st.number_input("Team 2 Wins", 0, 15, 0)

team1_talent = st.number_input("Team 1 247 Talent Composite Score", 0.0, 1000.0, 600.0)
team2_talent = st.number_input("Team 2 247 Talent Composite Score", 0.0, 1000.0, 600.0)

team1_brand = st.number_input("Team 1 Brand Power", 0.0, 500000.0, 0.0)
team2_brand = st.number_input("Team 2 Brand Power", 0.0, 500000.0, 0.0)

venue_tier = st.selectbox("Venue Tier", [1, 2, 3])
venue_capacity = st.number_input("Venue Capacity", 0, 120000, 50000)

team1_miles = st.number_input("Team 1 Distance to Venue (miles)", 0.0, 3000.0, 0.0)
team2_miles = st.number_input("Team 2 Distance to Venue (miles)", 0.0, 3000.0, 0.0)

weekend_flag = 1 if st.checkbox("Weekend Game?") else 0

matchup_power = st.selectbox(
    "Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)",
    [2, 1, 0]
)

bowl_tier = st.selectbox("Bowl Tier", [1, 2, 3])

bowl_avg_att = st.number_input("Historical Bowl Avg Attendance", 0, 100000, 0)

# -----------------------------------
# BUILD FEATURE VECTOR FOR MODEL
# -----------------------------------
def build_feature_row():
    data = {
        "Team 1 AP Ranking": team1_ap,
        "Team 1 Wins": team1_wins,
        "Team 1 247 Talent Composite Score": team1_talent,
        "Team 1 Brand Power": team1_brand,

        "Team 2 AP Ranking": team2_ap,
        "Team 2 Wins": team2_wins,
        "Team 2 247 Talent Composite Score": team2_talent,
        "Team 2 Brand Power": team2_brand,

        "Venue Tier": venue_tier,

        "Team 1 Miles to Venue": team1_miles,
        "Team 2 Miles to Venue": team2_miles,

        "Avg Wins": (team1_wins + team2_wins) / 2,
        "Total Wins": team1_wins + team2_wins,
        "Avg AP Ranking": (team1_ap + team2_ap) / 2,
        "Best AP Ranking": min(team1_ap, team2_ap),
        "Worst AP Ranking": max(team1_ap, team2_ap),

        "SEC Present": 0,
        "Big 10 Present": 0,
        "Big 12 Present": 0,
        "ACC Present": 0,

        "Avg Distace Traveled": (team1_miles + team2_miles) / 2,
        "Distance Minimum": min(team1_miles, team2_miles),
        "Distance Imbalance": abs(team1_miles - team2_miles),

        "Weekend Indicator (Weekend=1, Else 0)": weekend_flag,
        "Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)": matchup_power,
        "Bowl Tier": bowl_tier,
        "Bowl Ownership": 0,
        "Local Team Indicator": 1 if min(team1_miles, team2_miles) < 75 else 0,
        "AP Strength Score": 0,

        "Combined Fanbase (Log transformed)": 0,
        "MinMax Scale Distcance": 0,
        "AP Strength Normalized": 0,

        "Bowl Avg Viewers": 0,
        "Bowl Avg Attendees": bowl_avg_att
    }

    # Return row in the correct column order
    row = [data.get(col, 0) for col in feature_cols]
    return pd.DataFrame([row], columns=feature_cols)

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("Predict Attendance"):
    X_new = build_feature_row()
    raw_pred = model.predict(X_new)[0]

    # Match your pipeline: blend + cap at capacity
    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att
    final_pred = min(blended_pred, venue_capacity)

    st.subheader("Predicted Attendance")
    st.metric("Projected Attendance", f"{final_pred:,.0f}")
