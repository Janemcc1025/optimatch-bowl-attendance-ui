import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# -----------------------------------
# LOAD MODEL + FEATURES
# -----------------------------------
model = joblib.load("attendance_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# -----------------------------------
# LOAD LOOKUP TABLES
# -----------------------------------
teams = pd.read_csv("2025 Bowl Games - UI Team Lookup.csv")
venues = pd.read_csv("2025 Bowl Games - UI Venue Lookup.csv")
bowl_tiers = pd.read_csv("2025 Bowl Games - Bowl Tiers.csv")

team_list = sorted(teams["Team"].unique())
bowl_list = sorted(venues["Bowl Game Name"].unique())

# -----------------------------------
# HELPER: DISTANCE CALCULATION
# -----------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# -----------------------------------
# STREAMLIT UI
# -----------------------------------
st.set_page_config(page_title="Optimatch Bowl Attendance Predictor", layout="wide")

st.title("🏈 Optimatch Bowl Attendance Predictor")
st.write("Predict bowl attendance using CSMG’s Gradient Boosted Model + full Optimatch feature engine.")

# -----------------------------------
# SELECT BOWL
# -----------------------------------
st.header("Select Bowl Game")
bowl_choice = st.selectbox("Bowl Game", bowl_list)

venue_row = venues[venues["Bowl Game Name"] == bowl_choice].iloc[0]
venue_name = venue_row["Venue"]
venue_capacity = venue_row["Venue Capacity"]
venue_lat = venue_row["Venue Lat"]
venue_lon = venue_row["Venue Lon"]
venue_tier = venue_row["Venue Tier"]
bowl_avg_att = venue_row["Bowl Avg Attendees"]

st.subheader(f"Venue: {venue_name}")
st.write(f"Capacity: {venue_capacity:,}")

# -----------------------------------
# SELECT TEAMS
# -----------------------------------
st.header("Select Teams")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", team_list)

with col2:
    team2 = st.selectbox("Team 2", team_list)

row1 = teams[teams["Team"] == team1].iloc[0]
row2 = teams[teams["Team"] == team2].iloc[0]

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------
team1_miles = haversine(row1["Lat"], row1["Lon"], venue_lat, venue_lon)
team2_miles = haversine(row2["Lat"], row2["Lon"], venue_lat, venue_lon)

avg_wins = (row1["Wins"] + row2["Wins"]) / 2
total_wins = row1["Wins"] + row2["Wins"]

ap_values = [x for x in [row1["AP Rank"], row2["AP Rank"]] if x > 0]
avg_ap = np.mean(ap_values) if ap_values else 0
best_ap = min(ap_values) if ap_values else 0
worst_ap = max(ap_values) if ap_values else 0

distance_min = min(team1_miles, team2_miles)
distance_imbalance = abs(team1_miles - team2_miles)
avg_distance = (team1_miles + team2_miles) / 2

local_flag = 1 if distance_min < 75 else 0

# Matchup Power Score
conf1 = row1["Conference"]
conf2 = row2["Conference"]
power_confs = ["SEC", "Big 10", "Big 12", "ACC"]

if conf1 in power_confs and conf2 in power_confs:
    matchup_power = 2
elif conf1 in power_confs or conf2 in power_confs:
    matchup_power = 1
else:
    matchup_power = 0

# Bowl Tier + Ownership
bowl_info = bowl_tiers[bowl_tiers["Bowl Game Name"] == bowl_choice].iloc[0]
bowl_tier = bowl_info["Bowl Tier"]
bowl_owner = bowl_info["Bowl Ownership"]

# Weekend Flag
import datetime
game_date = venue_row["Date"]
weekday = pd.to_datetime(game_date).weekday()
weekend_flag = 1 if weekday >= 5 else 0

# Combined Fanbase Log
combined_fanbase = row1["Fanbase Size"] + row2["Fanbase Size"]
combined_fanbase_log = np.log1p(combined_fanbase)

# -----------------------------------
# BUILD MODEL INPUT ROW
# -----------------------------------
feature_row = pd.DataFrame([[

    row1["AP Rank"], row1["Wins"], row1["Talent Score"], row1["Brand Power"],
    row2["AP Rank"], row2["Wins"], row2["Talent Score"], row2["Brand Power"],

    venue_tier,
    team1_miles, team2_miles,

    avg_wins, total_wins, avg_ap,
    best_ap, worst_ap,

    int(conf1 == "SEC"), int(conf1 == "Big 10"), int(conf1 == "Big 12"), int(conf1 == "ACC"),
    
    avg_distance, distance_min, distance_imbalance,
    weekend_flag, matchup_power,

    bowl_tier, bowl_owner, local_flag,

    row1["AP Strength"] + row2["AP Strength"],

    combined_fanbase_log,
    (avg_distance / venues["Venue Capacity"].max()),
    avg_ap / 25,

    venue_row["Bowl Avg Viewers"],
    bowl_avg_att

]], columns=feature_cols)

# -----------------------------------
# PREDICTION
# -----------------------------------
st.header("Prediction")

raw_pred = model.predict(feature_row)[0]
blended = 0.7 * raw_pred + 0.3 * bowl_avg_att
final_pred = min(blended, venue_capacity)
pct_filled = final_pred / venue_capacity

st.metric("Predicted Attendance", f"{final_pred:,.0f}")
st.metric("% Filled", f"{pct_filled:.1%}")

# -----------------------------------
# KEY DRIVER SUMMARY
# -----------------------------------
st.subheader("Key Driver Summary")

drivers = []

if distance_imbalance > 400:
    drivers.append("Large travel imbalance between teams influences demand.")
elif distance_min < 150:
    drivers.append("At least one nearby team strongly boosts attendance.")

if combined_fanbase_log > teams["Fanbase Size"].median():
    drivers.append("Large combined fanbase contributes positively.")

if matchup_power == 2:
    drivers.append("Power vs. Power matchup increases general interest.")
elif matchup_power == 1:
    drivers.append("One Power conference team lifts attendance potential.")

if row1["Brand Power"] + row2["Brand Power"] > teams["Brand Power"].quantile(0.75) * 2:
    drivers.append("High brand visibility increases draw potential.")

for d in drivers:
    st.write("• " + d)

if len(drivers) == 0:
    st.write("• Balanced matchup with typical bowl attendance drivers.")
