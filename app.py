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

# Clean whitespace in headers
teams.columns = teams.columns.str.strip()
venues.columns = venues.columns.str.strip()
bowl_tiers.columns = bowl_tiers.columns.str.strip()

team_col = "Team Name"
team_list = sorted(teams[team_col].unique())
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
st.write("Predict bowl attendance using CSMG’s Gradient Boosted Model + Optimatch feature engine.")

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

row1 = teams[teams[team_col] == team1].iloc[0]
row2 = teams[teams[team_col] == team2].iloc[0]

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

# Distance calculations
team1_miles = haversine(row1["Lat"], row1["Lon"], venue_lat, venue_lon)
team2_miles = haversine(row2["Lat"], row2["Lon"], venue_lat, venue_lon)

# Wins & AP rankings
avg_wins = (row1["Wins"] + row2["Wins"]) / 2
total_wins = row1["Wins"] + row2["Wins"]

ap_values = [x for x in [row1["AP Rank"], row2["AP Rank"]] if x > 0]
avg_ap = np.mean(ap_values) if ap_values else 0
best_ap = min(ap_values) if ap_values else 0
worst_ap = max(ap_values) if ap_values else 0

# Distance features
distance_min = min(team1_miles, team2_miles)
distance_imbalance = abs(team1_miles - team2_miles)
avg_distance = (team1_miles + team2_miles) / 2

# Local team?
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

# Bowl Tier & Ownership
bowl_info = bowl_tiers[bowl_tiers["Bowl Game Name"] == bowl_choice].iloc[0]
bowl_tier = bowl_info["Bowl Tier"]
bowl_owner = bowl_info["Bowl Ownership"]

# Weekend flag (0 = weekday, 1 = weekend)
import datetime
try:
    weekday = pd.to_datetime(venue_row["Date"]).weekday()
    weekend_flag = 1 if weekday >= 5 else 0
except:
    weekend_flag = 0

# Combined Fanbase Log
combined_fanbase = row1["Fanbase Size"] + row2["Fanbase Size"]
combined_fanbase_log = np.log1p(combined_fanbase)

# AP Strength Score (sum)
ap_strength_score = row1["AP Strength"] + row2["AP Strength"]

# -----------------------------------
# BUILD FINAL FEATURE VECTOR
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

    ap_strength_score,

    combined_fanbase_log,
    (avg_distance / 3000),        # simple min/max scale
    avg_ap / 25 if avg_ap > 0 else 0,

    venue_row["Bowl Avg Viewers"],
    bowl_avg_att

]], columns=feature_cols)

# -----------------------------------
# PREDICT ATTENDANCE
# -----------------------------------
st.header("Prediction")

raw_pred = model.predict(feature_row)[0]
blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att
final_pred = min(blended_pred, venue_capacity)
pct_filled = final_pred / venue_capacity

st.metric("Predicted Attendance", f"{final_pred:,.0f}")
st.metric("Projected % Filled", f"{pct_filled:.1%}")

# -----------------------------------
# KEY DRIVER SUMMARY
# -----------------------------------
st.subheader("Key Driver Summary")

driver_messages = []

if distance_min < 150:
    driver_messages.append("At least one team is within strong travel range (<150 miles).")
if distance_imbalance > 400:
    driver_messages.append("Large travel imbalance influences turnout.")
if combined_fanbase_log > np.log1p(teams["Fanbase Size"].median()):
    driver_messages.append("Large combined fanbase adds demand potential.")
if matchup_power == 2:
    driver_messages.append("Power vs. Power matchup increases national interest.")
if local_flag == 1:
    driver_messages.append("A local team significantly lifts attendance.")

if len(driver_messages) == 0:
    driver_messages.append("This matchup aligns with typical bowl attendance patterns.")

for msg in driver_messages:
    st.write("• " + msg)
