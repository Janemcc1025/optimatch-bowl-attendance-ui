import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# -----------------------------------
# LOAD MODEL + FEATURE COLUMNS
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

TEAM_COL = "Team Name"
VENUE_COL = "Football Stadium"
BOWL_NAME_COL = "Bowl Name"

team_list = sorted(teams[TEAM_COL].unique())
venue_list = sorted(venues[VENUE_COL].unique())
bowl_list = sorted(bowl_tiers[BOWL_NAME_COL].unique())

# -----------------------------------
# HELPER: DISTANCE CALCULATION
# -----------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# -----------------------------------
# STREAMLIT UI
# -----------------------------------
st.set_page_config(page_title="Optimatch Bowl Attendance Predictor", layout="wide")

st.title("🏈 Optimatch Bowl Attendance Predictor")
st.write("Predict bowl attendance using CSMG’s Gradient Boosted model and Optimatch feature engine.")

# -----------------------------------
# BOWL + VENUE SELECTION
# -----------------------------------
st.header("Bowl & Venue Selection")

col_bowl, col_venue = st.columns(2)

with col_bowl:
    bowl_choice = st.selectbox("Bowl Name", bowl_list)

with col_venue:
    venue_choice = st.selectbox("Football Stadium", venue_list)

bowl_row = bowl_tiers[bowl_tiers[BOWL_NAME_COL] == bowl_choice].iloc[0]
venue_row = venues[venues[VENUE_COL] == venue_choice].iloc[0]

venue_capacity = venue_row["Football Capacity"]
venue_lat = venue_row["Lat"]
venue_lon = venue_row["Lon"]
venue_city = venue_row["City"]
venue_state = venue_row["State"]

bowl_tier = bowl_row["Tier"]
bowl_owner = bowl_row["Ownership"]
bowl_avg_viewers = bowl_row["Avg Viewers"]
bowl_avg_att = bowl_row["Avg Attendance"]

# For this UI, use bowl tier also as venue tier (you can change later if needed)
venue_tier = bowl_tier

st.subheader(f"{bowl_choice} at {venue_choice}")
st.write(f"Location: {venue_city}, {venue_state}")
st.write(f"Capacity: {venue_capacity:,}")
st.write(f"Historical Avg Attendance: {bowl_avg_att:,.0f}")

# Weekend flag as user input (no date in this lookup)
weekend_flag = 1 if st.checkbox("Is this game on a weekend?") else 0

# -----------------------------------
# TEAM SELECTION
# -----------------------------------
st.header("Team Selection")

col_t1, col_t2 = st.columns(2)

with col_t1:
    team1 = st.selectbox("Team 1", team_list)

with col_t2:
    team2 = st.selectbox("Team 2", team_list)

row1 = teams[teams[TEAM_COL] == team1].iloc[0]
row2 = teams[teams[TEAM_COL] == team2].iloc[0]

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

# 1. Current-season Wins, AP, Talent, Brand (2025)
t1_ap = row1["2025 AP Ranking"]
t2_ap = row2["2025 AP Ranking"]

t1_wins = row1["2025 Wins"]
t2_wins = row2["2025 Wins"]

t1_talent = row1["2025 Talent"]
t2_talent = row2["2025 Talent"]

t1_brand = row1["Team Brand Power 2025"]
t2_brand = row2["Team Brand Power 2025"]

t1_ap_strength = row1["2025 AP Strength Score"]
t2_ap_strength = row2["2025 AP Strength Score"]

# 2. Distances
t1_lat = row1["Latitude"]
t1_lon = row1["Longitude"]
t2_lat = row2["Latitude"]
t2_lon = row2["Longitude"]

team1_miles = haversine(t1_lat, t1_lon, venue_lat, venue_lon)
team2_miles = haversine(t2_lat, t2_lon, venue_lat, venue_lon)

avg_wins = (t1_wins + t2_wins) / 2
total_wins = t1_wins + t2_wins

ap_vals = [x for x in [t1_ap, t2_ap] if x > 0]
avg_ap = np.mean(ap_vals) if ap_vals else 0
best_ap = min(ap_vals) if ap_vals else 0
worst_ap = max(ap_vals) if ap_vals else 0

avg_distance = (team1_miles + team2_miles) / 2
distance_min = min(team1_miles, team2_miles)
distance_imbalance = abs(team1_miles - team2_miles)

local_flag = 1 if distance_min < 75 else 0

# 3. Conference flags & Matchup Power Score (P4 vs G5)
conf1 = row1["Football FBS Conference"]
conf2 = row2["Football FBS Conference"]
level1 = row1["Conference Level"]
level2 = row2["Conference Level"]

SEC_present = int((conf1 == "SEC") or (conf2 == "SEC"))
B10_present = int((conf1 == "Big 10") or (conf2 == "Big 10"))
B12_present = int((conf1 == "Big 12") or (conf2 == "Big 12"))
ACC_present = int((conf1 == "ACC") or (conf2 == "ACC"))

# Matchup Power Score based on Conference Level (assume "P4" vs "G5")
if level1 == "P4" and level2 == "P4":
    matchup_power = 2
elif (level1 == "P4" and level2 != "P4") or (level2 == "P4" and level1 != "P4"):
    matchup_power = 1
else:
    matchup_power = 0

# 4. AP Strength / Fanbase / Scaling
ap_strength_score = t1_ap_strength + t2_ap_strength

combined_fanbase = row1["Team Fanbase Size"] + row2["Team Fanbase Size"]
combined_fanbase_log = np.log1p(combined_fanbase)

# Simple distance scaling (0–1-ish)
minmax_distance = avg_distance / 3000.0

# Simple AP strength normalization (you can refine later)
ap_strength_norm = ap_strength_score / 50.0

# -----------------------------------
# BUILD FEATURE ROW IN CORRECT ORDER
# -----------------------------------
row_data = {
    "Team 1 AP Ranking": t1_ap,
    "Team 1 Wins": t1_wins,
    "Team 1 247 Talent Composite Score": t1_talent,
    "Team 1 Brand Power": t1_brand,

    "Team 2 AP Ranking": t2_ap,
    "Team 2 Wins": t2_wins,
    "Team 2 247 Talent Composite Score": t2_talent,
    "Team 2 Brand Power": t2_brand,

    "Venue Tier": bowl_tier,   # using bowl tier as proxy

    "Team 1 Miles to Venue": team1_miles,
    "Team 2 Miles to Venue": team2_miles,

    "Avg Wins": avg_wins,
    "Total Wins": total_wins,
    "Avg AP Ranking": avg_ap,
    "Best AP Ranking": best_ap,
    "Worst AP Ranking": worst_ap,

    "SEC Present": SEC_present,
    "Big 10 Present": B10_present,
    "Big 12 Present": B12_present,
    "ACC Present": ACC_present,

    "Avg Distace Traveled": avg_distance,
    "Distance Minimum": distance_min,
    "Distance Imbalance": distance_imbalance,

    "Weekend Indicator (Weekend=1, Else 0)": weekend_flag,
    "Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)": matchup_power,

    "Bowl Tier": bowl_tier,
    "Bowl Ownership": bowl_owner,
    "Local Team Indicator": local_flag,

    "AP Strength Score": ap_strength_score,

    "Combined Fanbase (Log transformed)": combined_fanbase_log,
    "MinMax Scale Distcance": minmax_distance,
    "AP Strength Normalized": ap_strength_norm,

    "Bowl Avg Viewers": bowl_avg_viewers,
    "Bowl Avg Attendees": bowl_avg_att,
}

# Ensure the DataFrame matches feature_cols order
feature_row = pd.DataFrame([[row_data.get(col, 0) for col in feature_cols]], columns=feature_cols)

# -----------------------------------
# PREDICT
# -----------------------------------
st.header("Prediction")

if st.button("Run Prediction"):
    raw_pred = model.predict(feature_row)[0]
    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att
    final_pred = min(blended_pred, venue_capacity)
    final_pred = max(final_pred, 0)
    pct_filled = final_pred / venue_capacity if venue_capacity > 0 else 0

    st.metric("Predicted Attendance", f"{final_pred:,.0f}")
    st.metric("Projected % Filled", f"{pct_filled:.1%}")

    # -----------------------------------
    # KEY DRIVER SUMMARY
    # -----------------------------------
    st.subheader("Key Driver Summary")

    drivers = []

    if distance_min < 150:
        drivers.append("At least one team is within strong drive range (<150 miles).")

    if distance_imbalance > 400:
        drivers.append("There is a significant travel imbalance between the fanbases.")

    if combined_fanbase_log > np.log1p(teams["Team Fanbase Size"].median()):
        drivers.append("Combined fanbase size is well above typical FBS levels.")

    if matchup_power == 2:
        drivers.append("Power vs. Power matchup elevates national interest.")
    elif matchup_power == 1:
        drivers.append("Presence of at least one Power conference team lifts demand.")

    if local_flag == 1:
        drivers.append("A local or near-local team strongly supports attendance.")

    if not drivers:
        drivers.append("This matchup aligns with typical bowl attendance patterns in our model.")

    for d in drivers:
        st.write("• " + d)

