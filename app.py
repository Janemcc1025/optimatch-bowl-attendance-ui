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
# FEATURE ENGINEERING (SAFE VERSION)
# -----------------------------------

def safe_num(x):
    """Convert to float and replace NAN with 0."""
    try:
        if pd.isna(x):
            return 0
        return float(x)
    except:
        return 0

# Safe numeric values
t1_ap = safe_num(row1["2025 AP Ranking"])
t2_ap = safe_num(row2["2025 AP Ranking"])

t1_wins = safe_num(row1["2025 Wins"])
t2_wins = safe_num(row2["2025 Wins"])

t1_talent = safe_num(row1["2025 Talent"])
t2_talent = safe_num(row2["2025 Talent"])

t1_brand = safe_num(row1["Team Brand Power 2025"])
t2_brand = safe_num(row2["Team Brand Power 2025"])

t1_ap_strength = safe_num(row1["2025 AP Strength Score"])
t2_ap_strength = safe_num(row2["2025 AP Strength Score"])

# Distances (safe)
t1_lat = safe_num(row1["Latitude"])
t1_lon = safe_num(row1["Longitude"])
t2_lat = safe_num(row2["Latitude"])
t2_lon = safe_num(row2["Longitude"])

team1_miles = safe_num(haversine(t1_lat, t1_lon, venue_lat, venue_lon))
team2_miles = safe_num(haversine(t2_lat, t2_lon, venue_lat, venue_lon))

avg_wins = safe_num((t1_wins + t2_wins) / 2)
total_wins = safe_num(t1_wins + t2_wins)

# AP Rankings
ap_vals = [x for x in [t1_ap, t2_ap] if x > 0]
avg_ap = safe_num(np.mean(ap_vals) if len(ap_vals) else 0)
best_ap = safe_num(min(ap_vals) if len(ap_vals) else 0)
worst_ap = safe_num(max(ap_vals) if len(ap_vals) else 0)

# Distances
avg_distance = safe_num((team1_miles + team2_miles) / 2)
distance_min = safe_num(min(team1_miles, team2_miles))
distance_imbalance = safe_num(abs(team1_miles - team2_miles))

local_flag = 1 if distance_min < 75 else 0

# Conference flags
conf1 = str(row1["Football FBS Conference"])
conf2 = str(row2["Football FBS Conference"])

SEC_present = int("SEC" in [conf1, conf2])
B10_present = int("Big 10" in [conf1, conf2])
B12_present = int("Big 12" in [conf1, conf2])
ACC_present = int("ACC" in [conf1, conf2])

level1 = str(row1["Conference Level"])
level2 = str(row2["Conference Level"])

# Matchup power score
if level1 == "P4" and level2 == "P4":
    matchup_power = 2
elif level1 == "P4" or level2 == "P4":
    matchup_power = 1
else:
    matchup_power = 0

# Bowl tier fields (safe)
bowl_tier = safe_num(bowl_row["Tier"])
bowl_owner = safe_num(bowl_row["Ownership"])
bowl_avg_viewers = safe_num(bowl_row["Avg Viewers"])
bowl_avg_att = safe_num(bowl_row["Avg Attendance"])

combined_fanbase = safe_num(row1["Team Fanbase Size"]) + safe_num(row2["Team Fanbase Size"])
combined_fanbase_log = safe_num(np.log1p(combined_fanbase))

ap_strength_score = safe_num(t1_ap_strength + t2_ap_strength)

minmax_distance = safe_num(avg_distance / 3000)
ap_strength_norm = safe_num(ap_strength_score / 50)

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

# =========================================================
#  CUSTOM OPTIMATCH POST-PROCESSING RULES
# =========================================================

raw_pred = model.predict(feature_row)[0]
blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att

# 1. Special rule for Hawaii + Bahamas Bowl
special_bowls_raw = ["Hawaii Bowl", "Bahamas Bowl"]
if bowl_choice in special_bowls_raw:
    final_pred = raw_pred
else:
    final_pred = blended_pred

# 2. Apply bowl-specific attendance boosts
boosts = {
    "Gator Bowl": 1.05,
    "Pop-Tarts Bowl": 1.05,
    "Texas Bowl": 1.05,
    "Music City Bowl": 1.05,
    "Alamo Bowl": 1.10,
    "Duke’s Mayo Bowl": 1.03
}

if bowl_choice in boosts:
    final_pred *= boosts[bowl_choice]

# 3. Capacity constraint
final_pred = min(final_pred, venue_capacity)

# 4. Floor at zero
final_pred = max(final_pred, 0)

pct_filled = final_pred / venue_capacity


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

