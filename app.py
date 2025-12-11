import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# CONSTANTS / CONFIG
# =====================================================

PRED_YEAR = 2025  # we are predicting 2025 bowls with 2025 team stats

TEAM_FILE = "2025 Bowl Games - UI Team Lookup.csv"
VENUE_FILE = "2025 Bowl Games - UI Venue Lookup.csv"
BOWL_TIERS_FILE = "2025 Bowl Games - Bowl Tiers.csv"
CALIB_FILE = "2025 Bowl Games - 2025 Bowl Games (8).csv"  # 2025 master for scalers

MODEL_FILE = "attendance_model.pkl"
FEATURE_COLS_FILE = "feature_columns.pkl"

TEAM_COL = "Team Name"
VENUE_COL = "Football Stadium"
BOWL_NAME_COL = "Bowl Name"

# =====================================================
# HELPERS
# =====================================================

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    R = 3958.8
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def safe_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

# =====================================================
# LOAD MODEL + FEATURE LIST
# =====================================================

model = joblib.load(MODEL_FILE)
feature_cols = joblib.load(FEATURE_COLS_FILE)

# =====================================================
# LOAD LOOKUP TABLES
# =====================================================

teams = pd.read_csv(TEAM_FILE)
venues = pd.read_csv(VENUE_FILE)
bowl_tiers = pd.read_csv(BOWL_TIERS_FILE)
calib = pd.read_csv(CALIB_FILE)

for df in (teams, venues, bowl_tiers, calib):
    df.columns = df.columns.str.strip()

# =====================================================
# DERIVE SCALERS / THRESHOLDS FROM 2025 MASTER
# =====================================================

# Distance MinMax scaling: use Avg Distace Traveled & MinMax Scale Distcance
dist_raw = pd.to_numeric(calib["Avg Distace Traveled"], errors="coerce")
dist_scaled = pd.to_numeric(calib["MinMax Scale Distcance"], errors="coerce")

# approximate min and max from rows where scaled is min/max
smin = dist_scaled.min()
smax = dist_scaled.max()
dist_min_train = dist_raw[dist_scaled <= smin + 1e-6].mean()
dist_max_train = dist_raw[dist_scaled >= smax - 1e-6].mean()

if np.isnan(dist_min_train):
    dist_min_train = dist_raw.min()
if np.isnan(dist_max_train):
    dist_max_train = dist_raw.max()

# AP Strength normalization
ap_raw = pd.to_numeric(calib["AP Strength Score"], errors="coerce")
ap_scaled = pd.to_numeric(calib["AP Strength Normalized"], errors="coerce")

asmin = ap_scaled.min()
asmax = ap_scaled.max()
ap_min_train = ap_raw[ap_scaled <= asmin + 1e-6].mean()
ap_max_train = ap_raw[ap_scaled >= asmax - 1e-6].mean()

if np.isnan(ap_min_train):
    ap_min_train = ap_raw.min()
if np.isnan(ap_max_train):
    ap_max_train = ap_raw.max()

# Local team distance threshold: largest distance_min where indicator == 1
if "Local Team Indicator" in calib.columns:
    local_rows = calib[["Distance Minimum", "Local Team Indicator"]].dropna()
    try:
        local_threshold = float(
            local_rows.loc[local_rows["Local Team Indicator"] == 1, "Distance Minimum"].max()
        )
        if np.isnan(local_threshold):
            local_threshold = 75.0
    except Exception:
        local_threshold = 75.0
else:
    local_threshold = 75.0

# For key-driver comparison: median combined fan log from calib
try:
    calib_combined_log_median = pd.to_numeric(
        calib["Combined Fanbase (Log transformed)"], errors="coerce"
    ).median()
except Exception:
    calib_combined_log_median = None

# =====================================================
# STREAMLIT SETUP
# =====================================================

st.set_page_config(page_title="Optimatch Bowl Attendance Predictor", layout="wide")

st.title("🏈 Optimatch Bowl Attendance Predictor")
st.write(
    "Predict bowl attendance using CSMG’s Gradient Boosted model and Optimatch "
    "feature engine. This UI recomputes all model features from team, bowl, and "
    "venue lookups, calibrated to your original 2022–2025 projections."
)

# BUILD DROPDOWNS
teams_list = sorted(teams[TEAM_COL].unique())
venues_list = sorted(venues[VENUE_COL].unique())
bowls_list = sorted(bowl_tiers[BOWL_NAME_COL].unique())

# =====================================================
# BOWL + VENUE SELECTION
# =====================================================

st.header("Bowl & Venue Selection")

col_bowl, col_venue = st.columns(2)

with col_bowl:
    bowl_choice = st.selectbox("Bowl Name", bowls_list)

with col_venue:
    venue_choice = st.selectbox("Football Stadium", venues_list)

bowl_row = bowl_tiers[bowl_tiers[BOWL_NAME_COL] == bowl_choice].iloc[0]
venue_row = venues[venues[VENUE_COL] == venue_choice].iloc[0]

venue_capacity = safe_num(venue_row["Football Capacity"])
venue_lat = safe_num(venue_row["Lat"])
venue_lon = safe_num(venue_row["Lon"])
venue_city = venue_row["City"]
venue_state = venue_row["State"]

# In your training set, Venue Tier is separate. We approximate by inheriting
# the typical venue tier from the calibration file when this bowl is played.
calib_for_bowl = calib[calib["Bowl Game Name"] == bowl_choice]
if "Venue Tier" in calib.columns and not calib_for_bowl.empty:
    venue_tier = safe_num(calib_for_bowl["Venue Tier"].iloc[0])
else:
    venue_tier = 1.0  # default

bowl_tier = safe_num(bowl_row["Tier"])
bowl_owner = safe_num(bowl_row["Ownership"])
bowl_avg_viewers = safe_num(bowl_row["Avg Viewers"])
bowl_avg_att = safe_num(bowl_row["Avg Attendance"])

st.subheader(f"{bowl_choice} at {venue_choice}")
st.write(f"Location: {venue_city}, {venue_state}")
st.write(f"Capacity: {venue_capacity:,.0f}")
st.write(f"Historical Avg Attendance: {bowl_avg_att:,.0f}")

weekend_flag = 1 if st.checkbox("Is this game on a weekend?") else 0

# =====================================================
# TEAM SELECTION
# =====================================================

st.header("Team Selection")

col_t1, col_t2 = st.columns(2)

with col_t1:
    team1 = st.selectbox("Team 1", teams_list)

with col_t2:
    team2 = st.selectbox("Team 2", teams_list)

row1 = teams[teams[TEAM_COL] == team1].iloc[0]
row2 = teams[teams[TEAM_COL] == team2].iloc[0]

# =====================================================
# FEATURE ENGINEERING (MIRRORING YOUR MODEL)
# =====================================================

year_str = str(PRED_YEAR)

AP_RANK_COL = f"{year_str} AP Ranking"
AP_STRENGTH_COL = f"{year_str} AP Strength Score"
WINS_COL = f"{year_str} Wins"
TALENT_COL = f"{year_str} Talent"
BRAND_POWER_COL = f"Team Brand Power {year_str}"

# Core team stats
t1_ap = safe_num(row1.get(AP_RANK_COL, 0))
t2_ap = safe_num(row2.get(AP_RANK_COL, 0))

t1_wins = safe_num(row1.get(WINS_COL, 0))
t2_wins = safe_num(row2.get(WINS_COL, 0))

t1_talent = safe_num(row1.get(TALENT_COL, 0))
t2_talent = safe_num(row2.get(TALENT_COL, 0))

t1_brand = safe_num(row1.get(BRAND_POWER_COL, 0))
t2_brand = safe_num(row2.get(BRAND_POWER_COL, 0))

t1_ap_strength = safe_num(row1.get(AP_STRENGTH_COL, 0))
t2_ap_strength = safe_num(row2.get(AP_STRENGTH_COL, 0))

# Lat/Lon & distances
t1_lat = safe_num(row1.get("Latitude", 0))
t1_lon = safe_num(row1.get("Longitude", 0))
t2_lat = safe_num(row2.get("Latitude", 0))
t2_lon = safe_num(row2.get("Longitude", 0))

if all(np.isfinite([t1_lat, t1_lon, venue_lat, venue_lon])):
    team1_miles = haversine(t1_lat, t1_lon, venue_lat, venue_lon)
else:
    team1_miles = 0.0

if all(np.isfinite([t2_lat, t2_lon, venue_lat, venue_lon])):
    team2_miles = haversine(t2_lat, t2_lon, venue_lat, venue_lon)
else:
    team2_miles = 0.0

avg_wins = (t1_wins + t2_wins) / 2.0
total_wins = t1_wins + t2_wins

ap_vals = [v for v in [t1_ap, t2_ap] if v > 0]
avg_ap = np.mean(ap_vals) if ap_vals else 0.0
best_ap = min(ap_vals) if ap_vals else 0.0
worst_ap = max(ap_vals) if ap_vals else 0.0

avg_distance = (team1_miles + team2_miles) / 2.0
distance_min = min(team1_miles, team2_miles)
distance_imbalance = abs(team1_miles - team2_miles)

# Local team indicator from derived threshold
local_flag = 1 if distance_min <= local_threshold else 0

# Conference flags
conf1 = str(row1.get("Football FBS Conference", ""))
conf2 = str(row2.get("Football FBS Conference", ""))

SEC_present = int("Southeastern Conference" in (conf1, conf2))
B10_present = int("Big Ten Conference" in (conf1, conf2) or "Big Ten Conference" in (conf1, conf2))
B12_present = int("Big 12 Conference" in (conf1, conf2))
ACC_present = int("Atlantic Coast Conference" in (conf1, conf2))

# Matchup power (P4 vs P4 etc.)
level1 = str(row1.get("Conference Level", ""))
level2 = str(row2.get("Conference Level", ""))

if level1 == "Power 4" and level2 == "Power 4":
    matchup_power = 2
elif level1 == "Power 4" or level2 == "Power 4":
    matchup_power = 1
else:
    matchup_power = 0

# Combined fanbase: social + enrollment + alumni (via Team Fanbase Size)
t1_fanbase = safe_num(row1.get("Team Fanbase Size", 0))
t2_fanbase = safe_num(row2.get("Team Fanbase Size", 0))
combined_fanbase = t1_fanbase + t2_fanbase

# Log10 transform (matches your 6.1388... for 1,376,643)
combined_fanbase_log = np.log10(combined_fanbase) if combined_fanbase > 0 else 0.0

# AP Strength Score = average of team AP strength scores
ap_strength_score = (t1_ap_strength + t2_ap_strength) / 2.0

# MinMax scaled distance using training min/max
if dist_max_train > dist_min_train:
    minmax_distance = (avg_distance - dist_min_train) / (dist_max_train - dist_min_train)
    minmax_distance = max(0.0, min(1.0, minmax_distance))
else:
    minmax_distance = 0.0

# AP Strength Normalized using training min/max
if ap_max_train > ap_min_train:
    ap_strength_norm = (ap_strength_score - ap_min_train) / (ap_max_train - ap_min_train)
    ap_strength_norm = max(0.0, min(1.0, ap_strength_norm))
else:
    ap_strength_norm = 0.0

# =====================================================
# TRAVEL MAP
# =====================================================

st.subheader("Travel Map")

map_df = pd.DataFrame(
    [
        {"name": team1, "lat": t1_lat, "lon": t1_lon},
        {"name": team2, "lat": t2_lat, "lon": t2_lon},
        {"name": bowl_choice, "lat": venue_lat, "lon": venue_lon},
    ]
)

st.map(map_df[["lat", "lon"]])
st.caption("Markers show Team 1, Team 2, and the bowl venue.")

# =====================================================
# DYNAMIC TRAVEL MAP LINES (TEAM → VENUE)
# =====================================================
st.subheader("🗺️ Travel Routes to the Venue")

import pydeck as pdk

# Build map layers
line_data = [
    {
        "from_lon": t1_lon, "from_lat": t1_lat,
        "to_lon": venue_lon, "to_lat": venue_lat,
        "team": team1
    },
    {
        "from_lon": t2_lon, "from_lat": t2_lat,
        "to_lon": venue_lon, "to_lat": venue_lat,
        "team": team2
    }
]

point_data = [
    {"lon": t1_lon, "lat": t1_lat, "name": team1},
    {"lon": t2_lon, "lat": t2_lat,"name": team2},
    {"lon": venue_lon,"lat": venue_lat, "name": venue_choice}
]

line_layer = pdk.Layer(
    "LineLayer",
    data=line_data,
    get_source_position="[from_lon, from_lat]",
    get_target_position="[to_lon, to_lat]",
    get_color=[255, 0, 0],
    width_scale=3,
    width_min_pixels=2,
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=point_data,
    get_position="[lon, lat]",
    get_radius=50000,
    get_color=[0, 100, 255],
    pickable=True
)

view_state = pdk.ViewState(
    latitude=venue_lat,
    longitude=venue_lon,
    zoom=4
)

r = pdk.Deck(
    layers=[line_layer, point_layer],
    initial_view_state=view_state,
    tooltip={"text": "{name}"}
)

st.pydeck_chart(r)


# =====================================================
# BUILD FEATURE ROW
# =====================================================

row_data = {
    "Team 1 AP Ranking": t1_ap,
    "Team 1 Wins": t1_wins,
    "Team 1 247 Talent Composite Score": t1_talent,
    "Team 1 Brand Power": t1_brand,

    "Team 2 AP Ranking": t2_ap,
    "Team 2 Wins": t2_wins,
    "Team 2 247 Talent Composite Score": t2_talent,
    "Team 2 Brand Power": t2_brand,

    "Venue Tier": venue_tier,

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

feature_row = pd.DataFrame([[row_data.get(col, 0) for col in feature_cols]],
                           columns=feature_cols)

# =====================================================
# PREDICTION
# =====================================================

st.header("Prediction")

if st.button("Run Prediction"):

    # 1) Raw Gradient Boosted output
    raw_pred = float(model.predict(feature_row)[0])

    # 2) 70/30 blend with historical average
    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att

    bowl_lower = bowl_choice.lower()

    # 3) Special case: Hawai'i & Bahamas bowls use raw only
    if ("hawai" in bowl_lower) or ("bahamas" in bowl_lower):
        final_pred = raw_pred
    else:
        final_pred = blended_pred

    # 4) Bowl-specific boosts
    boosts = [
        ("gator", 1.05),
        ("pop-tarts", 1.05),
        ("pop tarts", 1.05),
        ("texas bowl", 1.05),
        ("music city", 1.05),
        ("alamo", 1.10),
        ("duke’s mayo", 1.03),
        ("duke's mayo", 1.03),
    ]
    for key, factor in boosts:
        if key in bowl_lower:
            final_pred *= factor
            break

    # 5) Capacity cap + floor
    final_pred = min(final_pred, venue_capacity)
    final_pred = max(final_pred, 0.0)

    pct_filled = final_pred / venue_capacity if venue_capacity > 0 else 0.0

    # =================================================
    # SIDE-BY-SIDE COMPARISON
    # =================================================
    st.subheader("📊 Attendance Comparison")

    diff = final_pred - bowl_avg_att
    pct_diff = diff / bowl_avg_att if bowl_avg_att > 0 else 0.0

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Predicted Attendance", f"{final_pred:,.0f}")
    with colB:
        st.metric("Historical Avg Attendance", f"{bowl_avg_att:,.0f}")
    with colC:
        st.metric("Difference", f"{diff:,.0f}", delta=f"{pct_diff:.1%}")

    st.metric("Projected % Filled", f"{pct_filled:.1%}")
    st.write(f"Stadium Capacity: {venue_capacity:,.0f}")

 # =====================================================
# FEATURE IMPORTANCE BREAKDOWN (LOCAL TO THIS MATCHUP)
# =====================================================
st.subheader("📊 Feature Breakdown for This Prediction")

# Get model-wide feature importances
importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances,
    "Value": feature_row.iloc[0].values
})

# Normalize importances for readability
feat_df = feat_df.sort_values("Importance", ascending=False)
top_feats = feat_df.head(10)

# Display table
st.write("Top 10 Model Drivers:")
st.dataframe(top_feats)

# Plot bar chart
st.bar_chart(top_feats.set_index("Feature")["Importance"])

# =====================================================
# ALTERNATIVE SCENARIO: Swap Team 1 or Team 2
# =====================================================
st.subheader("🔄 Alternative Scenario")

with st.expander("Run a What-If Scenario (Swap in a Major Brand Team)"):
    big_brand_list = [
        "Michigan", "Ohio State", "Texas", "LSU", "Alabama", "Georgia",
        "Notre Dame", "USC", "Oklahoma", "Penn State", "Oregon",
        "Florida State", "Clemson", "Tennessee", "Auburn"
    ]

    # Which team slot to replace?
    replace_slot = st.radio("Replace which team?", ["Team 1", "Team 2"])

    # Select big brand team
    brand_team = st.selectbox("Choose a top brand team:", big_brand_list)

    if st.button("Run Scenario"):
        # Force replacement
        t1 = brand_team if replace_slot == "Team 1" else team1
        t2 = brand_team if replace_slot == "Team 2" else team2

        st.write(f"Scenario: **{t1} vs {t2}** at **{bowl_choice}**")

        # Recompute features with swapped team
        # (reuse your existing code; encapsulate into a function later)
        st.warning("This button requires integrating your feature-engineering block into a function. Once placed inside a function, the prediction, distance, and key drivers recompute instantly.")


# =====================================================
# ENHANCED KEY DRIVER SUMMARY
# =====================================================
st.subheader("Key Driver Summary")

drivers = []

# -------- TEAM PROXIMITY --------
if distance_min < 100:
    drivers.append("At least one team is very close to the venue (<100 miles), which strongly boosts attendance due to drive-in ease.")
elif distance_min < 250:
    drivers.append("A nearby team (<250 miles) provides a solid regional attendance lift.")
elif distance_min < 500:
    drivers.append("Both teams are within reasonable travel distance, supporting moderate fan travel.")
else:
    drivers.append("Both teams face longer travel distances, which historically moderates attendance.")

# -------- LONG TRAVEL IMBALANCE --------
if distance_imbalance > 400:
    drivers.append("There is a major travel imbalance between fanbases, meaning one school may dominate in-person turnout.")

# -------- BIG BRAND PROGRAMS --------
big_brands = [
    "Michigan", "Ohio State", "Texas", "LSU", "Alabama", "Georgia",
    "Notre Dame", "Oklahoma", "USC", "Penn State", "Oregon",
    "Florida State", "Clemson", "Tennessee", "Auburn"
]

brands_present = []
for t in [team1, team2]:
    for brand in big_brands:
        if brand.lower() in t.lower():
            brands_present.append(brand)

if brands_present:
    bp_str = ", ".join(sorted(set(brands_present)))
    drivers.append(f"National brands present ({bp_str}). These programs traditionally travel well and elevate overall demand.")

# -------- FANBASE STRENGTH --------
if combined_fanbase_log > calib_combined_log_median + 0.2:
    drivers.append("The combined fanbase size is significantly larger than typical FBS matchups, supporting stronger turnout.")
elif combined_fanbase_log < calib_combined_log_median - 0.2:
    drivers.append("This matchup features smaller fanbases, which historically leads to more modest attendance figures.")

# -------- MATCHUP QUALITY --------
if matchup_power == 2:
    drivers.append("Power vs. Power matchup substantially increases national interest and in-person attendance.")
elif matchup_power == 1:
    drivers.append("Having at least one Power Conference team helps boost the appeal of this bowl.")
else:
    drivers.append("Group of Five vs Group of Five matchups typically rely more on regional proximity for attendance strength.")

# -------- CONFERENCE BRAND EFFECT --------
if SEC_present:
    drivers.append("An SEC program is participating — SEC teams historically generate high demand and strong travel behavior.")
if B10_present:
    drivers.append("A Big Ten program is participating — their fans traditionally travel well and lift bowl attendance.")
if ACC_present:
    drivers.append("An ACC team contributes added brand visibility.")
if B12_present:
    drivers.append("A Big 12 program adds solid regional and national interest.")

# -------- LOCAL FANBASE EFFECT --------
if local_flag == 1:
    drivers.append("The presence of a local or near-local team provides a significant attendance boost.")

# -------- VENUE & MARKET -------—
if venue_tier >= 2:
    drivers.append("This game is hosted in a high-tier venue, enhancing the bowl experience and spectator draw.")

# -------- ALUMNI DISPERSION IMPACT --------
t1_alumni = str(row1.get("Alumni Dispersion", "")).lower()
t2_alumni = str(row2.get("Alumni Dispersion", "")).lower()

def alumni_comment(team, dispersion, distance):
    msg = None

    if "local" in dispersion:
        if distance < local_threshold:
            msg = f"{team} has a strong local alumni base near the bowl site, which should meaningfully lift attendance."
        else:
            msg = f"{team} typically relies on local alumni turnout, but this bowl is farther from its primary alumni region."

    elif "regional" in dispersion:
        if distance < 400:
            msg = f"{team}'s alumni network is regionally concentrated, and the relatively close travel distance should support strong turnout."
        else:
            msg = f"{team} has a regionally distributed alumni base, but the travel distance may moderate participation."

    elif "national" in dispersion:
        if distance > 700:
            msg = f"{team} has a nationally dispersed alumni base, which helps offset the longer travel distance to the bowl."
        else:
            msg = f"{team}'s national alumni footprint supports flexible travel and broader turnout potential."

    return msg

alumni_msgs = [
    alumni_comment(team1, t1_alumni, team1_miles),
    alumni_comment(team2, t2_alumni, team2_miles)
]

for m in alumni_msgs:
    if m:
        drivers.append(m)
# -------- DEFAULT IF EMPTY --------
if not drivers:
    drivers.append("This matchup aligns with typical bowl attendance patterns in our model.")

for d in drivers:
    st.write("• " + d)



