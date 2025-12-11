import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# CONSTANTS / CONFIG
# =====================================================

PRED_YEAR = 2025  # using 2025 stats for predictions

TEAM_FILE = "2025 Bowl Games - UI Team Lookup.csv"
VENUE_FILE = "2025 Bowl Games - UI Venue Lookup.csv"
BOWL_TIERS_FILE = "2025 Bowl Games - Bowl Tiers.csv"

# HISTORICAL CALIBRATION: 2022–2024 ACTUAL GAMES
CALIB_FILE = "2025 Bowl Games - 2022-2024 Bowl Games (7).csv"

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
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def safe_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def estimate_travel_hours(miles: float) -> float:
    """Very rough travel time estimate in hours."""
    if miles <= 0:
        return 0.0
    if miles < 400:
        # assume driving
        return miles / 55.0 + 0.5
    elif miles < 900:
        # longer drive / short flight
        return miles / 60.0 + 1.0
    else:
        # assume commercial flight + airport overhead
        return 3.0 + 2.5

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
calib = pd.read_csv(CALIB_FILE)  # 2022–2024 historic bowls

for df in (teams, venues, bowl_tiers, calib):
    df.columns = df.columns.str.strip()

# =====================================================
# DERIVE SCALERS / THRESHOLDS FROM HISTORICAL DATA
# =====================================================

# Distance MinMax scaling
dist_raw = pd.to_numeric(calib["Avg Distace Traveled"], errors="coerce")
dist_scaled = pd.to_numeric(calib["MinMax Scale Distcance"], errors="coerce")

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

# Local team distance threshold
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

# median log fanbase from calibration
try:
    calib_combined_log_median = pd.to_numeric(
        calib["Combined Fanbase (Log transformed)"], errors="coerce"
    ).median()
except Exception:
    calib_combined_log_median = None

# =====================================================
# STREAMLIT UI SETUP
# =====================================================

st.set_page_config(page_title="Optimatch Bowl Attendance Predictor", layout="wide")

st.title("🏈 Optimatch Bowl Attendance Predictor")
st.write(
    "Predict bowl attendance using CSMG’s Gradient Boosted model and Optimatch feature engine. "
    "This UI recomputes model features from team, bowl, and venue lookups, calibrated on 2022–2024 bowl games."
)

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

# approximate venue tier from historical data if available
calib_for_bowl = calib[calib["Bowl Game Name"] == bowl_choice]
if "Venue Tier" in calib.columns and not calib_for_bowl.empty:
    venue_tier = safe_num(calib_for_bowl["Venue Tier"].iloc[0])
else:
    venue_tier = 1.0

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
# FEATURE ENGINEERING (MIRRORING MODEL TRAINING)
# =====================================================

year_str = str(PRED_YEAR)
AP_RANK_COL = f"{year_str} AP Ranking"
AP_STRENGTH_COL = f"{year_str} AP Strength Score"
WINS_COL = f"{year_str} Wins"
TALENT_COL = f"{year_str} Talent"
BRAND_POWER_COL = f"Team Brand Power {year_str}"

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

local_flag = 1 if distance_min <= local_threshold else 0

conf1 = str(row1.get("Football FBS Conference", ""))
conf2 = str(row2.get("Football FBS Conference", ""))

SEC_present = int("Southeastern Conference" in (conf1, conf2))
B10_present = int("Big Ten Conference" in (conf1, conf2) or "Big Ten Conference" in (conf1, conf2))
B12_present = int("Big 12 Conference" in (conf1, conf2))
ACC_present = int("Atlantic Coast Conference" in (conf1, conf2))

level1 = str(row1.get("Conference Level", ""))
level2 = str(row2.get("Conference Level", ""))

if level1 == "Power 4" and level2 == "Power 4":
    matchup_power = 2
elif level1 == "Power 4" or level2 == "Power 4":
    matchup_power = 1
else:
    matchup_power = 0

t1_fanbase = safe_num(row1.get("Team Fanbase Size", 0))
t2_fanbase = safe_num(row2.get("Team Fanbase Size", 0))
combined_fanbase = t1_fanbase + t2_fanbase
combined_fanbase_log = np.log10(combined_fanbase) if combined_fanbase > 0 else 0.0

ap_strength_score = (t1_ap_strength + t2_ap_strength) / 2.0

if dist_max_train > dist_min_train:
    minmax_distance = (avg_distance - dist_min_train) / (dist_max_train - dist_min_train)
    minmax_distance = max(0.0, min(1.0, minmax_distance))
else:
    minmax_distance = 0.0

if ap_max_train > ap_min_train:
    ap_strength_norm = (ap_strength_score - ap_min_train) / (ap_max_train - ap_min_train)
    ap_strength_norm = max(0.0, min(1.0, ap_strength_norm))
else:
    ap_strength_norm = 0.0

# =====================================================
# TRAVEL MAP + ROUTE LINES
# =====================================================

st.subheader("🗺️ Travel Routes & Estimated Travel Time")

line_data = [
    {
        "from_lon": t1_lon, "from_lat": t1_lat,
        "to_lon": venue_lon, "to_lat": venue_lat,
        "name": team1,
        "miles": team1_miles,
        "hours": estimate_travel_hours(team1_miles),
    },
    {
        "from_lon": t2_lon, "from_lat": t2_lat,
        "to_lon": venue_lon, "to_lat": venue_lat,
        "name": team2,
        "miles": team2_miles,
        "hours": estimate_travel_hours(team2_miles),
    }
]

point_data = [
    {"lon": t1_lon, "lat": t1_lat, "name": team1},
    {"lon": t2_lon, "lat": t2_lat, "name": team2},
    {"lon": venue_lon, "lat": venue_lat, "name": venue_choice},
]

line_layer = pdk.Layer(
    "LineLayer",
    data=line_data,
    get_source_position="[from_lon, from_lat]",
    get_target_position="[to_lon, to_lat]",
    get_color=[255, 0, 0],
    width_scale=3,
    width_min_pixels=2,
    pickable=True,
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=point_data,
    get_position="[lon, lat]",
    get_radius=50000,
    get_color=[0, 100, 255],
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=venue_lat,
    longitude=venue_lon,
    zoom=4,
)

route_deck = pdk.Deck(
    layers=[line_layer, point_layer],
    initial_view_state=view_state,
    tooltip={"html": "<b>{name}</b>"},
)

st.pydeck_chart(route_deck)
st.caption(
    "Line paths show travel routes from each campus to the bowl venue. "
    "Travel hours are estimated internally for scenario planning."
)

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

feature_row = pd.DataFrame(
    [[row_data.get(col, 0.0) for col in feature_cols]],
    columns=feature_cols
)

# =====================================================
# PREDICTION
# =====================================================

st.header("Prediction")

if st.button("Run Prediction"):

    # Raw model prediction
    raw_pred = float(model.predict(feature_row)[0])

    # 70/30 blend with historical average
    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att

    bowl_lower = bowl_choice.lower()

    # Special case: Hawai'i & Bahamas bowls use raw only
    if ("hawai" in bowl_lower) or ("bahamas" in bowl_lower):
        final_pred = raw_pred
    else:
        final_pred = blended_pred

    # Bowl-specific boosts
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

    # Capacity cap + floor
    final_pred = min(final_pred, venue_capacity)
    final_pred = max(final_pred, 0.0)

    pct_filled = final_pred / venue_capacity if venue_capacity > 0 else 0.0

    # Side-by-side comparison
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
    # TEAM VISITATION INDEX (0–100)
    # =====================================================
    st.subheader("👥 Team Visitation Index")

    def compute_tvi(
        miles, 
        fanbase_size, 
        brand_power, 
        wins, 
        ap_strength, 
        alumni_dispersion,
        conference
    ):
        score = 50  # baseline

        # ---- Distance effect ----
        if miles <= 150:
            score += 20
        elif miles <= 350:
            score += 12
        elif miles <= 700:
            score += 4
        else:
            score -= 10

        # ---- Fanbase size ----
        if fanbase_size > 600000:
            score += 10
        elif fanbase_size > 300000:
            score += 6
        elif fanbase_size > 100000:
            score += 3

        # ---- Brand power ----
        if brand_power > 75:
            score += 8
        elif brand_power > 50:
            score += 4

        # ---- Wins ----
        if wins >= 9:
            score += 6
        elif wins >= 7:
            score += 3

        # ---- AP Strength ----
        if ap_strength > 0 and ap_strength <= 20:
            score += 4
        elif ap_strength <= 35:
            score += 2

        # ---- Alumni dispersion ----
        ad = str(alumni_dispersion).lower()
        if "national" in ad:
            score += 5
        elif "regional" in ad:
            score += 2
        elif "local" in ad:
            score -= 2

        # ---- Conference travel culture ----
        conf = str(conference).lower()
        if "sec" in conf:
            score += 8
        elif "big ten" in conf or "big 10" in conf:
            score += 6
        elif "big 12" in conf:
            score += 4
        elif "acc" in conf:
            score += 2

        return max(0, min(100, score))

    # ---- Compute TVI for each team ----
    t1_tvi = compute_tvi(
        miles=team1_miles,
        fanbase_size=t1_fanbase,
        brand_power=t1_brand,
        wins=t1_wins,
        ap_strength=t1_ap_strength,
        alumni_dispersion=row1.get("Alumni Dispersion", ""),
        conference=row1.get("Football FBS Conference", "")
    )

    t2_tvi = compute_tvi(
        miles=team2_miles,
        fanbase_size=t2_fanbase,
        brand_power=t2_brand,
        wins=t2_wins,
        ap_strength=t2_ap_strength,
        alumni_dispersion=row2.get("Alumni Dispersion", ""),
        conference=row2.get("Football FBS Conference", "")
    )

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.metric(f"{team1} Visitation Index", f"{t1_tvi} / 100")

    with col_t2:
        st.metric(f"{team2} Visitation Index", f"{t2_tvi} / 100")

    avg_tvi = (t1_tvi + t2_tvi) / 2
    st.write(f"**Overall Matchup Travel Expectation:** {avg_tvi:.1f} / 100")

    
    # =====================================================
    # CONFIDENCE INTERVALS + MODEL STABILITY + RISK
    # =====================================================
    
    st.subheader("Uncertainty & Risk")

    ci_width = 0.08  # ±8%
    ci_low = max(0.0, final_pred * (1 - ci_width))
    ci_high = final_pred * (1 + ci_width)

    st.write(
        f"Estimated Confidence Interval (±{int(ci_width*100)}%): "
        f"**{ci_low:,.0f} – {ci_high:,.0f}**"
    )

    stability_score = 10.0

    if avg_distance > 1000:
        stability_score -= 2
    elif avg_distance > 700:
        stability_score -= 1

    if calib_combined_log_median is not None and combined_fanbase_log < calib_combined_log_median:
        stability_score -= 2

    if matchup_power == 0:
        stability_score -= 1

    if distance_imbalance > 400:
        stability_score -= 1

    stability_score = max(1.0, min(10.0, stability_score))
    st.write(f"**Model Stability Score:** {stability_score:.1f} / 10")

    risk_raw = 1.0 - (stability_score / 10.0)
    if risk_raw <= 0.3:
        risk_label = "Low Risk"
        risk_color = "#2ecc71"
    elif risk_raw <= 0.6:
        risk_label = "Moderate Risk"
        risk_color = "#f1c40f"
    else:
        risk_label = "High Risk"
        risk_color = "#e74c3c"

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 12px;
            border-radius:16px;
            background-color:{risk_color};
            color:white;
            font-weight:bold;
            margin-top:4px;
        ">
        {risk_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # TRAVEL PROPENSITY + SELLOUT PROBABILITY
    # =====================================================
    st.subheader("Travel Behavior & Sellout Likelihood")

    travel_score = 10.0

    if avg_distance > 1200:
        travel_score -= 3
    elif avg_distance > 800:
        travel_score -= 2
    elif avg_distance > 500:
        travel_score -= 1

    if local_flag == 1:
        travel_score += 2
    elif distance_min < 250:
        travel_score += 1

    if calib_combined_log_median is not None:
        if combined_fanbase_log > calib_combined_log_median + 0.2:
            travel_score += 1
        elif combined_fanbase_log < calib_combined_log_median - 0.2:
            travel_score -= 1

    big_brands = [
        "Michigan", "Ohio State", "Texas", "LSU", "Alabama", "Georgia",
        "Notre Dame", "USC", "Oklahoma", "Penn State", "Oregon",
        "Florida State", "Clemson", "Tennessee", "Auburn"
    ]
    for t in [team1, team2]:
        if any(b.lower() in t.lower() for b in big_brands):
            travel_score += 1

    travel_score = max(1.0, min(10.0, travel_score))
    st.write(f"**Travel Propensity Score:** {travel_score:.1f} / 10")

    fill_ratio = final_pred / venue_capacity if venue_capacity > 0 else 0.0
    sellout_prob = fill_ratio

    if bowl_tier >= 2:
        sellout_prob += 0.05
    if matchup_power == 2:
        sellout_prob += 0.05

    sellout_prob += (stability_score - 5) * 0.01
    sellout_prob = float(np.clip(sellout_prob, 0.0, 1.0))

    st.write(f"**Estimated Sellout Probability:** {sellout_prob*100:.1f}%")

    # =====================================================
    # HEAD-TO-HEAD INTEREST INDEX
    # =====================================================
    st.subheader("Head-to-Head Interest Index")

    interest_index = 5.0

    brand_count = sum(any(b.lower() in t.lower() for b in big_brands) for t in [team1, team2])
    if brand_count == 2:
        interest_index += 2.0
    elif brand_count == 1:
        interest_index += 1.0

    if matchup_power == 2:
        interest_index += 1.0
    elif matchup_power == 1:
        interest_index += 0.5

    if bowl_tier >= 2:
        interest_index += 0.5

    if SEC_present:
        interest_index += 0.5
    if B10_present:
        interest_index += 0.5

    if ap_strength_score > np.percentile(ap_raw.dropna(), 75):
        interest_index += 0.5

    interest_index = max(1.0, min(10.0, interest_index))
    st.write(f"**Head-to-Head Interest Index:** {interest_index:.1f} / 10")

    # =====================================================
    # FEATURE BREAKDOWN PANEL
    # =====================================================
    st.subheader("📊 Top Model Drivers for This Prediction")

    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
        "Value": feature_row.iloc[0].values
    })

    feat_df = feat_df.sort_values("Importance", ascending=False)
    top_feats = feat_df.head(10)

    st.write("Top 10 Features by Importance:")
    st.dataframe(top_feats)

    st.bar_chart(top_feats.set_index("Feature")["Importance"])

    # =====================================================
    # ENHANCED KEY DRIVER SUMMARY (INCLUDING ALUMNI)
    # =====================================================
    st.subheader("Key Driver Summary")

    drivers = []

    if distance_min < 100:
        drivers.append("At least one team is very close to the venue (<100 miles), which strongly boosts attendance due to drive-in ease.")
    elif distance_min < 250:
        drivers.append("A nearby team (<250 miles) provides a strong regional attendance lift.")
    elif distance_min < 500:
        drivers.append("Both teams are within reasonable travel distance, supporting moderate fan travel.")
    else:
        drivers.append("Both teams face longer travel distances, which historically moderates attendance.")

    if distance_imbalance > 400:
        drivers.append("There is a major travel imbalance between fanbases, meaning one school may dominate in-person turnout.")

    brands_present = []
    for t in [team1, team2]:
        for brand in big_brands:
            if brand.lower() in t.lower():
                brands_present.append(brand)
    if brands_present:
        bp_str = ", ".join(sorted(set(brands_present)))
        drivers.append(f"National brands present ({bp_str}). These programs traditionally travel well and elevate overall demand.")

    if calib_combined_log_median is not None:
        if combined_fanbase_log > calib_combined_log_median + 0.2:
            drivers.append("The combined fanbase size is significantly larger than typical FBS matchups, supporting stronger turnout.")
        elif combined_fanbase_log < calib_combined_log_median - 0.2:
            drivers.append("This matchup features smaller fanbases, which historically leads to more modest attendance figures.")

    if matchup_power == 2:
        drivers.append("Power vs. Power matchup substantially increases national interest and in-person attendance.")
    elif matchup_power == 1:
        drivers.append("Having at least one Power Conference team helps boost the appeal of this bowl.")
    else:
        drivers.append("Group of Five vs Group of Five matchups typically rely more on regional proximity for attendance strength.")

    if SEC_present:
        drivers.append("An SEC program is participating — SEC teams historically generate high demand and strong travel behavior.")
    if B10_present:
        drivers.append("A Big Ten program is participating — their fans traditionally travel well and lift bowl attendance.")
    if ACC_present:
        drivers.append("An ACC team contributes added brand visibility.")
    if B12_present:
        drivers.append("A Big 12 program adds solid regional and national interest.")

    if venue_tier >= 2:
        drivers.append("This game is hosted in a high-tier venue, enhancing the bowl experience and spectator draw.")

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

    if local_flag == 1:
        drivers.append("The presence of a local or near-local team provides a significant attendance boost.")

    if not drivers:
        drivers.append("This matchup aligns with typical bowl attendance patterns in our model.")

    for d in drivers:
        st.write("• " + d)

    # =====================================================
    # SAVE CONTEXT FOR OTHER PANELS & SCENARIO HISTORY
    # =====================================================

    if "scenario_history" not in st.session_state:
        st.session_state["scenario_history"] = []
    st.session_state["scenario_history"].append(
        {
            "features": row_data,
            "prediction": final_pred,
            "team1": team1,
            "team2": team2,
            "bowl": bowl_choice,
            "venue": venue_choice,
        }
    )

    st.session_state["scenario_context"] = {
        "bowl": bowl_choice,
        "venue": venue_choice,
        "team1": team1,
        "team2": team2,
        "final_pred": final_pred,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "stability_score": stability_score,
        "travel_score": travel_score,
        "sellout_prob": sellout_prob,
        "risk_label": risk_label,
        "bowl_avg_att": bowl_avg_att,
        "venue_capacity": venue_capacity,
        "PRED_YEAR": PRED_YEAR,
        "interest_index": interest_index,
    }

# =====================================================
# LIVE COMPARISON TO SIMILAR HISTORICAL MATCHUPS
# =====================================================

st.header("Similar Historical Matchups")

if "scenario_history" in st.session_state and len(st.session_state["scenario_history"]) > 0:
    names = [
        f"{s['team1']} vs {s['team2']} ({s['bowl']})"
        for s in st.session_state["scenario_history"]
    ]
    selected = st.selectbox("Select a scenario to compare", names)
    idx = names.index(selected)
    current_features = st.session_state["scenario_history"][idx]["features"]

    sim_cols = [
        "Avg Distace Traveled",
        "Combined Fanbase (Log transformed)",
        "Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)",
        "Bowl Tier"
    ]

    calib_sim = calib.copy()
    for c in sim_cols:
        calib_sim[c] = pd.to_numeric(calib_sim[c], errors="coerce")

    calib_sim = calib_sim.dropna(subset=sim_cols)

    # Only past years (< 2025)
    if "Year" in calib_sim.columns:
        calib_sim = calib_sim[calib_sim["Year"] < 2025]

    if not calib_sim.empty:
        cur_vec = np.array([current_features[c] for c in sim_cols], dtype=float)
        hist_mat = calib_sim[sim_cols].to_numpy(dtype=float)

        dists = np.linalg.norm(hist_mat - cur_vec, axis=1)
        calib_sim["sim_distance"] = dists

        top_sim = calib_sim.nsmallest(3, "sim_distance")

        if not top_sim.empty:
            display_cols = [
                "Year",
                "Bowl Game Name",
                "Team 1",
                "Team 2",
                "Attendance",
                "Actual Attendance",
                "Bowl Avg Attendees"
            ]
            existing_cols = [c for c in display_cols if c in top_sim.columns]
            st.write("Most similar historical games based on distance, fanbase size, bowl tier, and matchup type:")
            st.dataframe(top_sim[existing_cols])
        else:
            st.write("No comparable historical games found.")
    else:
        st.write("Historical dataset is empty after filtering.")
else:
    st.write("Run one or more predictions to see similar historical matchups.")

# =====================================================
# VENUE SCENARIO: MOVE BOWL TO ANOTHER STADIUM
# =====================================================

st.header("📍 Venue Scenario: Move This Bowl to Another Stadium")

alt_venue_choice = st.selectbox(
    "Select an alternative venue",
    sorted(venues[VENUE_COL].unique()),
    index=sorted(venues[VENUE_COL].unique()).index(venue_choice)
    if venue_choice in venues[VENUE_COL].values else 0
)

if st.button("Run Alternative Venue Scenario"):
    alt_venue_row = venues[venues[VENUE_COL] == alt_venue_choice].iloc[0]
    alt_capacity = safe_num(alt_venue_row["Football Capacity"])
    alt_lat = safe_num(alt_venue_row["Lat"])
    alt_lon = safe_num(alt_venue_row["Lon"])

    if all(np.isfinite([t1_lat, t1_lon, alt_lat, alt_lon])):
        alt_t1_miles = haversine(t1_lat, t1_lon, alt_lat, alt_lon)
    else:
        alt_t1_miles = 0.0

    if all(np.isfinite([t2_lat, t2_lon, alt_lat, alt_lon])):
        alt_t2_miles = haversine(t2_lat, t2_lon, alt_lat, alt_lon)
    else:
        alt_t2_miles = 0.0

    alt_avg_distance = (alt_t1_miles + alt_t2_miles) / 2.0
    alt_distance_min = min(alt_t1_miles, alt_t2_miles)
    alt_distance_imbalance = abs(alt_t1_miles - alt_t2_miles)
    alt_local_flag = 1 if alt_distance_min <= local_threshold else 0

    calib_for_venue = calib[calib["Venue"] == alt_venue_choice]
    if "Venue Tier" in calib.columns and not calib_for_venue.empty:
        alt_venue_tier = safe_num(calib_for_venue["Venue Tier"].iloc[0])
    else:
        alt_venue_tier = venue_tier

    alt_row_data = dict(row_data)
    alt_row_data.update({
        "Venue Tier": alt_venue_tier,
        "Team 1 Miles to Venue": alt_t1_miles,
        "Team 2 Miles to Venue": alt_t2_miles,
        "Avg Distace Traveled": alt_avg_distance,
        "Distance Minimum": alt_distance_min,
        "Distance Imbalance": alt_distance_imbalance,
        "Local Team Indicator": alt_local_flag,
        "MinMax Scale Distcance": (
            (alt_avg_distance - dist_min_train) / (dist_max_train - dist_min_train)
            if dist_max_train > dist_min_train else 0.0
        ),
    })
    alt_row_data["MinMax Scale Distcance"] = max(
        0.0, min(1.0, alt_row_data["MinMax Scale Distcance"])
    )

    alt_feature_row = pd.DataFrame(
        [[alt_row_data.get(c, 0.0) for c in feature_cols]],
        columns=feature_cols
    )

    alt_raw = float(model.predict(alt_feature_row)[0])
    alt_blend = 0.7 * alt_raw + 0.3 * bowl_avg_att

    bowl_lower = bowl_choice.lower()
    if ("hawai" in bowl_lower) or ("bahamas" in bowl_lower):
        alt_final = alt_raw
    else:
        alt_final = alt_blend

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
            alt_final *= factor
            break

    alt_final = min(max(alt_final, 0.0), alt_capacity)
    alt_pct_filled = alt_final / alt_capacity if alt_capacity > 0 else 0.0

    st.write(f"**Alternative Venue:** {alt_venue_choice}")
    st.write(f"Predicted Attendance: **{alt_final:,.0f}**")
    st.write(f"Capacity: {alt_capacity:,.0f} | Projected % Filled: {alt_pct_filled:.1%}")

# =====================================================
# TEAM SCORECARDS + MATCHUP SCORECARD
# =====================================================

st.header("Team & Matchup Scorecards")

if "scenario_context" in st.session_state:
    ctx = st.session_state["scenario_context"]

    col_team1, col_team2 = st.columns(2)

    with col_team1:
        st.subheader(f"Team 1: {ctx['team1']}")
        st.write(f"Conference: {row1.get('Football FBS Conference', '')}")
        st.write(f"Wins ({ctx['PRED_YEAR']}): {t1_wins}")
        st.write(f"AP Strength Score ({ctx['PRED_YEAR']}): {t1_ap_strength:.1f}")
        st.write(f"Brand Power: {t1_brand:,.0f}")
        st.write(f"Fanbase Size Index: {t1_fanbase:,.0f}")
        st.write(f"Distance to Venue: {team1_miles:.0f} miles")
        st.write(f"Alumni Dispersion: {row1.get('Alumni Dispersion', '')}")

    with col_team2:
        st.subheader(f"Team 2: {ctx['team2']}")
        st.write(f"Conference: {row2.get('Football FBS Conference', '')}")
        st.write(f"Wins ({ctx['PRED_YEAR']}): {t2_wins}")
        st.write(f"AP Strength Score ({ctx['PRED_YEAR']}): {t2_ap_strength:.1f}")
        st.write(f"Brand Power: {t2_brand:,.0f}")
        st.write(f"Fanbase Size Index: {t2_fanbase:,.0f}")
        st.write(f"Distance to Venue: {team2_miles:.0f} miles")
        st.write(f"Alumni Dispersion: {row2.get('Alumni Dispersion', '')}")

    st.subheader("Matchup Scorecard")
    st.write(f"**Predicted Attendance:** {ctx['final_pred']:,.0f}")
    st.write(f"**Confidence Interval:** {ctx['ci_low']:,.0f} – {ctx['ci_high']:,.0f}")
    st.write(f"**Model Stability Score:** {ctx['stability_score']:.1f} / 10")
    st.write(f"**Travel Propensity Score:** {ctx['travel_score']:.1f} / 10")
    st.write(f"**Sellout Probability:** {ctx['sellout_prob']*100:.1f}%")
    st.write(f"**Head-to-Head Interest Index:** {ctx['interest_index']:.1f} / 10")
    st.write(f"**Matchup Power Score:** {matchup_power} (2 = P4vP4, 1 = P4vG5, 0 = G5vG5)")
else:
    st.write("Run a prediction to populate scorecards.")

# =====================================================
# SCENARIO SAVING
# =====================================================

st.header("💾 Save Scenario")

if "scenario_context" in st.session_state:
    ctx = st.session_state["scenario_context"]

    folder_name = st.text_input("Scenario Folder / Group", "2025 Bowl Predictions")
    default_scenario_name = f"{ctx['bowl']} – {ctx['team1']} vs {ctx['team2']}"
    scenario_name = st.text_input("Scenario Name", default_scenario_name)

    if st.button("Save This Scenario"):
        scenario_row = {
            "folder": folder_name,
            "scenario_name": scenario_name,
            "year": ctx["PRED_YEAR"],
            "bowl": ctx["bowl"],
            "venue": ctx["venue"],
            "team1": ctx["team1"],
            "team2": ctx["team2"],
            "predicted_attendance": ctx["final_pred"],
            "ci_low": ctx["ci_low"],
            "ci_high": ctx["ci_high"],
            "stability_score": ctx["stability_score"],
            "travel_score": ctx["travel_score"],
            "sellout_prob": ctx["sellout_prob"],
            "risk_label": ctx["risk_label"],
            "bowl_avg_att": ctx["bowl_avg_att"],
            "venue_capacity": ctx["venue_capacity"],
            "interest_index": ctx["interest_index"],
        }

        try:
            existing = pd.read_csv("saved_scenarios.csv")
            existing = pd.concat([existing, pd.DataFrame([scenario_row])], ignore_index=True)
        except FileNotFoundError:
            existing = pd.DataFrame([scenario_row])

        existing.to_csv("saved_scenarios.csv", index=False)
        st.success("Scenario saved.")
else:
    st.write("Run a prediction to save scenarios.")

st.subheader("📁 Saved Scenarios")

try:
    saved = pd.read_csv("saved_scenarios.csv")
    folder_options = sorted(saved["folder"].unique())
    selected_folder = st.selectbox("Select Scenario Folder", folder_options)
    st.dataframe(saved[saved["folder"] == selected_folder])
except FileNotFoundError:
    st.write("No scenarios saved yet.")


