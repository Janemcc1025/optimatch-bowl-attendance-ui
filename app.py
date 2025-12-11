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

AIRPORTS = {
    "LAX - Los Angeles": (33.9416, -118.4085),
    "SFO - San Francisco": (37.6213, -122.3790),
    "SEA - Seattle": (47.4502, -122.3088),
    "PHX - Phoenix Sky Harbor": (33.4342, -112.0116),
    "LAS - Las Vegas": (36.0840, -115.1537),
    "DEN - Denver": (39.8561, -104.6737),
    "DFW - Dallas/Fort Worth": (32.8998, -97.0403),
    "IAH - Houston Intercontinental": (29.9902, -95.3368),
    "ATL - Atlanta": (33.6407, -84.4277),
    "MCO - Orlando": (28.4312, -81.3081),
    "MIA - Miami": (25.7959, -80.2870),
    "TPA - Tampa": (27.9747, -82.5333),
    "CLT - Charlotte": (35.2144, -80.9473),
    "BOS - Boston Logan": (42.3656, -71.0096),
    "JFK - New York JFK": (40.6413, -73.7781),
    "EWR - Newark": (40.6895, -74.1745),
    "PHL - Philadelphia": (39.8744, -75.2424),
    "BNA - Nashville": (36.1263, -86.6774),
    "MSY - New Orleans": (29.9934, -90.2580),
    "ELP - El Paso": (31.7982, -106.3960),
    "HNL - Honolulu": (21.3245, -157.9251),
    "BOI - Boise": (43.5644, -116.2228),
    "SLC - Salt Lake City": (40.7899, -111.9791)
}


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


def find_nearest_airport(venue_lat, venue_lon):
    closest_airport = None
    closest_dist = float("inf")

    for name, (alat, alon) in AIRPORTS.items():
        dist = haversine(venue_lat, venue_lon, alat, alon)
        if dist < closest_dist:
            closest_dist = dist
            closest_airport = name

    return closest_airport, closest_dist


def compute_venue_accessibility_score(
    venue_capacity,
    team1_miles,
    team2_miles,
    venue_city,
    venue_state,
    airport_distance
):
    score = 50  # baseline

    # ---- 1. Airport Access (Major Hub) ----
    major_airports = [
        "los angeles", "phoenix", "las vegas", "atlanta", "chicago",
        "dallas", "houston", "orlando", "miami", "charlotte", "seattle",
        "denver", "new york", "boston", "philadelphia"
    ]
    city_key = f"{venue_city}, {venue_state}".lower()

    if any(a in city_key for a in major_airports):
        score += 12
    else:
        score += 5

    # ---- 1B. Airport Distance to Venue ----
    if airport_distance < 10:
        score += 15
    elif airport_distance < 20:
        score += 10
    elif airport_distance < 35:
        score += 6
    elif airport_distance < 50:
        score += 3
    else:
        score += 1

    # ---- 2. Driving Access ----
    avg_miles = (team1_miles + team2_miles) / 2
    if avg_miles < 250:
        score += 20
    elif avg_miles < 600:
        score += 12
    elif avg_miles < 1000:
        score += 6
    else:
        score += 2

    # ---- 3. Urban Access Level ----
    high_access_cities = [
        "los angeles", "tampa", "orlando", "phoenix", "new york",
        "boston", "denver", "atlanta", "san antonio", "charlotte"
    ]
    if venue_city.lower() in high_access_cities:
        score += 15
    else:
        score += 5

    # ---- 4. Stadium Capacity Fit ----
    if venue_capacity >= 70000:
        score -= 5
    elif venue_capacity <= 35000:
        score += 5

    return max(0, min(100, score))


def safe_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def compute_final_attendance(raw_pred, bowl_name, bowl_avg_att, venue_capacity):
    """
    Reproduce the Excel / Google Sheets logic used for the 2025
    'Final Attendance Prediction' column.
    """
    bowl_lower = str(bowl_name).lower()

    # 1) 70/30 blended prediction
    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att

    # 2) Hawaii & Bahamas use raw model prediction
    if ("hawai" in bowl_lower) or ("bahamas" in bowl_lower):
        base = raw_pred
    else:
        base = blended_pred

    # 3) Xbox Bowl manual cap at 6,500
    if "xbox" in bowl_lower:
        base = min(base, 6500.0)

    # 4) Bowl-specific boosts
    boosts = [
        ("gator", 1.05),
        ("pop-tart", 1.05),
        ("pop tarts", 1.05),
        ("texas bowl", 1.05),
        ("music city", 1.05),
        ("alamo", 1.10),
        ("duke’s mayo", 1.03),
        ("duke's mayo", 1.03),
    ]
    for key, factor in boosts:
        if key in bowl_lower:
            base *= factor
            break

    # 5) Capacity cap + floor
    final_pred = min(base, venue_capacity)
    final_pred = max(final_pred, 0.0)

    return final_pred


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
calib = pd.read_csv(CALIB_FILE)

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
            local_rows.loc[
                local_rows["Local Team Indicator"] == 1, "Distance Minimum"
            ].max()
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
# BOWL & VENUE SELECTION
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
B10_present = int("Big Ten Conference" in (conf1, conf2))
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
    minmax_distance = (avg_distance - dist_min_train) / (
        dist_max_train - dist_min_train
    )
    minmax_distance = max(0.0, min(1.0, minmax_distance))
else:
    minmax_distance = 0.0

if ap_max_train > ap_min_train:
    ap_strength_norm = (ap_strength_score - ap_min_train) / (
        ap_max_train - ap_min_train
    )
    ap_strength_norm = max(0.0, min(1.0, ap_strength_norm))
else:
    ap_strength_norm = 0.0

# =====================================================
# TRAVEL MAP + ROUTE LINES
# =====================================================

st.subheader("🗺️ Travel Routes & Estimated Travel Time")

def estimate_driving_hours(miles):
    if miles <= 0:
        return None
    if miles < 300:
        return miles / 55
    elif miles < 800:
        return miles / 62
    else:
        return (miles / 65) + 10  # adds overnight stop


def estimate_flying_hours(miles):
    if miles <= 0:
        return None
    return (miles / 250) + 1.7  # gate-to-gate + airport overhead


def fmt(hours):
    return f"{hours:.1f} hrs" if hours is not None else "N/A"


t1_drive_hours = estimate_driving_hours(team1_miles)
t2_drive_hours = estimate_driving_hours(team2_miles)
t1_flight_hours = estimate_flying_hours(team1_miles)
t2_flight_hours = estimate_flying_hours(team2_miles)

colA, colB = st.columns(2)
with colA:
    st.markdown(f"**{team1} → {venue_choice}**")
    st.write(f"Driving Time: {fmt(t1_drive_hours)}")
    st.write(f"Flying Time: {fmt(t1_flight_hours)}")

with colB:
    st.markdown(f"**{team2} → {venue_choice}**")
    st.write(f"Driving Time: {fmt(t2_drive_hours)}")
    st.write(f"Flying Time: {fmt(t2_flight_hours)}")

line_data = [
    {
        "from_lon": t1_lon,
        "from_lat": t1_lat,
        "to_lon": venue_lon,
        "to_lat": venue_lat,
        "name": team1,
        "miles": team1_miles,
        "hours": estimate_travel_hours(team1_miles),
    },
    {
        "from_lon": t2_lon,
        "from_lat": t2_lat,
        "to_lon": venue_lon,
        "to_lat": venue_lat,
        "name": team2,
        "miles": team2_miles,
        "hours": estimate_travel_hours(team2_miles),
    },
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
    columns=feature_cols,
)

# =====================================================
# PREDICTION + ALL SCORECARD PANELS
# =====================================================

st.header("Prediction")

if st.button("Run Prediction"):

    # -----------------------------
    # Core attendance prediction
    # -----------------------------
    raw_pred = float(model.predict(feature_row)[0])

    final_pred = compute_final_attendance(
        raw_pred=raw_pred,
        bowl_name=bowl_choice,
        bowl_avg_att=bowl_avg_att,
        venue_capacity=venue_capacity,
    )

    pct_filled = final_pred / venue_capacity if venue_capacity > 0 else 0.0

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
    # 📍 VENUE ACCESSIBILITY SCORE
    # =====================================================

    nearest_airport_name, airport_distance = find_nearest_airport(
        venue_lat, venue_lon
    )

    venue_access_score = compute_venue_accessibility_score(
        venue_capacity,
        team1_miles,
        team2_miles,
        venue_city,
        venue_state,
        airport_distance,
    )

    st.subheader("📍 Venue Accessibility Score")
    st.metric("Accessibility Score", f"{venue_access_score}/100")
    st.write("### Key Accessibility Factors")
    st.write(f"- Nearest airport: **{nearest_airport_name}**")
    st.write(f"- Distance from venue to airport: **{airport_distance:.1f} miles**")
    st.write(
        f"- Avg team distance traveled: **{(team1_miles + team2_miles)/2:.0f} miles**"
    )
    st.write(f"- Stadium capacity: **{venue_capacity:,} seats**")
    st.write(f"- City: **{venue_city}, {venue_state}**")

    if venue_access_score >= 80:
        st.success("This venue offers excellent accessibility for fans and travel logistics.")
    elif venue_access_score >= 65:
        st.info("This venue has strong overall accessibility.")
    elif venue_access_score >= 50:
        st.warning("This venue has moderate accessibility — travel may require planning.")
    else:
        st.error("This venue is challenging for most fans to reach.")

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
        conference,
    ):
        score = 50

        if miles <= 150:
            score += 20
        elif miles <= 350:
            score += 12
        elif miles <= 700:
            score += 4
        else:
            score -= 10

        if fanbase_size > 600000:
            score += 10
        elif fanbase_size > 300000:
            score += 6
        elif fanbase_size > 100000:
            score += 3

        if brand_power > 75:
            score += 8
        elif brand_power > 50:
            score += 4

        if wins >= 9:
            score += 6
        elif wins >= 7:
            score += 3

        if ap_strength > 0 and ap_strength <= 20:
            score += 4
        elif ap_strength <= 35:
            score += 2

        ad = str(alumni_dispersion).lower()
        if "national" in ad:
            score += 5
        elif "regional" in ad:
            score += 2
        elif "local" in ad:
            score -= 2

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

    t1_tvi = compute_tvi(
        miles=team1_miles,
        fanbase_size=t1_fanbase,
        brand_power=t1_brand,
        wins=t1_wins,
        ap_strength=t1_ap_strength,
        alumni_dispersion=row1.get("Alumni Dispersion", ""),
        conference=row1.get("Football FBS Conference", ""),
    )

    t2_tvi = compute_tvi(
        miles=team2_miles,
        fanbase_size=t2_fanbase,
        brand_power=t2_brand,
        wins=t2_wins,
        ap_strength=t2_ap_strength,
        alumni_dispersion=row2.get("Alumni Dispersion", ""),
        conference=row2.get("Football FBS Conference", ""),
    )

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.metric(f"{team1} Visitation Index", f"{t1_tvi} / 100")
    with col_t2:
        st.metric(f"{team2} Visitation Index", f"{t2_tvi} / 100")

    avg_tvi = (t1_tvi + t2_tvi) / 2
    st.write(f"**Overall Matchup Travel Expectation:** {avg_tvi:.1f} / 100")

    # =====================================================
    # MARKET IMPACT SCORE (MIS)
    # =====================================================

    st.subheader("🏙️ Market Impact Score")

    if avg_distance < 150:
        travel_pct = 0.25
    elif avg_distance < 300:
        travel_pct = 0.40
    elif avg_distance < 700:
        travel_pct = 0.55
    else:
        travel_pct = 0.70

    travel_pct *= (avg_tvi / 100)

    if local_flag == 1:
        travel_pct *= 0.8

    travel_pct = max(0.05, min(0.85, travel_pct))

    visitor_attendance = final_pred * travel_pct
    avg_nights = 1.7
    visitor_nights = visitor_attendance * avg_nights

    mis = 50
    if t1_brand > 70 or t2_brand > 70:
        mis += 10
    elif t1_brand > 50 or t2_brand > 50:
        mis += 5

    if matchup_power == 2:
        mis += 10
    elif matchup_power == 1:
        mis += 4

    if avg_distance > 900:
        mis += 12
    elif avg_distance > 600:
        mis += 6
    else:
        mis += 3

    if bowl_tier >= 2:
        mis += 7

    mis = max(0, min(100, mis))

    st.metric("Market Impact Score", f"{mis} / 100")
    st.write(f"Estimated Traveling Fans: **{visitor_attendance:,.0f}**")
    st.write(f"Estimated Visitor Nights: **{visitor_nights:,.0f}**")

    if mis >= 85:
        st.write("Projected Market Impact: **Very High**")
    elif mis >= 70:
        st.write("Projected Market Impact: **High**")
    elif mis >= 55:
        st.write("Projected Market Impact: **Moderate**")
    else:
        st.write("Projected Market Impact: **Limited**")

    # =====================================================
    # SPONSOR VISIBILITY SCORE (SVS)
    # =====================================================

    st.subheader("📢 Sponsor Visibility Score")

    svs = 50

    if bowl_tier >= 3:
        svs += 15
    elif bowl_tier == 2:
        svs += 8
    else:
        svs += 3

    if combined_fanbase_log > 6.2:
        svs += 15
    elif combined_fanbase_log > 5.7:
        svs += 10
    elif combined_fanbase_log > 5.3:
        svs += 5

    avg_brand = (t1_brand + t2_brand) / 2
    if avg_brand > 75:
        svs += 10
    elif avg_brand > 55:
        svs += 6
    elif avg_brand > 40:
        svs += 3

    if matchup_power == 2:
        svs += 12
    elif matchup_power == 1:
        svs += 6
    else:
        svs += 2

    avg_ap_strength = (t1_ap_strength + t2_ap_strength) / 2
    combined_wins = t1_wins + t2_wins

    if combined_wins >= 17:
        svs += 8
    elif combined_wins >= 14:
        svs += 5
    else:
        svs += 2

    if avg_ap_strength <= 20 and avg_ap_strength > 0:
        svs += 7
    elif avg_ap_strength <= 35:
        svs += 4

    if final_pred > 60000:
        svs += 12
    elif final_pred > 40000:
        svs += 7
    elif final_pred > 25000:
        svs += 4
    else:
        svs += 1

    if avg_tvi > 75:
        svs += 6
    elif avg_tvi > 55:
        svs += 4
    else:
        svs += 1

    svs = max(0, min(100, svs))

    st.metric("Sponsor Visibility Score", f"{svs} / 100")

    if svs >= 85:
        st.write("Exposure Classification: **Premium Event for Sponsors**")
    elif svs >= 70:
        st.write("Exposure Classification: **High-Value Matchup**")
    elif svs >= 55:
        st.write("Exposure Classification: **Moderate Visibility**")
    else:
        st.write("Exposure Classification: **Low to Medium Visibility**")

    st.write("Key Visibility Drivers:")
    if avg_brand > 60:
        st.write("- Strong team brand presence")
    if combined_fanbase_log > 5.7:
        st.write("- Large combined fanbase reach")
    if final_pred > 45000:
        st.write("- High in-stadium exposure expected")
    if matchup_power == 2:
        st.write("- Premium P4 vs. P4 matchup")
    if avg_tvi > 60:
        st.write("- Highly engaged, traveling fanbases")

    # =====================================================
    # CROWD COMPOSITION MODEL
    # =====================================================

    st.subheader("🧍‍♂️ Crowd Composition Breakdown")

    traveling_fans = travel_pct

    alumni_factor = 0.0
    if "local" in str(row1.get("Alumni Dispersion", "")).lower():
        alumni_factor += 0.15
    if "local" in str(row2.get("Alumni Dispersion", "")).lower():
        alumni_factor += 0.15
    if "regional" in str(row1.get("Alumni Dispersion", "")).lower():
        alumni_factor += 0.08
    if "regional" in str(row2.get("Alumni Dispersion", "")).lower():
        alumni_factor += 0.08
    if "national" in str(row1.get("Alumni Dispersion", "")).lower():
        alumni_factor += 0.03
    if "national" in str(row2.get("Alumni Dispersion", "")).lower():
        alumni_factor += 0.03

    alumni_factor = min(alumni_factor, 0.35)

    student_factor = 0.02
    if team1_miles < 150:
        student_factor += 0.03
    if team2_miles < 150:
        student_factor += 0.03

    general_public = 1.0 - (traveling_fans + alumni_factor + student_factor)
    general_public = max(0.05, general_public)

    total_share = traveling_fans + alumni_factor + student_factor + general_public
    traveling_fans /= total_share
    alumni_factor /= total_share
    student_factor /= total_share
    general_public /= total_share

    st.markdown(f"**Traveling Fans:** {traveling_fans*100:.1f}%")
    st.markdown(f"**Local Alumni:** {alumni_factor*100:.1f}%")
    st.markdown(f"**Students:** {student_factor*100:.1f}%")
    st.markdown(f"**General Public / Neutral Fans:** {general_public*100:.1f}%")

    # =====================================================
    # MATCHUP FIT SCORE (0–100)
    # =====================================================

    st.subheader("🔗 Matchup Fit Score")

    mfs = 50

    if distance_min <= 200:
        mfs += 12
    elif distance_min <= 400:
        mfs += 8
    elif distance_min <= 800:
        mfs += 4

    t1_state = str(row1.get("State", "")).lower()
    t2_state = str(row2.get("State", "")).lower()
    venue_state_lower = str(venue_row.get("State", "")).lower()

    if t1_state == venue_state_lower:
        mfs += 10
    if t2_state == venue_state_lower:
        mfs += 10

    if t1_state in ["tx", "ok", "la", "ar", "nm"] and t2_state in [
        "tx",
        "ok",
        "la",
        "ar",
        "nm",
    ]:
        mfs += 8
    if t1_state in ["fl", "ga", "sc", "nc", "al"] and t2_state in [
        "fl",
        "ga",
        "sc",
        "nc",
        "al",
    ]:
        mfs += 8
    if t1_state in ["ca", "az", "nv", "or", "wa"] and t2_state in [
        "ca",
        "az",
        "nv",
        "or",
        "wa",
    ]:
        mfs += 8

    conf1_lower = str(row1.get("Football FBS Conference", "")).lower()
    conf2_lower = str(row2.get("Football FBS Conference", "")).lower()

    if conf1_lower == conf2_lower:
        mfs += 6

    if matchup_power == 2:
        mfs += 12
    elif matchup_power == 1:
        mfs += 6
    else:
        mfs += 2

    alumni1 = str(row1.get("Alumni Dispersion", "")).lower()
    alumni2 = str(row2.get("Alumni Dispersion", "")).lower()

    if "local" in alumni1:
        mfs += 6
    if "local" in alumni2:
        mfs += 6
    if "regional" in alumni1:
        mfs += 4
    if "regional" in alumni2:
        mfs += 4
    if "national" in alumni1:
        mfs += 3
    if "national" in alumni2:
        mfs += 3

    if avg_brand > 75:
        mfs += 12
    elif avg_brand > 55:
        mfs += 8
    elif avg_brand > 40:
        mfs += 4

    big_name_list = ["texas", "michigan", "lsu", "ohio", "alabama", "georgia", "notre dame"]
    if any(b in row1.get("Team Name", "").lower() for b in big_name_list):
        mfs += 6
    if any(b in row2.get("Team Name", "").lower() for b in big_name_list):
        mfs += 6

    ap_diff = abs(t1_ap_strength - t2_ap_strength)
    talent_diff = abs(t1_talent - t2_talent)

    if ap_diff <= 5:
        mfs += 7
    elif ap_diff <= 12:
        mfs += 4

    if talent_diff <= 30:
        mfs += 6
    elif talent_diff <= 70:
        mfs += 3

    mfs = max(0, min(100, mfs))

    st.metric("Matchup Fit Score", f"{mfs} / 100")

    if mfs >= 85:
        st.write("Fit Category: **Excellent Regional or Conference Matchup**")
    elif mfs >= 70:
        st.write("Fit Category: **Strong Overall Bowl Fit**")
    elif mfs >= 55:
        st.write("Fit Category: **Moderate Fit**")
    else:
        st.write("Fit Category: **Limited Natural Fit**")

    st.write("Key Fit Drivers:")
    if distance_min < 300:
        st.write("- Teams are geographically well-aligned for travel")
    if conf1_lower == conf2_lower:
        st.write("- Conference familiarity enhances matchup relevance")
    if avg_brand > 50:
        st.write("- Strong brand presence supports bowl resonance")
    if "local" in alumni1 or "local" in alumni2:
        st.write("- Significant local alumni presence boosts turnout")
    if ap_diff < 10:
        st.write("- Competitive balance improves matchup interest")

    # =====================================================
    # CONFIDENCE INTERVALS + MODEL STABILITY + RISK
    # =====================================================

    st.subheader("Uncertainty & Risk")

    ci_width = 0.08
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

    if (
        calib_combined_log_median is not None
        and combined_fanbase_log < calib_combined_log_median
    ):
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
    feat_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": importances,
            "Value": feature_row.iloc[0].values,
        }
    )
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
        drivers.append(
            "At least one team is very close to the venue (<100 miles), "
            "which strongly boosts attendance due to drive-in ease."
        )
    elif distance_min < 250:
        drivers.append(
            "A nearby team (<250 miles) provides a strong regional attendance lift."
        )
    elif distance_min < 500:
        drivers.append(
            "Both teams are within reasonable travel distance, supporting moderate fan travel."
        )
    else:
        drivers.append(
            "Both teams face longer travel distances, which historically moderates attendance."
        )

    if distance_imbalance > 400:
        drivers.append(
            "There is a major travel imbalance between fanbases, meaning one school may dominate in-person turnout."
        )

    brands_present = []
    for t in [team1, team2]:
        for brand in big_brands:
            if brand.lower() in t.lower():
                brands_present.append(brand)
    if brands_present:
        bp_str = ", ".join(sorted(set(brands_present)))
        drivers.append(
            f"National brands present ({bp_str}). These programs traditionally travel well and elevate overall demand."
        )

    if calib_combined_log_median is not None:
        if combined_fanbase_log > calib_combined_log_median + 0.2:
            drivers.append(
                "The combined fanbase size is significantly larger than typical FBS matchups, "
                "supporting stronger turnout."
            )
        elif combined_fanbase_log < calib_combined_log_median - 0.2:
            drivers.append(
                "This matchup features smaller fanbases, which historically leads to more modest attendance figures."
            )

    if matchup_power == 2:
        drivers.append(
            "Power vs. Power matchup substantially increases national interest and in-person attendance."
        )
    elif matchup_power == 1:
        drivers.append(
            "Having at least one Power Conference team helps boost the appeal of this bowl."
        )
    else:
        drivers.append(
            "Group of Five vs Group of Five matchups typically rely more on regional proximity for attendance strength."
        )

    if SEC_present:
        drivers.append(
            "An SEC program is participating — SEC teams historically generate high demand and strong travel behavior."
        )
    if B10_present:
        drivers.append(
            "A Big Ten program is participating — their fans traditionally travel well and lift bowl attendance."
        )
    if ACC_present:
        drivers.append("An ACC team contributes added brand visibility.")
    if B12_present:
        drivers.append("A Big 12 program adds solid regional and national interest.")

    if venue_tier >= 2:
        drivers.append(
            "This game is hosted in a high-tier venue, enhancing the bowl experience and spectator draw."
        )

    def alumni_comment(team, dispersion, distance):
        msg = None
        dispersion = dispersion.lower()
        if "local" in dispersion:
            if distance < local_threshold:
                msg = (
                    f"{team} has a strong local alumni base near the bowl site, "
                    "which should meaningfully lift attendance."
                )
            else:
                msg = (
                    f"{team} typically relies on local alumni turnout, "
                    "but this bowl is farther from its primary alumni region."
                )
        elif "regional" in dispersion:
            if distance < 400:
                msg = (
                    f"{team}'s alumni network is regionally concentrated, "
                    "and the relatively close travel distance should support strong turnout."
                )
            else:
                msg = (
                    f"{team} has a regionally distributed alumni base, "
                    "but the travel distance may moderate participation."
                )
        elif "national" in dispersion:
            if distance > 700:
                msg = (
                    f"{team} has a nationally dispersed alumni base, "
                    "which helps offset the longer travel distance to the bowl."
                )
            else:
                msg = (
                    f"{team}'s national alumni footprint supports flexible travel and broader turnout potential."
                )
        return msg

    t1_alumni = str(row1.get("Alumni Dispersion", "")).lower()
    t2_alumni = str(row2.get("Alumni Dispersion", "")).lower()

    alumni_msgs = [
        alumni_comment(team1, t1_alumni, team1_miles),
        alumni_comment(team2, t2_alumni, team2_miles),
    ]
    for m in alumni_msgs:
        if m:
            drivers.append(m)

    if local_flag == 1:
        drivers.append(
            "The presence of a local or near-local team provides a significant attendance boost."
        )

    if not drivers:
        drivers.append(
            "This matchup aligns with typical bowl attendance patterns in our model."
        )

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
# 📚 HISTORICAL COMPARISON: SAME BOWL (2022, 2023, 2024)
# =====================================================

st.subheader("📚 Historical Comparison (Same Bowl: 2022–2024)")

# Filter to same bowl name across historical seasons
hist_bowl = calib[calib["Bowl Game Name"] == bowl_choice].copy()

# Ensure we only compare 2022, 2023, 2024 (no projections)
if "Year" in hist_bowl.columns:
    hist_bowl = hist_bowl[hist_bowl["Year"].isin([2022, 2023, 2024])]

# Convert numeric columns
for col in ["Attendance", "Venue Capacity", "Avg Distace Traveled",
            "Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)",
            "Combined Fanbase (Log transformed)", "AP Strength Score",
            "Distance Minimum", "Distance Imbalance"]:
    if col in hist_bowl.columns:
        hist_bowl[col] = pd.to_numeric(hist_bowl[col], errors="coerce")

# Build comparison table
comparison_rows = []

for _, row in hist_bowl.iterrows():
    year = int(row.get("Year", 0))
    attendance = safe_num(row.get("Attendance"))
    cap = safe_num(row.get("Venue Capacity"))
    pct_filled_hist = attendance / cap if cap > 0 else 0

    comparison_rows.append({
        "Year": year,
        "Matchup": f"{row.get('Team 1', '')} vs {row.get('Team 2', '')}",
        "Actual Attendance": f"{attendance:,.0f}",
        "% Filled": f"{pct_filled_hist:.1%}",
        "Avg Distance (mi)": f"{safe_num(row.get('Avg Distace Traveled')):,.0f}",
        "Min Distance": f"{safe_num(row.get('Distance Minimum')):,.0f}",
        "Distance Imbalance": f"{safe_num(row.get('Distance Imbalance')):,.0f}",
        "Matchup Power": safe_num(row.get("Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)")),
        "Combined Fanbase (log)": f"{safe_num(row.get('Combined Fanbase (Log transformed)')):.2f}",
        "AP Strength": f"{safe_num(row.get('AP Strength Score')):.1f}",
    })

# Add the 2025 projection row
comparison_rows.append({
    "Year": 2025,
    "Matchup": f"{team1} vs {team2}",
    "Actual Attendance": f"{final_pred:,.0f}",
    "% Filled": f"{pct_filled:.1%}",
    "Avg Distance (mi)": f"{avg_distance:,.0f}",
    "Min Distance": f"{distance_min:,.0f}",
    "Distance Imbalance": f"{distance_imbalance:,.0f}",
    "Matchup Power": matchup_power,
    "Combined Fanbase (log)": f"{combined_fanbase_log:.2f}",
    "AP Strength": f"{ap_strength_score:.1f}",
})

hist_df = pd.DataFrame(comparison_rows)

st.dataframe(hist_df, use_container_width=True)

# Trend Summary
if len(hist_bowl) > 0:
    hist_att_avg = hist_bowl["Attendance"].mean()
    delta_vs_hist = final_pred - hist_att_avg
    pct_delta_hist = delta_vs_hist / hist_att_avg if hist_att_avg > 0 else 0

    st.markdown("### 📈 Trend Summary")
    st.write(f"- **Historical Avg Attendance (2022–24):** {hist_att_avg:,.0f}")
    st.write(f"- **2025 Projection:** {final_pred:,.0f}")
    st.write(f"- **Difference:** {delta_vs_hist:,.0f} (**{pct_delta_hist:.1%}**)")

    if final_pred > hist_att_avg:
        st.success("2025 projection exceeds the bowl's recent 3-year average.")
    else:
        st.warning("2025 projection is below the bowl's recent 3-year average.")
else:
    st.write("No valid historical bowl data found for this matchup.")


# =====================================================
# VENUE SCENARIO: MOVE BOWL TO ANOTHER STADIUM
# =====================================================

st.header("📍 Venue Scenario: Move This Bowl to Another Stadium")

alt_venue_choice = st.selectbox(
    "Select an alternative venue",
    sorted(venues[VENUE_COL].unique()),
    index=sorted(venues[VENUE_COL].unique()).index(venue_choice)
    if venue_choice in venues[VENUE_COL].values
    else 0,
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
    alt_row_data.update(
        {
            "Venue Tier": alt_venue_tier,
            "Team 1 Miles to Venue": alt_t1_miles,
            "Team 2 Miles to Venue": alt_t2_miles,
            "Avg Distace Traveled": alt_avg_distance,
            "Distance Minimum": alt_distance_min,
            "Distance Imbalance": alt_distance_imbalance,
            "Local Team Indicator": alt_local_flag,
            "MinMax Scale Distcance": (
                (alt_avg_distance - dist_min_train)
                / (dist_max_train - dist_min_train)
                if dist_max_train > dist_min_train
                else 0.0
            ),
        }
    )
    alt_row_data["MinMax Scale Distcance"] = max(
        0.0, min(1.0, alt_row_data["MinMax Scale Distcance"])
    )

    alt_feature_row = pd.DataFrame(
        [[alt_row_data.get(c, 0.0) for c in feature_cols]],
        columns=feature_cols,
    )

    alt_raw = float(model.predict(alt_feature_row)[0])

    alt_final = compute_final_attendance(
        raw_pred=alt_raw,
        bowl_name=bowl_choice,
        bowl_avg_att=bowl_avg_att,
        venue_capacity=alt_capacity,
    )

    alt_pct_filled = alt_final / alt_capacity if alt_capacity > 0 else 0.0

    st.write(f"**Alternative Venue:** {alt_venue_choice}")
    st.write(f"Predicted Attendance: **{alt_final:,.0f}**")
    st.write(
        f"Capacity: {alt_capacity:,.0f} | Projected % Filled: {alt_pct_filled:.1%}"
    )

# =====================================================
# TEAM & MATCHUP SCORECARDS
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
    st.write(
        f"**Confidence Interval:** {ctx['ci_low']:,.0f} – {ctx['ci_high']:,.0f}"
    )
    st.write(f"**Model Stability Score:** {ctx['stability_score']:.1f} / 10")
    st.write(f"**Travel Propensity Score:** {ctx['travel_score']:.1f} / 10")
    st.write(f"**Sellout Probability:** {ctx['sellout_prob']*100:.1f}%")
    st.write(f"**Head-to-Head Interest Index:** {ctx['interest_index']:.1f} / 10")
    st.write(
        f"**Matchup Power Score:** {matchup_power} (2 = P4vP4, 1 = P4vG5, 0 = G5vG5)"
    )
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
            existing = pd.concat(
                [existing, pd.DataFrame([scenario_row])], ignore_index=True
            )
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


