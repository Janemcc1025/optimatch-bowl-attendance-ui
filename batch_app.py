import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# CONFIG / CONSTANTS
# =====================================================

PRED_YEAR = 2025

TEAM_FILE = "2025 Bowl Games - UI Team Lookup.csv"
VENUE_FILE = "2025 Bowl Games - UI Venue Lookup.csv"
BOWL_TIERS_FILE = "2025 Bowl Games - Bowl Tiers.csv"
CALIB_FILE = "2025 Bowl Games - 2022-2024 Bowl Games (7).csv"

MODEL_FILE = "attendance_model.pkl"
FEATURE_COLS_FILE = "feature_columns.pkl"

TEAM_COL = "Team Name"
VENUE_COL = "Football Stadium"
BOWL_NAME_COL = "Bowl Name"

# =====================================================
# HELPER FUNCTIONS
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
    "SLC - Salt Lake City": (40.7899, -111.9791),
}


def safe_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


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
    airport_distance,
):
    score = 50  # baseline

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

    avg_miles = (team1_miles + team2_miles) / 2
    if avg_miles < 250:
        score += 20
    elif avg_miles < 600:
        score += 12
    elif avg_miles < 1000:
        score += 6
    else:
        score += 2

    high_access_cities = [
        "los angeles", "tampa", "orlando", "phoenix", "new york",
        "boston", "denver", "atlanta", "san antonio", "charlotte"
    ]
    if venue_city.lower() in high_access_cities:
        score += 15
    else:
        score += 5

    if venue_capacity >= 70000:
        score -= 5
    elif venue_capacity <= 35000:
        score += 5

    return max(0, min(100, score))


def compute_final_attendance(raw_pred, bowl_name, bowl_avg_att, venue_capacity):
    """Match the Excel/Sheets logic: 70/30 blend + special bowls + boosts + capacity cap."""
    bowl_lower = str(bowl_name).lower()

    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att

    # Raw-only bowls
    if ("hawai" in bowl_lower) or ("bahamas" in bowl_lower):
        base = raw_pred
    else:
        base = blended_pred

    # Xbox cap
    if "xbox" in bowl_lower:
        base = min(base, 6500.0)

    # Boosts
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

    final_pred = min(base, venue_capacity)
    final_pred = max(final_pred, 0.0)
    return final_pred


def compute_tvi(
    miles,
    fanbase_size,
    brand_power,
    wins,
    ap_strength,
    alumni_dispersion,
    conference,
):
    """Team Visitation Index 0–100 (same heuristic as single-game app)."""
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


# =====================================================
# LOAD MODEL & DATA
# =====================================================

model = joblib.load(MODEL_FILE)
feature_cols = joblib.load(FEATURE_COLS_FILE)

teams = pd.read_csv(TEAM_FILE)
venues = pd.read_csv(VENUE_FILE)
bowl_tiers = pd.read_csv(BOWL_TIERS_FILE)
calib = pd.read_csv(CALIB_FILE)

for df in (teams, venues, bowl_tiers, calib):
    df.columns = df.columns.str.strip()

# =====================================================
# CALIBRATION FOR DISTANCE/AP NORMALIZATION
# =====================================================

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

try:
    calib_combined_log_median = pd.to_numeric(
        calib["Combined Fanbase (Log transformed)"], errors="coerce"
    ).median()
except Exception:
    calib_combined_log_median = None

# =====================================================
# FEATURE ENGINEERING FOR ONE MATCHUP
# =====================================================

def build_features_for_matchup(row):
    """
    row must contain: Team 1, Team 2, Venue, Bowl Game, Date
    Returns (features_dict, metrics_dict) or (None, {"error": ...})
    """
    team1_name = str(row.get("Team 1", "")).strip()
    team2_name = str(row.get("Team 2", "")).strip()
    venue_name = str(row.get("Venue", "")).strip()
    bowl_name = str(row.get("Bowl Game", "")).strip()
    date_val = row.get("Date", "")

    # Lookups
    t1_rows = teams[teams[TEAM_COL] == team1_name]
    t2_rows = teams[teams[TEAM_COL] == team2_name]
    v_rows = venues[venues[VENUE_COL] == venue_name]
    b_rows = bowl_tiers[bowl_tiers[BOWL_NAME_COL] == bowl_name]

    if t1_rows.empty or t2_rows.empty or v_rows.empty or b_rows.empty:
        return None, {"error": "Missing lookup for team or venue or bowl."}

    row1 = t1_rows.iloc[0]
    row2 = t2_rows.iloc[0]
    vrow = v_rows.iloc[0]
    brow = b_rows.iloc[0]

    # Weekend flag from date (Sat/Sun = weekend)
    try:
        dt = pd.to_datetime(date_val)
        weekend_flag = 1 if dt.weekday() >= 5 else 0
    except Exception:
        weekend_flag = 0

    # Basic bowl attributes
    venue_capacity = safe_num(vrow.get("Football Capacity"))
    venue_lat = safe_num(vrow.get("Lat"))
    venue_lon = safe_num(vrow.get("Lon"))
    venue_city = vrow.get("City", "")
    venue_state = vrow.get("State", "")

    # From bowl_tiers
    bowl_tier = safe_num(brow.get("Tier", 1.0))
    bowl_owner = safe_num(brow.get("Ownership", 0.0))
    bowl_avg_viewers = safe_num(brow.get("Avg Viewers", 0.0))
    bowl_avg_att = safe_num(brow.get("Avg Attendance", 0.0))

    # If bowl avg attendees not present, fallback to calib
    if bowl_avg_att == 0.0 and "Bowl Avg Attendees" in calib.columns:
        cb = calib[calib["Bowl Game Name"] == bowl_name]
        if not cb.empty:
            bowl_avg_att = safe_num(cb["Bowl Avg Attendees"].iloc[0])
        else:
            bowl_avg_att = safe_num(calib["Bowl Avg Attendees"].median())

    # Venue tier from calibration if present
    calib_for_bowl = calib[calib["Bowl Game Name"] == bowl_name]
    if "Venue Tier" in calib.columns and not calib_for_bowl.empty:
        venue_tier = safe_num(calib_for_bowl["Venue Tier"].iloc[0])
    else:
        venue_tier = 1.0

    # Team season columns
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
        minmax_distance = (avg_distance - dist_min_train) / (dist_max_train - dist_min_train)
        minmax_distance = max(0.0, min(1.0, minmax_distance))
    else:
        minmax_distance = 0.0

    if ap_max_train > ap_min_train:
        ap_strength_norm = (ap_strength_score - ap_min_train) / (ap_max_train - ap_min_train)
        ap_strength_norm = max(0.0, min(1.0, ap_strength_norm))
    else:
        ap_strength_norm = 0.0

    # Build feature dict
    features = {
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

    # ---------------- PREDICT ATTENDANCE ----------------
    feature_row = pd.DataFrame([[features.get(c, 0.0) for c in feature_cols]],
                               columns=feature_cols)
    raw_pred = float(model.predict(feature_row)[0])
    final_pred = compute_final_attendance(raw_pred, bowl_name, bowl_avg_att, venue_capacity)
    pct_filled = final_pred / venue_capacity if venue_capacity > 0 else 0.0

    # ---------------- SECONDARY METRICS -----------------

    # Venue accessibility
    _, airport_dist = find_nearest_airport(venue_lat, venue_lon)
    venue_access_score = compute_venue_accessibility_score(
        venue_capacity, team1_miles, team2_miles, venue_city, venue_state, airport_dist
    )

    alumni1 = row1.get("Alumni Dispersion", "")
    alumni2 = row2.get("Alumni Dispersion", "")

    t1_tvi = compute_tvi(
        team1_miles, t1_fanbase, t1_brand, t1_wins, t1_ap_strength, alumni1, conf1
    )
    t2_tvi = compute_tvi(
        team2_miles, t2_fanbase, t2_brand, t2_wins, t2_ap_strength, alumni2, conf2
    )
    avg_tvi = (t1_tvi + t2_tvi) / 2.0

    # Market Impact Score
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

    # Sponsor Visibility Score
    svs = 50
    avg_brand = (t1_brand + t2_brand) / 2
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

    # Matchup Fit Score (condensed)
    mfs = 50
    if distance_min <= 200:
        mfs += 12
    elif distance_min <= 400:
        mfs += 8
    elif distance_min <= 800:
        mfs += 4
    if matchup_power == 2:
        mfs += 12
    elif matchup_power == 1:
        mfs += 6
    else:
        mfs += 2
    if avg_brand > 75:
        mfs += 12
    elif avg_brand > 55:
        mfs += 8
    elif avg_brand > 40:
        mfs += 4
    mfs = max(0, min(100, mfs))

    # Interest index (simple)
    interest_index = 5.0
    big_brands = [
        "Michigan", "Ohio State", "Texas", "LSU", "Alabama", "Georgia",
        "Notre Dame", "USC", "Oklahoma", "Penn State", "Oregon",
        "Florida State", "Clemson", "Tennessee", "Auburn"
    ]
    brand_count = sum(any(b.lower() in t.lower() for b in big_brands)
                      for t in [team1_name, team2_name])
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
    interest_index = max(1.0, min(10.0, interest_index))

    # Sellout probability
    fill_ratio = pct_filled
    if bowl_tier >= 2:
        fill_ratio += 0.05
    if matchup_power == 2:
        fill_ratio += 0.05
    if calib_combined_log_median is not None:
        if combined_fanbase_log > calib_combined_log_median + 0.2:
            fill_ratio += 0.03
        elif combined_fanbase_log < calib_combined_log_median - 0.2:
            fill_ratio -= 0.03
    sellout_prob = float(np.clip(fill_ratio, 0.0, 1.0))

    metrics = {
        "Predicted Attendance": final_pred,
        "Raw Pred": raw_pred,
        "% Filled": pct_filled,
        "Venue Capacity": venue_capacity,
        "Venue Accessibility": venue_access_score,
        "TVI Avg": avg_tvi,
        "MIS": mis,
        "SVS": svs,
        "MFS": mfs,
        "Interest Index": interest_index,
        "Sellout Prob": sellout_prob,
        "Team 1 Miles": team1_miles,
        "Team 2 Miles": team2_miles,
        "Avg Distance": avg_distance,
        "Distance Min": distance_min,
        "Distance Imbalance": distance_imbalance,
        "Matchup Power": matchup_power,
        "Combined Fanbase Log": combined_fanbase_log,
        "AP Strength Score": ap_strength_score,
    }

    return features, metrics


# =====================================================
# STREAMLIT UI - BATCH APP
# =====================================================

st.set_page_config(page_title="Optimatch Batch Bowl Attendance Predictions", layout="wide")

st.title("🏈 Optimatch Batch Bowl Attendance Predictions")
st.write(
    "Upload a CSV of bowl matchups (Team 1, Team 2, Venue, Bowl Game, Date) "
    "and this app will run the trained attendance model for every game."
)

st.markdown(
    """
**Required columns in your CSV:**

- `Team 1` – must match Team Name in UI Team Lookup  
- `Team 2` – same as above  
- `Venue` – must match Football Stadium in Venue Lookup  
- `Bowl Game` – must match Bowl Name in Bowl Tiers  
- `Date` – any parsable date format (used to set weekend flag)
"""
)

uploaded = st.file_uploader("Upload Matchups CSV", type=["csv"])

if uploaded is not None:
    df_input = pd.read_csv(uploaded)
    st.subheader("Preview of Uploaded Matchups")
    st.dataframe(df_input.head(20), use_container_width=True)

    if st.button("Run Batch Predictions"):
        results = []
        for idx, row in df_input.iterrows():
            feats, mets = build_features_for_matchup(row)
            if feats is None:
                results.append(
                    {
                        "Row": idx,
                        "Team 1": row.get("Team 1", ""),
                        "Team 2": row.get("Team 2", ""),
                        "Venue": row.get("Venue", ""),
                        "Bowl Game": row.get("Bowl Game", ""),
                        "Date": row.get("Date", ""),
                        "Error": mets.get("error", "Unknown error"),
                    }
                )
                continue

            results.append(
                {
                    "Row": idx,
                    "Team 1": row.get("Team 1", ""),
                    "Team 2": row.get("Team 2", ""),
                    "Venue": row.get("Venue", ""),
                    "Bowl Game": row.get("Bowl Game", ""),
                    "Date": row.get("Date", ""),
                    "Predicted Attendance": round(mets["Predicted Attendance"]),
                    "% Filled": f"{mets['% Filled']:.1%}",
                    "Venue Capacity": int(mets["Venue Capacity"]),
                    "Venue Accessibility": mets["Venue Accessibility"],
                    "TVI Avg": round(mets["TVI Avg"]),
                    "MIS": mets["MIS"],
                    "SVS": mets["SVS"],
                    "MFS": mets["MFS"],
                    "Interest Index": round(mets["Interest Index"], 1),
                    "Sellout Probability": f"{mets['Sellout Prob']*100:.1f}%",
                    "Avg Distance (mi)": round(mets["Avg Distance"]),
                    "Min Distance (mi)": round(mets["Distance Min"]),
                    "Distance Imbalance (mi)": round(mets["Distance Imbalance"]),
                    "Matchup Power": mets["Matchup Power"],
                    "Combined Fanbase (log)": round(mets["Combined Fanbase Log"], 2),
                    "AP Strength Score": round(mets["AP Strength Score"], 1),
                    "Raw Model Pred": round(mets["Raw Pred"]),
                }
            )

        results_df = pd.DataFrame(results)

        st.subheader("Batch Prediction Results")
        st.dataframe(results_df, use_container_width=True)

        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            data=csv_data,
            file_name="batch_bowl_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV to get started.")
