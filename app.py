import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# LOAD MODEL + FEATURE COLUMNS
# =====================================================
model = joblib.load("attendance_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# =====================================================
# LOAD 2025 BOWL GAMES DATA (PRE-ENGINEERED FEATURES)
# =====================================================
# IMPORTANT: make sure this filename matches your repo.
# It should be the same "future" file you used in Colab:
# "2025 Bowl Games - 2025 Bowl Games (8).csv" or similar.
BOWL_FILE = "2025 Bowl Games - 2025 Bowl Games.csv"

bowl_df = pd.read_csv(BOWL_FILE)
bowl_df.columns = bowl_df.columns.str.strip()

# Ensure numeric types for feature columns
for col in feature_cols:
    bowl_df[col] = pd.to_numeric(bowl_df[col], errors="coerce")

# Clean capacity (in case there are commas)
if "Venue Capacity" in bowl_df.columns:
    bowl_df["Venue Capacity"] = (
        bowl_df["Venue Capacity"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    bowl_df["Venue Capacity"] = pd.to_numeric(
        bowl_df["Venue Capacity"], errors="coerce"
    )

# Create a nice label for the dropdown
bowl_df["Game Label"] = (
    bowl_df["Bowl Game Name"] + " – " +
    bowl_df["Team 1"] + " vs " + bowl_df["Team 2"]
)

game_labels = bowl_df["Game Label"].tolist()

# =====================================================
# STREAMLIT UI SETUP
# =====================================================
st.set_page_config(page_title="Optimatch Bowl Attendance Predictor", layout="wide")

st.title("🏈 Optimatch Bowl Attendance Predictor")
st.write(
    "This tool uses CSMG’s Gradient Boosted model and the same feature "
    "set used in your 2022–2025 bowl projections. For 2025 games, "
    "predictions will match your master spreadsheet exactly."
)

# =====================================================
# GAME SELECTION
# =====================================================
st.header("Select 2025 Bowl Game")

selected_label = st.selectbox("Bowl Matchup", sorted(game_labels))

# Get the selected row
row = bowl_df[bowl_df["Game Label"] == selected_label].iloc[0]

# Basic info
bowl_name = row["Bowl Game Name"]
team1 = row["Team 1"]
team2 = row["Team 2"]
venue = row["Venue"]
city = row["City"]
state = row["State"]
year = int(row["Year"])

venue_capacity = row["Venue Capacity"]
bowl_avg_att = row["Bowl Avg Attendees"]
bowl_avg_viewers = row.get("Bowl Avg Viewers", np.nan)

st.subheader(f"{bowl_name}: {team1} vs {team2}")
st.write(f"Year: {year}")
st.write(f"Venue: {venue} — {city}, {state}")
st.write(f"Stadium Capacity: {venue_capacity:,.0f}")
st.write(f"Historical Avg Attendance: {bowl_avg_att:,.0f}")

# =====================================================
# TRAVEL MAP
# =====================================================
st.subheader("Travel Map")

map_points = []

if pd.notna(row.get("Team 1 Lat")) and pd.notna(row.get("Team 1 Lon")):
    map_points.append(
        {"name": team1, "lat": row["Team 1 Lat"], "lon": row["Team 1 Lon"]}
    )

if pd.notna(row.get("Team 2 Lat")) and pd.notna(row.get("Team 2 Lon")):
    map_points.append(
        {"name": team2, "lat": row["Team 2 Lat"], "lon": row["Team 2 Lon"]}
    )

if pd.notna(row.get("Venue Lat")) and pd.notna(row.get("Venue Lon")):
    map_points.append(
        {"name": venue, "lat": row["Venue Lat"], "lon": row["Venue Lon"]}
    )

if map_points:
    map_df = pd.DataFrame(map_points)
    st.map(map_df[["lat", "lon"]])
    st.caption("Markers show Team 1, Team 2, and the bowl venue.")
else:
    st.caption("No latitude/longitude data available for this game.")

# =====================================================
# BUILD FEATURE ROW DIRECTLY FROM CSV
# =====================================================
X_row = row[feature_cols].to_frame().T

# Coerce all to numeric (safety)
X_row = X_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# =====================================================
# PREDICTION WITH YOUR CUSTOM RULES
# =====================================================
st.header("Prediction")

if st.button("Run Prediction"):

    # Raw model prediction (same as BR column in your sheet)
    raw_pred = float(model.predict(X_row)[0])

    # 70/30 blend vs historical avg (same as BS column)
    blended_pred = 0.7 * raw_pred + 0.3 * bowl_avg_att

    # 1. Hawaiʻi & Bahamas bowls use raw prediction only
    bowl_lower = str(bowl_name).lower()
    if ("hawai" in bowl_lower) or ("bahamas" in bowl_lower):
        final_pred = raw_pred
    else:
        final_pred = blended_pred

    # 2. Bowl-specific boosts
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

    # 3. Capacity cap & zero floor
    final_pred = min(final_pred, venue_capacity)
    final_pred = max(final_pred, 0.0)

    pct_filled = final_pred / venue_capacity if venue_capacity > 0 else 0.0

    # =====================================================
    # SIDE-BY-SIDE COMPARISON
    # =====================================================
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

    # Optional: show the stored predictions from your sheet for sanity check
    stored_raw = row.get("Raw Gradient Boosted Attendance", np.nan)
    stored_blend = row.get("Gradient Boosted Predicted Attendance", np.nan)
    stored_final = row.get("Final Attendance Prediction", np.nan)

    with st.expander("Show stored model outputs from 2025 spreadsheet"):
        st.write(f"Stored Raw GBR (BR): {stored_raw:,.5f}")
        st.write(f"Stored Blended (BS): {stored_blend:,.5f}")
        st.write(f"Stored Final (BT): {stored_final:,.5f}")

    # =====================================================
    # KEY DRIVER SUMMARY (LIGHTWEIGHT)
    # =====================================================
    st.subheader("Key Driver Summary")

    drivers = []

    # We can re-use some pre-engineered columns from the row:
    distance_min = row.get("Distance Minimum", np.nan)
    distance_imbalance = row.get("Distance Imbalance", np.nan)
    avg_distance = row.get("Avg Distace Traveled", np.nan)
    combined_fan_log = row.get("Combined Fanbase (Log transformed)", np.nan)
    matchup_power = row.get("Matchup Power Score (2=P4vP4, 1=P4vG5, 0=G5vG5)", np.nan)
    local_team = row.get("Local Team Indicator", 0)

    try:
        overall_median_log = bowl_df["Combined Fanbase (Log transformed)"].median()
    except Exception:
        overall_median_log = combined_fan_log

    if pd.notna(distance_min) and distance_min < 150:
        drivers.append("At least one team is within strong drive range (<150 miles).")

    if pd.notna(distance_imbalance) and distance_imbalance > 400:
        drivers.append("There is a significant travel imbalance between the fanbases.")

    if pd.notna(combined_fan_log) and pd.notna(overall_median_log) and combined_fan_log > overall_median_log:
        drivers.append("Combined fanbase size is well above typical FBS levels.")

    if matchup_power == 2:
        drivers.append("Power vs. Power matchup elevates national interest.")
    elif matchup_power == 1:
        drivers.append("Presence of at least one power-conference team lifts demand.")

    if local_team == 1:
        drivers.append("A local or near-local team strongly supports attendance.")

    if not drivers:
        drivers.append("This matchup aligns with typical bowl attendance patterns in our model.")

    for d in drivers:
        st.write("• " + d)




