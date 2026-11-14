import pandas as pd
import numpy as np

GRID_SIZE = 0.25

CONFIDENCE_WEIGHTS = {
    "l": 0.25,
    "n": 1.0,
    "h": 1.5,
    "low": 0.25,
    "nominal": 1.0,
    "high": 1.5
}


def assign_grid_cell(lat, lon, grid_size=GRID_SIZE):
    lat_cell = np.floor(lat / grid_size)
    lon_cell = np.floor(lon / grid_size)
    return f"{lat_cell}_{lon_cell}"


def confidence_weight(value):
    if pd.isna(value):
        return 1.0

    value = str(value).lower()

    if value in CONFIDENCE_WEIGHTS:
        return CONFIDENCE_WEIGHTS[value]

    try:
        numeric = float(value)

        if numeric < 30:
            return 0.25
        elif numeric < 70:
            return 1.0
        else:
            return 1.5

    except ValueError:
        return 1.0


def harmonize_data(df):
    df = df.copy()

    df["grid_cell"] = df.apply(
        lambda row: assign_grid_cell(row["latitude"], row["longitude"]),
        axis=1
    )

    df["confidence_weight"] = df["confidence"].apply(confidence_weight)

    df["is_modis"] = df["instrument"].str.contains("MODIS", case=False, na=False)
    df["is_viirs"] = df["instrument"].str.contains("VIIRS", case=False, na=False)

    grouped = df.groupby(["acq_date", "grid_cell"]).agg(
        modis_count=("is_modis", "sum"),
        viirs_count=("is_viirs", "sum"),
        total_detections=("confidence_weight", "count"),
        weighted_detections=("confidence_weight", "sum"),
        frp_total=("frp", "sum"),
        day_count=("daynight", lambda x: (x == "D").sum()),
        night_count=("daynight", lambda x: (x == "N").sum())
    ).reset_index()

    grouped["activity_index"] = (
        (grouped["modis_count"] / 1.0) +
        (grouped["viirs_count"] / 2.67)
    )

    return grouped