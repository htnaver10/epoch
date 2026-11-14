import pandas as pd
import numpy as np


def calculate_daily_activity(harmonized_df, aoi_area_km2):
    daily = (
        harmonized_df
        .groupby("acq_date")
        .agg(
            modis_count=("modis_count", "sum"),
            viirs_count=("viirs_count", "sum"),
            weighted_detections=("weighted_detections", "sum"),
            frp_total=("frp_total", "sum"),
            activity_index=("activity_index", "sum")
        )
        .reset_index()
    )

    daily["activity_per_km2"] = daily["activity_index"] / aoi_area_km2

    daily["activity_score"] = (
        100 * daily["activity_per_km2"] / daily["activity_per_km2"].max()
    )

    daily["activity_score"] = daily["activity_score"].fillna(0)

    return daily


def calculate_monthly_activity(daily_df):
    daily_df = daily_df.copy()
    daily_df["acq_date"] = pd.to_datetime(daily_df["acq_date"])
    daily_df["month"] = daily_df["acq_date"].dt.to_period("M")

    monthly = (
        daily_df
        .groupby("month")
        .agg(
            total_activity=("activity_index", "sum"),
            active_days=("activity_index", lambda x: (x > 0).sum()),
            peak_daily_activity=("activity_index", "max")
        )
        .reset_index()
    )

    return monthly


def identify_fire_season(monthly_df, threshold_percent=80):
    monthly_df = monthly_df.copy()

    threshold = monthly_df["total_activity"].quantile(
        threshold_percent / 100
    )

    fire_months = monthly_df[
        monthly_df["total_activity"] >= threshold
    ]["month"].tolist()

    return {
        "threshold": threshold,
        "fire_months": fire_months
    }