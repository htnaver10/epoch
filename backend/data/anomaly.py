import pandas as pd
import numpy as np


def add_rolling_baseline(daily_df, window=15):
    daily_df = daily_df.copy()
    daily_df["acq_date"] = pd.to_datetime(daily_df["acq_date"])
    daily_df = daily_df.sort_values("acq_date")

    daily_df["rolling_median"] = (
        daily_df["activity_index"]
        .rolling(window, min_periods=3)
        .median()
    )

    daily_df["rolling_mad"] = (
        daily_df["activity_index"]
        .rolling(window, min_periods=3)
        .apply(lambda x: np.median(np.abs(x - np.median(x))))
    )

    return daily_df


def calculate_anomaly_score(daily_df):
    daily_df = daily_df.copy()

    daily_df["anomaly_z"] = (
        (daily_df["activity_index"] - daily_df["rolling_median"]) /
        (1.4826 * daily_df["rolling_mad"] + 1e-6)
    )

    def classify(value):
        if value < 1:
            return "normal"
        elif value < 2:
            return "elevated"
        elif value < 3:
            return "high"
        else:
            return "extreme"

    daily_df["status"] = daily_df["anomaly_z"].apply(classify)

    return daily_df