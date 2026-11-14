import pandas as pd
import numpy as np


def create_forecast(daily_df, days=7):
    daily_df = daily_df.copy()
    daily_df["acq_date"] = pd.to_datetime(daily_df["acq_date"])
    daily_df = daily_df.sort_values("acq_date")

    activity = daily_df["activity_index"].values

    if len(activity) < 8:
        return pd.DataFrame()

    last_day = activity[-1]
    previous_day = activity[-2]
    week_ago = activity[-8]

    same_day_last_year = None

    if len(activity) >= 365:
        same_day_last_year = activity[-365]

    if same_day_last_year is None:
        same_day_last_year = week_ago

    forecast = []

    for i in range(days):
        predicted = (
            0.35 * last_day +
            0.20 * previous_day +
            0.20 * week_ago +
            0.25 * same_day_last_year
        )

        forecast.append({
            "day": i + 1,
            "predicted_activity": predicted,
            "low": predicted * 0.65,
            "high": predicted * 1.35
        })

        previous_day = last_day
        last_day = predicted

    return pd.DataFrame(forecast)