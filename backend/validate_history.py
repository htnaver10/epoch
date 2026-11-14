from pathlib import Path
import json

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "historical" / "pilot_2025_06"
INPUT_FILE = DATA_DIR / "canada_boreal_combined.csv"


def main():
    if not INPUT_FILE.exists():
        raise RuntimeError(f"Missing historical CSV: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {
        "latitude",
        "longitude",
        "acq_date",
        "instrument",
        "source",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    valid_coordinates = (
        lat.between(50, 70)
        & lon.between(-140, -60)
    )

    valid_dates = dates.between("2025-06-01", "2025-06-05")

    expected_pairs = {
        ("MODIS_SP", "MODIS"),
        ("VIIRS_NOAA20_SP", "VIIRS"),
    }

    valid_pairs = pd.Series(
        [
            pair in expected_pairs
            for pair in zip(df["source"], df["instrument"])
        ],
        index=df.index,
    )

    expected_dates = set(
        pd.date_range("2025-06-01", "2025-06-05")
        .strftime("%Y-%m-%d")
    )

    dates_by_source = {}
    for source in ["MODIS_SP", "VIIRS_NOAA20_SP"]:
        observed_dates = set(
            dates[df["source"].eq(source)]
            .dropna()
            .dt.strftime("%Y-%m-%d")
        )

        dates_by_source[source] = {
            "rows": int(df["source"].eq(source).sum()),
            "dates_without_detection_rows": sorted(
                expected_dates - observed_dates
            ),
        }

    report = {
        "total_rows": int(len(df)),
        "invalid_or_outside_bbox_rows": int(
            (~valid_coordinates).sum()
        ),
        "invalid_or_outside_date_range_rows": int(
            (~valid_dates).sum()
        ),
        "unexpected_source_instrument_rows": int(
            (~valid_pairs).sum()
        ),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "sources": dates_by_source,
    }

    report_file = DATA_DIR / "quality_report.json"
    report_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    daily = (
        df.groupby(["acq_date", "instrument"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    daily.to_csv(DATA_DIR / "daily_counts.csv")

    print(json.dumps(report, indent=2))
    print(f"\nQuality report saved: {report_file}")
    print("Daily summary saved alongside the report.")
    print(
        "\nDates without detection rows do not automatically "
        "mean zero burning or a failed download."
    )


if __name__ == "__main__":
    main()