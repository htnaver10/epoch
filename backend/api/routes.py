from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "canada_boreal_combined.csv"
)


@router.get("/sample")
def get_sample():
    if not CSV_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Sample CSV not found. Run download_sample.py first.",
        )

    df = pd.read_csv(CSV_PATH)

    # Convert coordinates to numbers and reject invalid locations.
    for column in ["latitude", "longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    df["instrument"] = (
        df["instrument"].fillna("UNKNOWN").str.upper()
    )

    # Daily raw counts, kept separate for each instrument.
    counts = (
        df.groupby(["acq_date", "instrument"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["MODIS", "VIIRS"], fill_value=0)
        .sort_index()
    )

    daily = [
        {
            "date": str(date),
            "modis_count": int(row["MODIS"]),
            "viirs_count": int(row["VIIRS"]),
        }
        for date, row in counts.iterrows()
    ]

    # Coordinates for the map, separate from daily summaries.
    hotspots = [
        {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "date": str(row.acq_date),
            "instrument": str(row.instrument),
        }
        for row in df.itertuples(index=False)
    ]

    return {
        "aoi": "canada_boreal",
        "description": "Rectangular test region; raw detections only.",
        "total_detections": len(hotspots),
        "daily": daily,
        "hotspots": hotspots,
    }

@router.get("/history/summary")
def historical_summary():
    from pathlib import Path

    import pandas as pd
    from fastapi import HTTPException

    project_dir = Path(__file__).resolve().parents[2]
    history_dir = project_dir / "data" / "historical" / "2025_06"

    summary_file = history_dir / "daily_counts.csv"
    manifest_file = history_dir / "download_manifest.csv"

    if not summary_file.exists() or not manifest_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Historical summary or download manifest is missing. "
                "Run download_history_month.py first."
            ),
        )

    daily_frame = pd.read_csv(summary_file)
    manifest = pd.read_csv(manifest_file)

    required_daily = {"acq_date", "MODIS", "VIIRS"}

    if not required_daily.issubset(daily_frame.columns):
        raise HTTPException(
            status_code=500,
            detail="Historical daily summary has unexpected columns.",
        )

    required_manifest = {
        "source",
        "start_date",
        "end_date",
        "rows",
        "status",
    }

    if not required_manifest.issubset(manifest.columns):
        raise HTTPException(
            status_code=500,
            detail="Historical manifest has unexpected columns.",
        )

    if manifest.empty or not manifest["status"].eq("validated").all():
        raise HTTPException(
            status_code=500,
            detail="Historical manifest contains unvalidated batches.",
        )

    daily_frame = daily_frame.sort_values("acq_date")

    daily = [
        {
            "date": str(row["acq_date"]),
            "modis_count": int(row["MODIS"]),
            "viirs_count": int(row["VIIRS"]),
        }
        for _, row in daily_frame.iterrows()
    ]

    modis_total = sum(day["modis_count"] for day in daily)
    viirs_total = sum(day["viirs_count"] for day in daily)

    if modis_total + viirs_total != int(manifest["rows"].sum()):
        raise HTTPException(
            status_code=500,
            detail="Summary counts do not match the download manifest.",
        )

    return {
        "aoi": "canada_boreal",
        "description": (
            "Rectangular test region; raw standard-processing "
            "thermal anomaly detections, separated by sensor."
        ),
        "period_start": "2025-06-01",
        "period_end": "2025-06-30",
        "sources": ["MODIS_SP", "VIIRS_NOAA20_SP"],
        "total_detections": modis_total + viirs_total,
        "modis_detections": modis_total,
        "viirs_detections": viirs_total,
        "dates_with_detection_rows": len(daily),
        "validated_batches": len(manifest),
        "daily": daily,
        "limitations": [
            "Detections are not counts of unique fires or burned area.",
            "Raw MODIS and VIIRS counts are not harmonized.",
            "One month is not a multi-year seasonal baseline.",
            "Dates absent from the summary are not automatically zero.",
        ],
    }