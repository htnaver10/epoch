from datetime import date, timedelta
from pathlib import Path
import time

import pandas as pd

import download_history as downloader


PROJECT_DIR = Path(__file__).resolve().parent.parent
MONTH_DIR = PROJECT_DIR / "data" / "historical" / "2025_06"
CHUNK_DIR = MONTH_DIR / "chunks"

START = date(2025, 6, 1)
END = date(2025, 6, 30)

SOURCES = ["MODIS_SP", "VIIRS_NOAA20_SP"]

REQUIRED = {
    "latitude",
    "longitude",
    "acq_date",
    "instrument",
    "confidence",
    "frp",
    "daynight",
    "source",
}


def validate_chunk(frame, source, start, stop):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{source}, {start}: missing columns {sorted(missing)}"
        )

    if frame.empty:
        return

    dates = pd.to_datetime(frame["acq_date"], errors="coerce")
    lat = pd.to_numeric(frame["latitude"], errors="coerce")
    lon = pd.to_numeric(frame["longitude"], errors="coerce")

    expected_instrument = (
        "MODIS" if source == "MODIS_SP" else "VIIRS"
    )

    valid = (
        dates.between(str(start), str(stop))
        & lat.between(50, 70)
        & lon.between(-140, -60)
        & frame["source"].eq(source)
        & frame["instrument"].eq(expected_instrument)
    )

    if not valid.all():
        raise RuntimeError(
            f"{source}, {start}: "
            f"{int((~valid).sum())} rows failed validation."
        )

    if frame.duplicated().any():
        raise RuntimeError(
            f"{source}, {start}: exact duplicate rows found."
        )


def main():
    if not downloader.MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY was not found in backend/.env."
        )

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    manifest = []
    current = START

    while current <= END:
        stop = min(current + timedelta(days=4), END)
        days = (stop - current).days + 1

        batch_dir = CHUNK_DIR / current.isoformat()
        batch_dir.mkdir(parents=True, exist_ok=True)

        downloader.OUTPUT_DIR = batch_dir
        downloader.START_DATE = current.isoformat()
        downloader.DAY_RANGE = days

        for source in SOURCES:
            path = batch_dir / (
                f"{downloader.AOI_NAME}_{source}.csv"
            )

            if path.exists():
                print(f"\nUsing saved batch: {source}, {current}")
                frame = pd.read_csv(path)
            else:
                print(f"\nRequesting {current} through {stop}")
                frame = downloader.download_source(source)
                time.sleep(2)

            validate_chunk(frame, source, current, stop)

            frames.append(frame)
            manifest.append({
                "source": source,
                "start_date": current.isoformat(),
                "end_date": stop.isoformat(),
                "rows": len(frame),
                "status": "validated",
            })

        current = stop + timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True)

    if combined.duplicated().any():
        raise RuntimeError(
            "Exact duplicate rows found across batches. "
            "Combined output was not written."
        )

    output = MONTH_DIR / "canada_boreal_combined.csv"
    combined.to_csv(output, index=False)

    daily = (
        combined.groupby(["acq_date", "instrument"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    daily.to_csv(MONTH_DIR / "daily_counts.csv")
    pd.DataFrame(manifest).to_csv(
        MONTH_DIR / "download_manifest.csv",
        index=False,
    )

    print(f"\nCombined CSV saved to: {output}")
    print(f"Total detections: {len(combined):,}")

    print("\nDetections by source:")
    print(combined.groupby("source").size().to_string())

    print("\nDaily detections:")
    print(daily.to_string())


if __name__ == "__main__":
    main()