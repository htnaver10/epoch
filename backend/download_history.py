from io import StringIO
from pathlib import Path
import os

import pandas as pd
import requests
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")
MAP_KEY = os.getenv("FIRMS_MAP_KEY")

OUTPUT_DIR = PROJECT_DIR / "data" / "historical" / "pilot_2025_06"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same bounding box as the existing canada_boreal configuration.
# This is a rectangular test region, not an exact boreal boundary.
AOI_NAME = "canada_boreal"
BBOX = "-140,50,-60,70"

SOURCES = ["MODIS_SP", "VIIRS_NOAA20_SP"]

START_DATE = "2025-06-01"
DAY_RANGE = 5


def download_source(source):
    url = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{source}/{BBOX}/{DAY_RANGE}/{START_DATE}"
    )

    print(f"Downloading {source}...")
    response = requests.get(url, timeout=120)

    # Avoid printing the request URL because it contains your key.
    if response.status_code != 200:
        raise RuntimeError(
            f"{source}: HTTP {response.status_code}. "
            "Check your FIRMS key and service availability."
        )

    text = response.text.strip()

    if not text:
        raise RuntimeError(f"{source}: empty response from FIRMS.")

    try:
        frame = pd.read_csv(StringIO(text))
    except pd.errors.ParserError:
        raise RuntimeError(f"{source}: response was not a valid CSV.")

    required = {
        "latitude",
        "longitude",
        "acq_date",
        "instrument",
        "confidence",
        "frp",
        "daynight",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{source}: expected hotspot CSV columns were missing: "
            f"{sorted(missing)}. Check your key and source availability."
        )

    frame["source"] = source
    frame.to_csv(OUTPUT_DIR / f"{AOI_NAME}_{source}.csv", index=False)

    print(f"Saved {len(frame):,} detections.")
    return frame


def main():
    if not MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY was not found in backend/.env."
        )

    frames = [download_source(source) for source in SOURCES]
    combined = pd.concat(frames, ignore_index=True)

    output_file = OUTPUT_DIR / f"{AOI_NAME}_combined.csv"
    combined.to_csv(output_file, index=False)

    print(f"\nCombined CSV saved to: {output_file}")

    if combined.empty:
        print("No detections returned. This is not necessarily an error.")
        return

    daily_counts = (
        combined.groupby(["acq_date", "instrument"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print("\nRaw detections by date and instrument:")
    print(daily_counts.to_string())


if __name__ == "__main__":
    main()