from io import StringIO
from pathlib import Path
import os

import pandas as pd
import requests
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

SOURCES = [
    "MODIS_SP",
    "VIIRS_NOAA20_SP",
]


def check_source(source):
    url = (
        "https://firms.modaps.eosdis.nasa.gov"
        f"/api/data_availability/csv/{MAP_KEY}/{source}"
    )

    print(f"\nChecking {source}...")

    try:
        response = requests.get(url, timeout=60)
    except requests.RequestException:
        raise RuntimeError(
            f"{source}: request failed. Check your internet connection "
            "and try again."
        ) from None

    if response.status_code != 200:
        raise RuntimeError(
            f"{source}: HTTP {response.status_code}. "
            "Check your key and FIRMS service availability."
        )

    text = response.text.strip()

    if not text:
        raise RuntimeError(f"{source}: FIRMS returned an empty response.")

    try:
        frame = pd.read_csv(StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        raise RuntimeError(
            f"{source}: FIRMS did not return a readable CSV."
        ) from None

    required = {"data_id", "min_date", "max_date"}

    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"{source}: unexpected response format. "
            "Check your key and source availability."
        )

    if frame.empty:
        raise RuntimeError(
            f"{source}: no availability records were returned."
        )

    print(frame[["data_id", "min_date", "max_date"]].to_string(index=False))


def main():
    if not MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY was not found in backend/.env."
        )

    for source in SOURCES:
        check_source(source)


if __name__ == "__main__":
    main()