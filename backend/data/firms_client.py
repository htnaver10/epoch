import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")

RAW_DIR = Path("../data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

AOIS = {
    "canada_boreal": (-140, 50, -60, 70),
    "amazon_basin": (-75, -15, -45, 5),
    "indogangetic_plain": (68, 21, 92, 32)
}

SENSORS = [
    "MODIS_SP",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP"
]


def build_url(sensor, bbox, day_range, start_date):
    west, south, east, north = bbox
    coords = f"{west},{south},{east},{north}"
    return (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/{sensor}/{coords}/{day_range}/{start_date}"
    )


def fetch_firms_data(sensor, aoi_name, start_date, day_range=5):
    bbox = AOIS[aoi_name]
    url = build_url(sensor, bbox, day_range, start_date)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    file_path = RAW_DIR / f"{aoi_name}_{sensor}_{start_date}.csv"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(response.text)

    return file_path


def load_csv(file_path):
    return pd.read_csv(file_path)


def fetch_date_range(aoi_name, start_date, end_date):
    current = pd.to_datetime(start_date)
    final = pd.to_datetime(end_date)

    all_files = []

    while current <= final:
        for sensor in SENSORS:
            file_path = fetch_firms_data(
                sensor=sensor,
                aoi_name=aoi_name,
                start_date=current.strftime("%Y-%m-%d"),
                day_range=5
            )
            all_files.append(file_path)

        current += pd.Timedelta(days=5)

    return all_files