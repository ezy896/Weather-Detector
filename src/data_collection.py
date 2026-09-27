import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import pandas as pd
from config import (
    OPEN_METEO_ARCHIVE_URL, DATA_RAW_DIR, PAKISTAN_CITIES,
    HISTORICAL_START_DATE, HISTORICAL_END_DATE, city_slug,
)


def fetch_historical_weather(latitude, longitude, start_date, end_date):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean",
        ],
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date"}, inplace=True)
    return df


def save_raw_data(df, city_name):
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    file_path = os.path.join(DATA_RAW_DIR, f"{city_slug(city_name)}_historical.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")
    return file_path


def already_collected(city_name):
    file_path = os.path.join(DATA_RAW_DIR, f"{city_slug(city_name)}_historical.csv")
    return os.path.exists(file_path)


if __name__ == "__main__":
    for city in PAKISTAN_CITIES:
        if already_collected(city["name"]):
            print(f"Skipping {city['name']} — already have data.")
            continue

        print(f"Fetching historical weather for {city['name']}...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                df = fetch_historical_weather(
                    city["latitude"], city["longitude"], HISTORICAL_START_DATE, HISTORICAL_END_DATE
                )
                save_raw_data(df, city["name"])
                break
            except requests.exceptions.HTTPError as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = 20 * attempt
                    print(f"  Rate limited, retrying in {wait}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"  Failed for {city['name']}: {e}")

        time.sleep(10)  # longer gap between cities