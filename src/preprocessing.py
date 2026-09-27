import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, PAKISTAN_CITIES, city_slug


def load_raw_data(city_name):
    file_path = os.path.join(DATA_RAW_DIR, f"{city_slug(city_name)}_historical.csv")
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df


def add_date_features(df):
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"] % 12 // 3 + 1
    return df


def add_lag_features(df, target_col, lags=(1, 2, 3, 7)):
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, windows=(3, 7, 14)):
    for window in windows:
        df[f"{target_col}_roll_mean{window}"] = df[target_col].shift(1).rolling(window).mean()
    return df


def clean_data(df):
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


def preprocess(city_name):
    df = load_raw_data(city_name)
    df = add_date_features(df)
    df = add_lag_features(df, "temperature_2m_mean")
    df = add_rolling_features(df, "temperature_2m_mean")
    df = add_lag_features(df, "precipitation_sum")
    df = add_rolling_features(df, "precipitation_sum")
    df = clean_data(df)
    return df


def save_processed_data(df, city_name):
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    file_path = os.path.join(DATA_PROCESSED_DIR, f"{city_slug(city_name)}_processed.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} rows to {file_path}")
    return file_path


if __name__ == "__main__":
    for city in PAKISTAN_CITIES:
        print(f"Preprocessing data for {city['name']}...")
        try:
            df = preprocess(city["name"])
            save_processed_data(df, city["name"])
        except FileNotFoundError:
            print(f"  Skipping {city['name']} — no raw data found. Run data_collection.py for it first.")