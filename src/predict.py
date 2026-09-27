import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import time
import joblib
import pandas as pd
import requests

from config import (
    MODELS_DIR, OPEN_METEO_BASE_URL, PAKISTAN_CITIES, MAX_FORECAST_DAYS,
    MAX_NEAREST_CITY_DISTANCE_KM, FORECAST_CACHE_TTL_SECONDS, city_slug,
)
from src.train_model import TEMP_FEATURE_COLUMNS, PRECIP_FEATURE_COLUMNS

_weather_cache = {}  # key: (round(lat,2), round(lon,2)) -> (timestamp, dataframe)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_city(latitude, longitude):
    def distance(city):
        return haversine_km(latitude, longitude, city["latitude"], city["longitude"])
    nearest = min(PAKISTAN_CITIES, key=distance)
    return nearest, distance(nearest)


def _load_model_with_check(path, expected_features):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}. Run src/train_model.py first.")

    meta_path = path.replace(".pkl", "_features.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            saved_features = json.load(f)
        if saved_features != expected_features:
            raise RuntimeError(
                f"Model at {path} was trained with different features than expected. "
                "Retrain with src/train_model.py."
            )
    return joblib.load(path)


def load_models_for_city(city_name):
    slug = city_slug(city_name)
    temp_path = os.path.join(MODELS_DIR, f"{slug}_temp.pkl")
    precip_path = os.path.join(MODELS_DIR, f"{slug}_precip.pkl")

    temp_model = _load_model_with_check(temp_path, TEMP_FEATURE_COLUMNS)
    precip_model = _load_model_with_check(precip_path, PRECIP_FEATURE_COLUMNS)
    return temp_model, precip_model


def fetch_recent_weather(latitude, longitude, days=14):
    cache_key = (round(latitude, 2), round(longitude, 2))
    cached = _weather_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < FORECAST_CACHE_TTL_SECONDS:
        return cached[1]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": days,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "windspeed_10m_max", "relative_humidity_2m_mean",
        ],
        "timezone": "auto",
    }
    response = requests.get(f"{OPEN_METEO_BASE_URL}/forecast", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()["daily"]
    df = pd.DataFrame(data)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    _weather_cache[cache_key] = (time.time(), df)
    return df


def _build_feature_row(df, next_date, last_real_row):
    temp_series = df["temperature_2m_mean"]
    precip_series = df["precipitation_sum"]

    temp_features = {
        "month": next_date.month, "day_of_year": next_date.dayofyear,
        "season": next_date.month % 12 // 3 + 1,
        "temperature_2m_mean_lag1": temp_series.iloc[-1],
        "temperature_2m_mean_lag2": temp_series.iloc[-2],
        "temperature_2m_mean_lag3": temp_series.iloc[-3],
        "temperature_2m_mean_lag7": temp_series.iloc[-7],
        "temperature_2m_mean_roll_mean3": temp_series.iloc[-3:].mean(),
        "temperature_2m_mean_roll_mean7": temp_series.iloc[-7:].mean(),
        "temperature_2m_mean_roll_mean14": temp_series.iloc[-14:].mean(),
        "precipitation_sum_lag1": precip_series.iloc[-1],  # fixed, matches training now
        "windspeed_10m_max": last_real_row["windspeed_10m_max"],
        "relative_humidity_2m_mean": last_real_row["relative_humidity_2m_mean"],
    }

    precip_features = {
        "month": next_date.month, "day_of_year": next_date.dayofyear,
        "season": next_date.month % 12 // 3 + 1,
        "precipitation_sum_lag1": precip_series.iloc[-1],
        "precipitation_sum_lag2": precip_series.iloc[-2],
        "precipitation_sum_lag3": precip_series.iloc[-3],
        "precipitation_sum_lag7": precip_series.iloc[-7],
        "precipitation_sum_roll_mean3": precip_series.iloc[-3:].mean(),
        "precipitation_sum_roll_mean7": precip_series.iloc[-7:].mean(),
        "precipitation_sum_roll_mean14": precip_series.iloc[-14:].mean(),
        "windspeed_10m_max": last_real_row["windspeed_10m_max"],
        "relative_humidity_2m_mean": last_real_row["relative_humidity_2m_mean"],
        "temperature_2m_mean_lag1": temp_series.iloc[-1],
    }

    return (
        pd.DataFrame([temp_features])[TEMP_FEATURE_COLUMNS],
        pd.DataFrame([precip_features])[PRECIP_FEATURE_COLUMNS],
    )


def predict_forecast(latitude, longitude, days=1):
    days = max(1, min(days, MAX_FORECAST_DAYS))

    nearest_city, distance_km = find_nearest_city(latitude, longitude)
    if distance_km > MAX_NEAREST_CITY_DISTANCE_KM:
        raise ValueError(
            f"This location is too far ({round(distance_km)}km) from any city we have trained data for. "
            "Try a major Pakistani city."
        )

    temp_model, precip_model = load_models_for_city(nearest_city["name"])

    history = fetch_recent_weather(latitude, longitude)
    last_real_row = history.iloc[-1]

    forecasts = []
    working_df = history.copy()

    for _ in range(days):
        next_date = working_df["date"].iloc[-1] + pd.Timedelta(days=1)
        temp_row, precip_row = _build_feature_row(working_df, next_date, last_real_row)

        predicted_temp = round(float(temp_model.predict(temp_row)[0]), 1)
        predicted_precip = round(max(0.0, float(precip_model.predict(precip_row)[0])), 1)

        forecasts.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_mean_temp_c": predicted_temp,
            "predicted_precipitation_mm": predicted_precip,
        })

        new_row = {
            "date": next_date,
            "temperature_2m_mean": predicted_temp,
            "precipitation_sum": predicted_precip,
            "windspeed_10m_max": last_real_row["windspeed_10m_max"],
            "relative_humidity_2m_mean": last_real_row["relative_humidity_2m_mean"],
        }
        working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)

    return forecasts, nearest_city["name"]