import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import DATA_PROCESSED_DIR, MODELS_DIR, PAKISTAN_CITIES, city_slug


TEMP_FEATURE_COLUMNS = [
    "month", "day_of_year", "season",
    "temperature_2m_mean_lag1", "temperature_2m_mean_lag2",
    "temperature_2m_mean_lag3", "temperature_2m_mean_lag7",
    "temperature_2m_mean_roll_mean3", "temperature_2m_mean_roll_mean7",
    "temperature_2m_mean_roll_mean14",
    "precipitation_sum_lag1",  # fixed: was same-day precipitation_sum (train/serve mismatch)
    "windspeed_10m_max", "relative_humidity_2m_mean",
]
TEMP_TARGET_COLUMN = "temperature_2m_mean"

PRECIP_FEATURE_COLUMNS = [
    "month", "day_of_year", "season",
    "precipitation_sum_lag1", "precipitation_sum_lag2",
    "precipitation_sum_lag3", "precipitation_sum_lag7",
    "precipitation_sum_roll_mean3", "precipitation_sum_roll_mean7",
    "precipitation_sum_roll_mean14",
    "windspeed_10m_max", "relative_humidity_2m_mean",
    "temperature_2m_mean_lag1",
]
PRECIP_TARGET_COLUMN = "precipitation_sum"


def load_processed_data(city_name):
    file_path = os.path.join(DATA_PROCESSED_DIR, f"{city_slug(city_name)}_processed.csv")
    return pd.read_csv(file_path, parse_dates=["date"])


def train_single_model(df, feature_cols, target_col, lag1_col, label):
    X, y = df[feature_cols], df[target_col]
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    naive_mae = mean_absolute_error(y_test, X_test[lag1_col])
    improvement = (1 - mae / naive_mae) * 100 if naive_mae else 0

    print(f"  [{label}] MAE: {mae:.2f}  (naive: {naive_mae:.2f}, {improvement:.1f}% better)")
    return model


def save_model_with_metadata(model, feature_cols, path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, path)
    meta_path = path.replace(".pkl", "_features.json")
    with open(meta_path, "w") as f:
        json.dump(feature_cols, f)


def train_city(city_name):
    df = load_processed_data(city_name)
    slug = city_slug(city_name)

    temp_model = train_single_model(
        df, TEMP_FEATURE_COLUMNS, TEMP_TARGET_COLUMN, "temperature_2m_mean_lag1", "temp"
    )
    precip_model = train_single_model(
        df, PRECIP_FEATURE_COLUMNS, PRECIP_TARGET_COLUMN, "precipitation_sum_lag1", "precip"
    )

    save_model_with_metadata(temp_model, TEMP_FEATURE_COLUMNS, os.path.join(MODELS_DIR, f"{slug}_temp.pkl"))
    save_model_with_metadata(precip_model, PRECIP_FEATURE_COLUMNS, os.path.join(MODELS_DIR, f"{slug}_precip.pkl"))


if __name__ == "__main__":
    for city in PAKISTAN_CITIES:
        print(f"Training models for {city['name']}...")
        try:
            train_city(city["name"])
        except FileNotFoundError:
            print(f"  Skipping {city['name']} — no processed data found.")