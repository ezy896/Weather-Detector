import os
from dotenv import load_dotenv

load_dotenv()

# API
OPEN_METEO_BASE_URL = os.getenv("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1")
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
GEOCODING_COUNTRY_CODE = "PK"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Historical data range for training
HISTORICAL_START_DATE = "2015-01-01"
HISTORICAL_END_DATE = "2024-12-31"

# Forecast
MAX_FORECAST_DAYS = 5
MAX_NEAREST_CITY_DISTANCE_KM = 300  # beyond this, refuse rather than guess

# Live weather cache
FORECAST_CACHE_TTL_SECONDS = 1800  # 30 minutes

# Default (used only if geocoding fails and no city given)
DEFAULT_CITY = "Abbottabad"
DEFAULT_LATITUDE = 34.1688
DEFAULT_LONGITUDE = 73.2215

PAKISTAN_CITIES = [
    {"name": "Karachi", "latitude": 24.8607, "longitude": 67.0011},
    {"name": "Lahore", "latitude": 31.5497, "longitude": 74.3436},
    {"name": "Islamabad", "latitude": 33.6844, "longitude": 73.0479},
    {"name": "Rawalpindi", "latitude": 33.5651, "longitude": 73.0169},
    {"name": "Faisalabad", "latitude": 31.4504, "longitude": 73.1350},
    {"name": "Multan", "latitude": 30.1575, "longitude": 71.5249},
    {"name": "Peshawar", "latitude": 34.0151, "longitude": 71.5249},
    {"name": "Quetta", "latitude": 30.1798, "longitude": 66.9750},
    {"name": "Abbottabad", "latitude": 34.1688, "longitude": 73.2215},
    {"name": "Hyderabad", "latitude": 25.3960, "longitude": 68.3578},
    {"name": "Sialkot", "latitude": 32.4945, "longitude": 74.5229},
    {"name": "Gujranwala", "latitude": 32.1877, "longitude": 74.1945},
]


def city_slug(city_name):
    return city_name.lower().replace(" ", "_")