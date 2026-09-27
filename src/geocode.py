import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from config import GEOCODING_URL, GEOCODING_COUNTRY_CODE


def get_coordinates(city_name):
    """Look up latitude/longitude for a city name, restricted to Pakistan."""
    params = {
        "name": city_name,
        "count": 5,
        "language": "en",
        "format": "json",
        "countryCode": GEOCODING_COUNTRY_CODE,
    }

    response = requests.get(GEOCODING_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        raise ValueError(
            f"Could not find '{city_name}' in Pakistan. Try a different spelling or a nearby major city."
        )

    match = results[0]
    return {
        "city": match["name"],
        "country": match.get("country", ""),
        "latitude": match["latitude"],
        "longitude": match["longitude"],
    }


if __name__ == "__main__":
    print(get_coordinates("Lahore"))