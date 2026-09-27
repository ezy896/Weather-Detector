import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import requests
from flask import Flask, render_template, request, jsonify
from src.predict import predict_forecast
from src.geocode import get_coordinates
from config import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, DEFAULT_CITY, MAX_FORECAST_DAYS, PAKISTAN_CITIES
import datetime

app = Flask(__name__, template_folder="../templates", static_folder="../static")


def _get_demo_forecast(city_name, days=5):
    """Generate realistic forecast simulation for preview or offline use."""
    base_temps = {
        "Karachi": 29.5, "Lahore": 27.0, "Islamabad": 24.0, "Rawalpindi": 24.5,
        "Peshawar": 25.0, "Quetta": 17.5, "Abbottabad": 19.0, "Multan": 28.5,
        "Faisalabad": 27.5, "Hyderabad": 30.0, "Sialkot": 26.0, "Gujranwala": 26.5
    }
    base_t = base_temps.get(city_name, 23.5)
    today = datetime.date.today()
    forecasts = []
    for i in range(1, days + 1):
        d = today + datetime.timedelta(days=i)
        # slight natural variation
        day_offset = math.sin(i * 1.3) * 2.2
        precip = round(max(0.0, (math.sin(i * 0.9) - 0.2) * 4.5), 1)
        temp = round(base_t + day_offset, 1)
        forecasts.append({
            "date": d.strftime("%Y-%m-%d"),
            "predicted_mean_temp_c": temp,
            "predicted_precipitation_mm": precip,
        })
    return forecasts


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_city=DEFAULT_CITY,
        max_days=MAX_FORECAST_DAYS,
        cities=PAKISTAN_CITIES,
    )


@app.route("/api/cities")
def api_cities():
    return jsonify({"success": True, "cities": PAKISTAN_CITIES})


def _parse_days(raw_value):
    try:
        days = int(raw_value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(days, MAX_FORECAST_DAYS))


@app.route("/api/forecast")
def api_forecast():
    city = request.args.get("city", "").strip()
    days = _parse_days(request.args.get("days"))
    is_demo = request.args.get("demo", "").lower() in ("1", "true", "yes")
    lat_arg = request.args.get("lat")
    lon_arg = request.args.get("lon")

    if is_demo:
        searched_city = city if city else DEFAULT_CITY
        forecasts = _get_demo_forecast(searched_city, days=days)
        return jsonify({
            "success": True,
            "searched_city": searched_city,
            "model_city": searched_city,
            "forecasts": forecasts,
            "is_demo": True,
        })

    try:
        if lat_arg and lon_arg:
            try:
                latitude = float(lat_arg)
                longitude = float(lon_arg)
                searched_city = city if city else f"Coord ({latitude:.2f}, {longitude:.2f})"
            except ValueError:
                latitude, longitude = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
                searched_city = DEFAULT_CITY
        elif city:
            location = get_coordinates(city)
            latitude, longitude = location["latitude"], location["longitude"]
            searched_city = location["city"]
        else:
            latitude, longitude = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
            searched_city = DEFAULT_CITY

        forecasts, model_city = predict_forecast(latitude, longitude, days=days)

        return jsonify({
            "success": True,
            "searched_city": searched_city,
            "model_city": model_city,
            "forecasts": forecasts,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except requests.exceptions.RequestException:
        # If open-meteo is unreachable and user asked for city, provide fallback option info
        return jsonify({
            "success": False,
            "error": "Weather service is unreachable right now. You can try again or preview with Demo Mode.",
            "can_demo": True,
        }), 503
    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception:
        return jsonify({"success": False, "error": "Something went wrong generating the forecast."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)