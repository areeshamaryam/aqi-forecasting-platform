import requests
from .config import(
    OPENWEATHER_API_KEY,
    GEOCODING_API_URL,
    WEATHER_API_URL,
    AIR_POLLUTION_API_URL,
    REQUEST_TIMEOUT,
)


class OpenWeatherClient:
    def __init__(self):
        self.api_key = OPENWEATHER_API_KEY

    def get_coordinates(self, city: str):
        """Get latitude and longitude for a city."""

        params = {
            "q": city,
            "limit": 1,
            "appid": self.api_key,
        }

        response = requests.get(
            GEOCODING_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if not data:
            raise ValueError(f"City '{city}' not found.")

        return data[0]["lat"], data[0]["lon"]

    def get_weather(self, lat: float, lon: float):
        """Fetch current weather data."""

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        response = requests.get(
            WEATHER_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    def get_air_pollution(self, lat: float, lon: float):
        """Fetch current air pollution data."""

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
        }

        response = requests.get(
            AIR_POLLUTION_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()