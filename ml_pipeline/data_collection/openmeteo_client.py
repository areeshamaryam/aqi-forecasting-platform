import requests


class OpenMeteoClient:
    """
    Client for Open-Meteo weather and air-quality APIs.
    """

    WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
    AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def get_historical_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ):
        """
        Fetch hourly historical weather and air-quality data.
        """

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": (
                "temperature_2m,"
                "apparent_temperature,"
                "relative_humidity_2m,"
                "surface_pressure,"
                "wind_speed_10m,"
                "cloud_cover"
            ),
            "timezone": "Asia/Karachi",
        }

        air_quality_params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": (
                "pm2_5,"
                "pm10,"
                "carbon_monoxide,"
                "nitrogen_dioxide,"
                "ozone,"
                "sulphur_dioxide,"
                "us_aqi"
            ),
            "timezone": "Asia/Karachi",
        }

        print("Fetching historical weather data...")

        weather_response = requests.get(
            self.WEATHER_URL,
            params=weather_params,
            timeout=30,
        )
        weather_response.raise_for_status()

        print("Fetching historical air-quality data...")

        air_quality_response = requests.get(
            self.AIR_QUALITY_URL,
            params=air_quality_params,
            timeout=30,
        )
        air_quality_response.raise_for_status()

        return (
            weather_response.json(),
            air_quality_response.json(),
        )

    def get_recent_data(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 3,
    ):
        """
        Fetch RECENT + CURRENT hourly weather and air-quality
        data using Open-Meteo's forecast endpoints (NOT the
        archive endpoint above, which typically lags a few
        days behind and does not include the latest hour).

        `past_days` controls how much history is included
        alongside the current hour. We request extra history
        (default 3 days) so that lag/rolling features (which
        need up to 24 hours of prior AQI) can be computed for
        the most recent hour, not just raw current values.
        """

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "past_days": past_days,
            "forecast_days": 1,
            "hourly": (
                "temperature_2m,"
                "apparent_temperature,"
                "relative_humidity_2m,"
                "surface_pressure,"
                "wind_speed_10m,"
                "cloud_cover"
            ),
            "timezone": "Asia/Karachi",
        }

        air_quality_params = {
            "latitude": latitude,
            "longitude": longitude,
            "past_days": past_days,
            "forecast_days": 1,
            "hourly": (
                "pm2_5,"
                "pm10,"
                "carbon_monoxide,"
                "nitrogen_dioxide,"
                "ozone,"
                "sulphur_dioxide,"
                "us_aqi"
            ),
            "timezone": "Asia/Karachi",
        }

        print("Fetching recent + current weather data...")

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=weather_params,
            timeout=30,
        )
        weather_response.raise_for_status()

        print("Fetching recent + current air-quality data...")

        air_quality_response = requests.get(
            self.AIR_QUALITY_URL,
            params=air_quality_params,
            timeout=30,
        )
        air_quality_response.raise_for_status()

        return (
            weather_response.json(),
            air_quality_response.json(),
        )