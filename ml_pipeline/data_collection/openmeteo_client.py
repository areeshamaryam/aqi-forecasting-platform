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