import json
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_CITY
from .weather_client import OpenWeatherClient


RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def save_json(data, folder_name, city):
    """Save API response as a JSON file."""

    folder = RAW_DATA_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{city.lower()}.json"

    file_path = folder / filename

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"✅ Saved: {file_path}")


def main():
    client = OpenWeatherClient()

    city = DEFAULT_CITY

    print(f"Fetching data for {city}...")

    lat, lon = client.get_coordinates(city)

    weather_data = client.get_weather(lat, lon)
    pollution_data = client.get_air_pollution(lat, lon)

    save_json(weather_data, "weather", city)
    save_json(pollution_data, "pollution", city)

    print("✅ Data collection completed.")


if __name__ == "__main__":
    main()