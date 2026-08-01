
from pathlib import Path
from datetime import datetime

from .utils import load_json, save_json, get_latest_file


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_WEATHER_DIR = BASE_DIR / "data" / "raw" / "weather"
RAW_POLLUTION_DIR = BASE_DIR / "data" / "raw" / "pollution"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
def extract_weather_features(weather_data: dict) -> dict:
    """
    Extract relevant weather features from OpenWeather response.
    """

    return {
        "city": weather_data.get("name"),

        "timestamp": datetime.fromtimestamp(
            weather_data["dt"]
        ).strftime("%Y-%m-%d %H:%M:%S"),

        "temperature": weather_data["main"]["temp"],
        "feels_like": weather_data["main"]["feels_like"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],

        "wind_speed": weather_data["wind"]["speed"],

        "visibility": weather_data.get("visibility"),

        "clouds": weather_data["clouds"]["all"],
    }
def extract_pollution_features(pollution_data: dict) -> dict:
    """
    Extract relevant air pollution features from OpenWeather response.
    """

    pollution = pollution_data["list"][0]

    return {
        "aqi": pollution["main"]["aqi"],

        "co": pollution["components"]["co"],
        "no": pollution["components"]["no"],
        "no2": pollution["components"]["no2"],
        "o3": pollution["components"]["o3"],
        "so2": pollution["components"]["so2"],
        "pm2_5": pollution["components"]["pm2_5"],
        "pm10": pollution["components"]["pm10"],
        "nh3": pollution["components"]["nh3"],
    }
def build_feature_record(weather_data: dict, pollution_data: dict) -> dict:
    """
    Build a complete feature record by combining weather and pollution data.
    """

    weather_features = extract_weather_features(weather_data)
    pollution_features = extract_pollution_features(pollution_data)

    feature_record = {
        **weather_features,
        **pollution_features,
    }

    # Extract time-based features
    timestamp = datetime.strptime(
        feature_record["timestamp"],
        "%Y-%m-%d %H:%M:%S"
    )

    feature_record["hour"] = timestamp.hour
    feature_record["day"] = timestamp.day
    feature_record["month"] = timestamp.month
    feature_record["day_of_week"] = timestamp.weekday()
    feature_record["is_weekend"] = int(timestamp.weekday() >= 5)

    return feature_record

def save_processed_features(feature_record: dict):
    """
    Save processed feature record as a JSON file.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = (
        f"{feature_record['city'].lower()}_{timestamp}_features.json"
    )

    file_path = PROCESSED_DIR / filename

    save_json(feature_record, file_path)

    print(f"✅ Processed features saved to: {file_path}")


def main():
    """
    Build features from the latest weather and pollution data.
    """

    weather_file = get_latest_file(RAW_WEATHER_DIR)
    pollution_file = get_latest_file(RAW_POLLUTION_DIR)

    print(f"📄 Weather File: {weather_file.name}")
    print(f"📄 Pollution File: {pollution_file.name}")

    weather_data = load_json(weather_file)
    pollution_data = load_json(pollution_file)

    feature_record = build_feature_record(
        weather_data,
        pollution_data,
    )

    save_processed_features(feature_record)

    print("\n✅ Feature Engineering Completed Successfully!")


if __name__ == "__main__":
    main()