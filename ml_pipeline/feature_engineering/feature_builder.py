from pathlib import Path
from datetime import datetime

import pandas as pd
from .utils import load_json, get_latest_file


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_WEATHER_DIR = BASE_DIR / "data" / "raw" / "weather"
RAW_POLLUTION_DIR = BASE_DIR / "data" / "raw" / "pollution"

PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# WEATHER FEATURE EXTRACTION
# ============================================================

def extract_weather_features(weather_data: dict) -> list:
    """
    Extract hourly weather features from Open-Meteo response.
    """

    hourly = weather_data["hourly"]

    timestamps = hourly["time"]

    records = []

    for i, timestamp in enumerate(timestamps):

        record = {
            "timestamp": timestamp,

            "temperature": hourly["temperature_2m"][i],
            "feels_like": hourly["apparent_temperature"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "pressure": hourly["surface_pressure"][i],
            "wind_speed": hourly["wind_speed_10m"][i],
            "clouds": hourly["cloud_cover"][i],
        }

        records.append(record)

    return records


# ============================================================
# AIR QUALITY FEATURE EXTRACTION
# ============================================================

def extract_pollution_features(air_quality_data: dict) -> list:
    """
    Extract hourly air-quality features from Open-Meteo response.
    """

    hourly = air_quality_data["hourly"]

    timestamps = hourly["time"]

    records = []

    for i, timestamp in enumerate(timestamps):

        record = {
            "timestamp": timestamp,

            "aqi": hourly["us_aqi"][i],

            "co": hourly["carbon_monoxide"][i],
            "no2": hourly["nitrogen_dioxide"][i],
            "o3": hourly["ozone"][i],
            "so2": hourly["sulphur_dioxide"][i],

            "pm2_5": hourly["pm2_5"][i],
            "pm10": hourly["pm10"][i],
        }

        records.append(record)

    return records


# ============================================================
# BUILD COMPLETE FEATURE DATASET
# ============================================================

def build_feature_dataframe(
    weather_data: dict,
    air_quality_data: dict,
    city: str,
) -> pd.DataFrame:
    """
    Combine weather and air-quality data into a single
    hourly feature dataset.
    """

    weather_records = extract_weather_features(weather_data)
    pollution_records = extract_pollution_features(air_quality_data)

    weather_df = pd.DataFrame(weather_records)
    pollution_df = pd.DataFrame(pollution_records)

    # Merge using timestamp
    df = pd.merge(
        weather_df,
        pollution_df,
        on="timestamp",
        how="inner",
    )

    # Add city
    df["city"] = city

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ========================================================
    # TIME-BASED FEATURES
    # ========================================================

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ========================================================
    # AQI CHANGE RATE
    # ========================================================

    df["aqi_change_rate"] = df["aqi"].diff()

    return df


# ============================================================
# SAVE PROCESSED FEATURES
# ============================================================

def save_processed_features(df: pd.DataFrame):
    """
    Save processed features to Parquet.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_file = (
        PROCESSED_DIR / "features.parquet"
    )

    df.to_parquet(
        parquet_file,
        index=False,
        engine="pyarrow",
    )

    print(
        f"✅ Features saved to: {parquet_file}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def display_dataset_summary(df: pd.DataFrame):
    """
    Display basic information about the processed dataset.
    """

    print("\n========== DATASET SUMMARY ==========")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Build features from the latest Open-Meteo data.
    """

    weather_file = get_latest_file(
        RAW_WEATHER_DIR
    )

    pollution_file = get_latest_file(
        RAW_POLLUTION_DIR
    )

    print(
        f"📄 Weather File: {weather_file.name}"
    )

    print(
        f"📄 Pollution File: {pollution_file.name}"
    )

    weather_data = load_json(
        weather_file
    )

    air_quality_data = load_json(
        pollution_file
    )

    # Get city from configuration if available
    city = "Islamabad"

    df = build_feature_dataframe(
        weather_data,
        air_quality_data,
        city,
    )

    save_processed_features(df)

    display_dataset_summary(df)

    print(
        "\n✅ Open-Meteo Feature Engineering "
        "Completed Successfully!"
    )


if __name__ == "__main__":
    main()