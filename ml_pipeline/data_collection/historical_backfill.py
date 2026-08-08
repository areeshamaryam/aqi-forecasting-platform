from pathlib import Path

import pandas as pd

from .openmeteo_client import OpenMeteoClient


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Islamabad coordinates
CITY = "Islamabad"
LATITUDE = 33.6844
LONGITUDE = 73.0479


def build_historical_features(
    weather_data: dict,
    air_quality_data: dict,
) -> pd.DataFrame:
    """
    Convert Open-Meteo historical weather and air-quality
    responses into our standardized feature dataset.
    """

    weather = weather_data["hourly"]
    air = air_quality_data["hourly"]

    weather_df = pd.DataFrame(weather)
    air_df = pd.DataFrame(air)

    # Convert timestamps to datetime
    weather_df["timestamp"] = pd.to_datetime(weather_df["time"])
    air_df["timestamp"] = pd.to_datetime(air_df["time"])

    # Remove original time column
    weather_df = weather_df.drop(columns=["time"])
    air_df = air_df.drop(columns=["time"])

    # Combine weather and air-quality data
    df = pd.merge(
        weather_df,
        air_df,
        on="timestamp",
        how="inner",
    )

    # Add city
    df["city"] = CITY

    # Rename Open-Meteo fields to our project schema
    df = df.rename(
        columns={
            "temperature_2m": "temperature",
            "apparent_temperature": "feels_like",
            "relative_humidity_2m": "humidity",
            "surface_pressure": "pressure",
            "wind_speed_10m": "wind_speed",
            "cloud_cover": "clouds",
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "ozone": "o3",
            "sulphur_dioxide": "so2",
            "ammonia": "nh3",
            "us_aqi": "aqi",
        }
    )

    # Open-Meteo does not provide NO in this request.
    # Keep the existing project schema by setting it to NaN.
    df["no"] = pd.NA

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # AQI change rate
    df["aqi_change_rate"] = df["aqi"].diff()

    # Arrange columns
    columns = [
        "city",
        "timestamp",
        "temperature",
        "feels_like",
        "humidity",
        "pressure",
        "wind_speed",
        "visibility",
        "clouds",
        "aqi",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "aqi_change_rate",
    ]

    df = df[columns]

    return df


def save_historical_features(df: pd.DataFrame):
    """
    Save historical features to a separate Parquet file.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DIR / "historical_features.parquet"

    df.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    print(f"\n✅ Historical features saved to:")
    print(output_file)

    print(f"\nDataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nLast 5 records:")
    print(df.tail())


def main():
    """
    Run historical data backfill for the initial test period.
    """

    client = OpenMeteoClient()

    start_date = "2024-08-01"
    end_date =   "2026-07-31"

    print("=" * 60)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 60)

    print(f"City: {CITY}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    weather_data, air_quality_data = client.get_historical_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=start_date,
        end_date=end_date,
    )

    df = build_historical_features(
        weather_data,
        air_quality_data,
    )

    save_historical_features(df)

    print("\n✅ Historical backfill completed successfully!")


if __name__ == "__main__":
    main()