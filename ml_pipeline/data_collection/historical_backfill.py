from pathlib import Path

import pandas as pd

from .openmeteo_client import OpenMeteoClient


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# CITY CONFIGURATION
# ============================================================

CITY = "Islamabad"

LATITUDE = 33.6844
LONGITUDE = 73.0479


# ============================================================
# BUILD HISTORICAL FEATURES
# ============================================================

def build_historical_features(
    weather_data: dict,
    air_quality_data: dict,
) -> pd.DataFrame:
    """
    Convert Open-Meteo historical weather and air-quality
    responses into the standardized project feature dataset.
    """

    weather = weather_data["hourly"]
    air = air_quality_data["hourly"]

    weather_df = pd.DataFrame(weather)
    air_df = pd.DataFrame(air)

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    weather_df["timestamp"] = pd.to_datetime(
        weather_df["time"]
    )

    air_df["timestamp"] = pd.to_datetime(
        air_df["time"]
    )

    # Remove original Open-Meteo time column
    weather_df = weather_df.drop(
        columns=["time"]
    )

    air_df = air_df.drop(
        columns=["time"]
    )

    # --------------------------------------------------------
    # Merge weather + air quality
    # --------------------------------------------------------

    df = pd.merge(
        weather_df,
        air_df,
        on="timestamp",
        how="inner",
    )

    # --------------------------------------------------------
    # Add city
    # --------------------------------------------------------

    df["city"] = CITY

    # --------------------------------------------------------
    # Rename Open-Meteo fields
    # --------------------------------------------------------

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

            "us_aqi": "aqi",
        }
    )

    # --------------------------------------------------------
    # Time-based features
    # --------------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    df["day"] = df["timestamp"].dt.day

    df["month"] = df["timestamp"].dt.month

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # AQI change rate
    # --------------------------------------------------------

    df["aqi_change_rate"] = df["aqi"].diff()

    # --------------------------------------------------------
    # Select final project schema
    # --------------------------------------------------------

    columns = [
        "city",
        "timestamp",

        "temperature",
        "feels_like",
        "humidity",
        "pressure",
        "wind_speed",
        "clouds",

        "aqi",
        "co",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",

        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",

        "aqi_change_rate",
    ]

    df = df[columns]

    return df


# ============================================================
# SAVE HISTORICAL FEATURES
# ============================================================

def save_historical_features(df: pd.DataFrame):
    """
    Save historical features to Parquet.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DIR
        / "historical_features.parquet"
    )

    df.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    print("\n✅ Historical features saved to:")
    print(output_file)

    print(
        f"\nDataset shape: {df.shape}"
    )

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())

    print("\nLast 5 records:")
    print(df.tail())

    print("\nMissing values:")
    print(df.isnull().sum())


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run historical data backfill.
    """

    client = OpenMeteoClient()

    start_date = "2024-08-01"
    end_date = "2026-07-31"

    print("=" * 60)
    print("HISTORICAL DATA BACKFILL")
    print("=" * 60)

    print(f"City: {CITY}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    # --------------------------------------------------------
    # Fetch historical data
    # --------------------------------------------------------

    weather_data, air_quality_data = (
        client.get_historical_data(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            start_date=start_date,
            end_date=end_date,
        )
    )

    # --------------------------------------------------------
    # Build standardized features
    # --------------------------------------------------------

    df = build_historical_features(
        weather_data,
        air_quality_data,
    )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    save_historical_features(df)

    print(
        "\n✅ Historical backfill completed successfully!"
    )


if __name__ == "__main__":
    main()