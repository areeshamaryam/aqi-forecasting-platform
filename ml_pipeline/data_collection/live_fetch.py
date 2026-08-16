from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from .openmeteo_client import OpenMeteoClient


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# CITY CONFIGURATION
# (must match historical_backfill.py exactly)
# ============================================================

CITY = "Islamabad"

LATITUDE = 33.6844
LONGITUDE = 73.0479


# ============================================================
# BUILD LIVE FEATURES
#
# Uses the SAME schema/column logic as
# historical_backfill.build_historical_features(), so the
# output matches the 21-column schema already registered in
# the Hopsworks feature group. Keeping this logic duplicated
# (rather than importing across files) is intentional here so
# the hourly job has no dependency on backfill-specific code.
# ============================================================

def build_live_features(
    weather_data: dict,
    air_quality_data: dict,
) -> pd.DataFrame:

    weather = weather_data["hourly"]
    air = air_quality_data["hourly"]

    weather_df = pd.DataFrame(weather)
    air_df = pd.DataFrame(air)

    weather_df["timestamp"] = pd.to_datetime(weather_df["time"])
    air_df["timestamp"] = pd.to_datetime(air_df["time"])

    weather_df = weather_df.drop(columns=["time"])
    air_df = air_df.drop(columns=["time"])

    df = pd.merge(
        weather_df,
        air_df,
        on="timestamp",
        how="inner",
    )

    df["city"] = CITY

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

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Sort chronologically before computing aqi_change_rate,
    # since it depends on row order.
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_change_rate"] = df["aqi"].diff()

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
# SAVE LIVE FEATURES (separate file from historical backfill)
# ============================================================

def save_live_features(df: pd.DataFrame) -> Path:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DIR / "latest_features.parquet"

    df.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    print(f"\n✅ Live features saved to:\n{output_file}")
    print(f"\nDataset shape: {df.shape}")
    print("\nMost recent rows:")
    print(df.tail())

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Fetch the latest hourly weather + air-quality data and
    save it locally as latest_features.parquet. This is
    intended to run every hour via GitHub Actions, followed
    by an upload step to Hopsworks.
    """

    client = OpenMeteoClient()

    print("=" * 60)
    print("LIVE HOURLY DATA FETCH")
    print("=" * 60)

    print(f"City: {CITY}")
    print(f"Run time (UTC): {datetime.now(timezone.utc)}")

    weather_data, air_quality_data = client.get_recent_data(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        past_days=3,
    )

    df = build_live_features(weather_data, air_quality_data)

    # Drop rows with any missing pollutant/weather values —
    # Open-Meteo occasionally has not-yet-finalized values for
    # the very latest hour(s).
    before = len(df)
    df = df.dropna()
    after = len(df)

    if before != after:
        print(
            f"\nDropped {before - after} incomplete row(s) "
            f"(likely the most recent hour not yet finalized)."
        )

    save_live_features(df)

    print("\n" + "=" * 60)
    print("✅ LIVE FETCH COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()