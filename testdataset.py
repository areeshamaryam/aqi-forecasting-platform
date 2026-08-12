# import pandas as pd

# df = pd.read_parquet("ml_pipeline/data/processed/features.parquet")

# print(df.head())

# print("\nColumns:")
# print(df.columns)

# print("\nData Types:")
# print(df.dtypes)
# from ml_pipeline.feature_store.hopsworks_client import get_feature_store
# from ml_pipeline.feature_store.config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

# fs = get_feature_store()
# fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
# fg.delete()
# print("Deleted.")

from ml_pipeline.data_collection.openmeteo_client import OpenMeteoClient

client = OpenMeteoClient()

weather_data, air_quality_data = client.get_historical_data(
    latitude=33.6844,
    longitude=73.0479,
    start_date="2024-08-01",
    end_date="2024-08-03",
)

print("\n========== WEATHER VARIABLES ==========")

for key, values in weather_data["hourly"].items():
    if key == "time":
        continue

    missing = sum(v is None for v in values)

    print(f"{key}: {len(values)} values | missing: {missing}")

print("\n========== AIR QUALITY VARIABLES ==========")

for key, values in air_quality_data["hourly"].items():
    if key == "time":
        continue

    missing = sum(v is None for v in values)

    print(f"{key}: {len(values)} values | missing: {missing}")