# import pandas as pd

# from .config import (
#     FEATURES_PATH,
#     FEATURE_GROUP_NAME,
#     FEATURE_GROUP_VERSION,
# )

# from .hopsworks_client import get_feature_store
# def upload_features():
#     """
#     Upload processed features to Hopsworks Feature Store.
#     """
#     # Connect to Hopsworks
#     fs = get_feature_store()

#     # Load local dataset
#     df = pd.read_parquet(FEATURES_PATH)

#     # Convert timestamp to datetime
#     df["timestamp"] = pd.to_datetime(df["timestamp"])

#     # Remove duplicate rows
#     df = df.drop_duplicates(subset=["city", "timestamp"])
    
#     print(f"Loaded {len(df)} records for upload.")
#     print(f"Dataset Shape: {df.shape}")
#     print(df.head()) 

#     # Create or get Feature Group
#     feature_group = fs.get_or_create_feature_group(
#     name=FEATURE_GROUP_NAME,
#     version=FEATURE_GROUP_VERSION,
#     description="Open-Meteo AQI forecasting features",
#     primary_key=["city", "timestamp"],
#     event_time="timestamp",
#     online_enabled=True,
#     time_travel_format="HUDI",
#     statistics_config=False,
# )


#     # Upload data
#     feature_group.insert(
#     df,
#     write_options={"wait_for_job": True},
# )

#     print("✅ Features uploaded successfully!")


# if __name__ == "__main__":
#     upload_features()
from pathlib import Path

import pandas as pd

from .config import (
    FEATURES_PATH,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

from .hopsworks_client import get_feature_store


def upload_features(source_path: Path = FEATURES_PATH):
    """
    Upload processed features to Hopsworks Feature Store.

    `source_path` defaults to the historical backfill file
    (FEATURES_PATH from config.py) so existing calls to
    upload_features() with no arguments behave exactly as
    before. Pass a different path (e.g. the live hourly
    features file) to upload that instead.
    """

    # Connect to Hopsworks
    fs = get_feature_store()

    # Load local dataset
    df = pd.read_parquet(source_path)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicate rows
    df = df.drop_duplicates(subset=["city", "timestamp"])

    print(f"Loaded {len(df)} records for upload from:\n{source_path}")
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Create or get Feature Group
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Open-Meteo AQI forecasting features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
        time_travel_format="HUDI",
        statistics_config=False,
    )

    # Upload data.
    # Because the feature group uses (city, timestamp) as its
    # primary key and Hudi as its storage format, this call is
    # an UPSERT: rows with a timestamp already present get
    # updated, and new timestamps get appended. This makes it
    # safe to run hourly without creating duplicate rows.
    feature_group.insert(
        df,
        write_options={"wait_for_job": True},
    )

    print("✅ Features uploaded successfully!")


if __name__ == "__main__":
    upload_features()