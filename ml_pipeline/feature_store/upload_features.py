import pandas as pd

from .config import (
    FEATURES_PATH,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

from .hopsworks_client import get_feature_store
def upload_features():
    """
    Upload processed features to Hopsworks Feature Store.
    """
    # Connect to Hopsworks
    fs = get_feature_store()

    # Load local dataset
    df = pd.read_parquet(FEATURES_PATH)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Remove duplicate rows
    df = df.drop_duplicates(subset=["city", "timestamp"])
    
    print(f"Loaded {len(df)} records for upload.")
    print(f"Dataset Shape: {df.shape}")
    print(df.head()) 

    # Create or get Feature Group
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Engineered AQI forecasting features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
        time_travel_format="HUDI",

    )

    # Upload data
    feature_group.insert(
    df,
    write_options={"wait_for_job": True},
    validation_options={"save_report": True},

)

    print("✅ Features uploaded successfully!")


if __name__ == "__main__":
    upload_features()