import pandas as pd

from .config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

from .hopsworks_client import get_feature_store


def get_features():
    """
    Retrieve AQI forecasting features from Hopsworks Feature Store
    and sort them chronologically for time-series processing.
    """

    # Connect to Hopsworks
    fs = get_feature_store()

    print("\nRetrieving Feature Group...")
    print(f"Name: {FEATURE_GROUP_NAME}")
    print(f"Version: {FEATURE_GROUP_VERSION}")

    # Get existing Feature Group
    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    print("✅ Feature Group found!")

    # Create query for all features
    query = feature_group.select_all()

    # Retrieve data from Hopsworks
    df = query.read()

    # ============================================================
    # SORT CHRONOLOGICALLY
    # ============================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values(
            ["city", "timestamp"]
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("FEATURE RETRIEVAL SUCCESSFUL")
    print("=" * 60)

    print(f"\nRetrieved shape: {df.shape}")

    print("\nFeatures:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    # ============================================================
    # CHRONOLOGICAL ORDER CHECK
    # ============================================================

    print("\nFirst 5 chronological rows:")
    print(
        df[
            ["city", "timestamp", "aqi"]
        ].head()
    )

    print("\nLast 5 chronological rows:")
    print(
        df[
            ["city", "timestamp", "aqi"]
        ].tail()
    )

    # Verify chronological order
    is_sorted = (
        df[["city", "timestamp"]]
        .equals(
            df[["city", "timestamp"]]
            .sort_values(
                ["city", "timestamp"]
            )
            .reset_index(drop=True)
        )
    )

    print(
        f"\nChronological order verified: "
        f"{'✅ YES' if is_sorted else '❌ NO'}"
    )

    print("\nData types:")
    print(df.dtypes)

    return df


if __name__ == "__main__":
    get_features()