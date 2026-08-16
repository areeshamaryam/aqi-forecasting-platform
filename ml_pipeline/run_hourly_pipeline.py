"""
Entry point for the HOURLY automated pipeline.

Run with:
    python -m ml_pipeline.run_hourly_pipeline

This is the single command the GitHub Actions hourly workflow
calls. It does two things in order:
    1. Fetch the latest weather + air-quality data and save it
       locally as latest_features.parquet
    2. Upload that file into the Hopsworks feature group
"""

from pathlib import Path

from .data_collection.live_fetch import main as fetch_live_data
from .feature_store.upload_features import upload_features


def main():

    print("#" * 60)
    print("# HOURLY PIPELINE START")
    print("#" * 60)

    # --------------------------------------------------------
    # Step 1: Fetch latest data
    # --------------------------------------------------------

    fetch_live_data()

    # --------------------------------------------------------
    # Step 2: Upload to Hopsworks
    # --------------------------------------------------------

    latest_features_path = (
        Path(__file__).resolve().parent
        / "data"
        / "processed"
        / "latest_features.parquet"
    )

    print("\n" + "#" * 60)
    print("# UPLOADING LIVE FEATURES TO HOPSWORKS")
    print("#" * 60)

    upload_features(source_path=latest_features_path)

    print("\n" + "#" * 60)
    print("# HOURLY PIPELINE COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()