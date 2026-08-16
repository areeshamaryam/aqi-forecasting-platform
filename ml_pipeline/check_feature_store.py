"""
Quick sanity check: confirms how much data is actually sitting
in your Hopsworks Feature Store right now, before you run
prepare_training_data.py.

Run this from your project root with:
    python -m ml_pipeline.check_feature_store
"""

from .feature_store.get_features import get_features


def main():

    print("=" * 60)
    print("CHECKING HOPSWORKS FEATURE STORE")
    print("=" * 60)

    df = get_features()

    print(f"\nTotal rows in feature group: {len(df)}")
    print(f"Columns: {df.shape[1]}")

    if "timestamp" in df.columns:

        import pandas as pd

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        print(
            f"\nDate range: "
            f"{df['timestamp'].min()} "
            f"to "
            f"{df['timestamp'].max()}"
        )

        span_days = (
            df["timestamp"].max() - df["timestamp"].min()
        ).days

        print(f"Span: ~{span_days} days")

    print("\n" + "=" * 60)

    if len(df) < 5000:
        print(
            "⚠️  WARNING: This looks like a small/partial dataset.\n"
            "You need ~2 years of hourly data (≈17,000+ rows) for\n"
            "72-hour forecasting to work well and to have enough\n"
            "rows left after dropping the last 72 hours per target.\n"
            "\n"
            "If your backfill hasn't been uploaded to Hopsworks yet,\n"
            "re-run your historical_backfill.py -> upload_features.py\n"
            "steps before continuing."
        )
    else:
        print(
            f"✅ Looks good — {len(df)} rows is enough for "
            f"72-hour multi-horizon training."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()