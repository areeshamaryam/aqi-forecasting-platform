from pathlib import Path

import numpy as np
import pandas as pd

from ..feature_store.get_features import get_features


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "training"
)

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "aqi"

# --------------------------------------------------------
# FULL HOURLY FORECASTING (next 3 days = 72 hours)
#
# Instead of a single 1-hour-ahead target, or a handful of
# checkpoint targets (24h/48h/72h), we now create ONE target
# column PER HOUR from t+1 to t+72. This lets a single
# multi-output model predict the complete hourly AQI curve
# for the next 3 days in one call, matching the project
# brief exactly ("predict AQI in the next 3 days").
# --------------------------------------------------------

FORECAST_HORIZON_HOURS = 72

TARGET_COLUMNS = [
    f"target_aqi_h{h}"
    for h in range(1, FORECAST_HORIZON_HOURS + 1)
]

# Chronological split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA FROM HOPSWORKS
# ============================================================

def load_dataset():
    """
    Retrieve the engineered AQI features from Hopsworks
    and prepare them for chronological time-series processing.
    """

    print("=" * 60)
    print("LOADING FEATURES FROM HOPSWORKS")
    print("=" * 60)

    # Existing get_features.py handles the Hopsworks connection
    df = get_features()

    print(
        f"\nRetrieved dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df.sort_values(
            "timestamp"
        )
        .reset_index(drop=True)
    )

    print(
        f"\nDate range: "
        f"{df['timestamp'].min()} "
        f"to "
        f"{df['timestamp'].max()}"
    )

    # --------------------------------------------------------
    # Verify chronological order
    # --------------------------------------------------------

    chronological = (
        df["timestamp"]
        .is_monotonic_increasing
    )

    print(
        f"Chronological order: "
        f"{'✅ YES' if chronological else '❌ NO'}"
    )

    if not chronological:
        raise ValueError(
            "Dataset is not in chronological order."
        )

    return df


# ============================================================
# CLEAN INITIAL MISSING VALUE
# ============================================================

def clean_initial_missing_values(df):

    print("\n" + "=" * 60)
    print("HANDLING INITIAL MISSING VALUES")
    print("=" * 60)

    if "aqi_change_rate" in df.columns:

        missing_before = (
            df["aqi_change_rate"]
            .isna()
            .sum()
        )

        print(
            f"aqi_change_rate missing values before: "
            f"{missing_before}"
        )

        df["aqi_change_rate"] = (
            df["aqi_change_rate"]
            .fillna(0)
        )

        missing_after = (
            df["aqi_change_rate"]
            .isna()
            .sum()
        )

        print(
            f"aqi_change_rate missing values after: "
            f"{missing_after}"
        )

    return df


# ============================================================
# CREATE LAG FEATURES
# ============================================================

def create_lag_features(df):

    print("\n" + "=" * 60)
    print("CREATING AQI LAG FEATURES")
    print("=" * 60)

    lag_hours = [1, 3, 6, 12, 24]

    for lag in lag_hours:

        column_name = f"aqi_lag_{lag}h"

        df[column_name] = (
            df[TARGET_COLUMN]
            .shift(lag)
        )

        print(f"Created: {column_name}")

    return df


# ============================================================
# CREATE ROLLING FEATURES
# ============================================================

def create_rolling_features(df):

    print("\n" + "=" * 60)
    print("CREATING ROLLING AQI FEATURES")
    print("=" * 60)

    windows = [3, 6, 24]

    previous_aqi = df[TARGET_COLUMN].shift(1)

    for window in windows:

        mean_column = f"aqi_rolling_mean_{window}h"
        std_column = f"aqi_rolling_std_{window}h"

        df[mean_column] = previous_aqi.rolling(window=window).mean()
        df[std_column] = previous_aqi.rolling(window=window).std()

        print(f"Created: {mean_column}")
        print(f"Created: {std_column}")

    return df


# ============================================================
# CREATE CYCLICAL TIME FEATURES
# ============================================================

def create_cyclical_time_features(df):

    print("\n" + "=" * 60)
    print("CREATING CYCLICAL TIME FEATURES")
    print("=" * 60)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print("Created: hour_sin")
    print("Created: hour_cos")
    print("Created: month_sin")
    print("Created: month_cos")

    return df


# ============================================================
# CREATE FULL HOURLY FORECAST TARGETS (t+1 ... t+72)
# ============================================================

def create_targets(df):
    """
    Create ONE target column per future hour, from 1 hour
    ahead to 72 hours (3 days) ahead. This lets a single
    multi-output model output the entire hourly AQI curve
    for the next 3 days in one prediction call.
    """

    print("\n" + "=" * 60)
    print(
        f"CREATING {FORECAST_HORIZON_HOURS}-HOUR "
        f"FORECAST TARGETS"
    )
    print("=" * 60)

    for h in range(1, FORECAST_HORIZON_HOURS + 1):

        column_name = f"target_aqi_h{h}"

        df[column_name] = df[TARGET_COLUMN].shift(-h)

    print(
        f"Created {len(TARGET_COLUMNS)} target columns: "
        f"target_aqi_h1 ... target_aqi_h{FORECAST_HORIZON_HOURS}"
    )

    return df


# ============================================================
# REMOVE INVALID TRAINING ROWS
# ============================================================

def remove_invalid_rows(df):

    print("\n" + "=" * 60)
    print("REMOVING INVALID TRAINING ROWS")
    print("=" * 60)

    rows_before = len(df)

    required_columns = [
        "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h",
        "aqi_lag_12h", "aqi_lag_24h",
        "aqi_rolling_mean_3h", "aqi_rolling_mean_6h",
        "aqi_rolling_mean_24h",
        "aqi_rolling_std_3h", "aqi_rolling_std_6h",
        "aqi_rolling_std_24h",
    ] + TARGET_COLUMNS

    df = (
        df.dropna(subset=required_columns)
        .reset_index(drop=True)
    )

    rows_after = len(df)

    print(f"Rows before: {rows_before}")
    print(f"Rows after: {rows_after}")
    print(f"Rows removed: {rows_before - rows_after}")

    return df


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def chronological_split(df):

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 60)

    total_rows = len(df)

    train_end = int(total_rows * TRAIN_RATIO)
    validation_end = int(total_rows * (TRAIN_RATIO + VALIDATION_RATIO))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print(f"Total rows: {total_rows}")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")

    print("\nTraining period:")
    print(train_df["timestamp"].min())
    print("to")
    print(train_df["timestamp"].max())

    print("\nValidation period:")
    print(validation_df["timestamp"].min())
    print("to")
    print(validation_df["timestamp"].max())

    print("\nTest period:")
    print(test_df["timestamp"].min())
    print("to")
    print(test_df["timestamp"].max())

    return train_df, validation_df, test_df


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(train_df, validation_df, test_df):

    print("\n" + "=" * 60)
    print("SAVING TRAINING DATASETS")
    print("=" * 60)

    train_path = TRAINING_DIR / "train.parquet"
    validation_path = TRAINING_DIR / "validation.parquet"
    test_path = TRAINING_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    validation_df.to_parquet(validation_path, index=False, engine="pyarrow")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")

    print(f"✅ Training dataset saved:\n{train_path}")
    print(f"\n✅ Validation dataset saved:\n{validation_path}")
    print(f"\n✅ Test dataset saved:\n{test_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()
    df = clean_initial_missing_values(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_cyclical_time_features(df)
    df = create_targets(df)
    df = remove_invalid_rows(df)

    train_df, validation_df, test_df = chronological_split(df)

    save_datasets(train_df, validation_df, test_df)

    print("\n" + "=" * 60)
    print("✅ TRAINING DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()