from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "historical_features.parquet"

EDA_DIR = BASE_DIR / "data" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print("\n" + "=" * 60)
    print("DATASET LOADED")
    print("=" * 60)

    print(f"Path: {DATA_PATH}")
    print(f"Shape: {df.shape}")

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_dataset(df):

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # Data types
    print("\n--- Data Types ---")
    print(df.dtypes)

    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    # Duplicate rows
    print("\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Duplicate city/timestamp combinations
    print("\n--- Duplicate City/Timestamp ---")

    if "city" in df.columns and "timestamp" in df.columns:
        duplicate_keys = df.duplicated(
            subset=["city", "timestamp"]
        ).sum()

        print(
            f"Duplicate city/timestamp records: "
            f"{duplicate_keys}"
        )

    # Timestamp validation
    print("\n--- Timestamp Validation ---")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Minimum timestamp: {df['timestamp'].min()}")
    print(f"Maximum timestamp: {df['timestamp'].max()}")

    # Time gaps
    print("\n--- Time Interval Check ---")

    sorted_df = df.sort_values("timestamp")

    time_diff = (
        sorted_df["timestamp"]
        .diff()
        .dropna()
    )

    print("Most common time intervals:")
    print(time_diff.value_counts().head())

    # Numeric summary
    print("\n--- Numeric Summary ---")
    print(df.describe().T)


# ============================================================
# AQI ANALYSIS
# ============================================================

def analyze_aqi(df):

    print("\n" + "=" * 60)
    print("AQI ANALYSIS")
    print("=" * 60)

    print("\nAQI statistics:")
    print(df["aqi"].describe())

    print("\nAQI minimum:", df["aqi"].min())
    print("AQI maximum:", df["aqi"].max())
    print("AQI mean:", df["aqi"].mean())
    print("AQI median:", df["aqi"].median())

    # AQI categories based on Open-Meteo/OpenWeather AQI scale
    def aqi_category(aqi):

        if pd.isna(aqi):
            return "Unknown"
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"

    df["aqi_category"] = df["aqi"].apply(aqi_category)

    print("\nAQI category distribution:")
    print(
        df["aqi_category"]
        .value_counts()
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    print("\nCorrelation with AQI:")
    print(
        correlation["aqi"]
        .sort_values(ascending=False)
    )

    # Save correlation matrix
    correlation.to_csv(
        EDA_DIR / "correlation_matrix.csv"
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_visualizations(df):

    print("\n" + "=" * 60)
    print("CREATING EDA VISUALIZATIONS")
    print("=" * 60)

    # --------------------------------------------------------
    # AQI over time
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp"],
        df["aqi"]
    )

    plt.title("AQI Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("AQI")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_over_time.png",
        dpi=150
    )

    plt.close()

    print("Saved: aqi_over_time.png")

    # --------------------------------------------------------
    # AQI distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["aqi"].dropna(),
        bins=30
    )

    plt.title("AQI Distribution")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_distribution.png",
        dpi=150
    )

    plt.close()

    print("Saved: aqi_distribution.png")

    # --------------------------------------------------------
    # PM2.5 distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["pm2_5"].dropna(),
        bins=30
    )

    plt.title("PM2.5 Distribution")
    plt.xlabel("PM2.5")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "pm25_distribution.png",
        dpi=150
    )

    plt.close()

    print("Saved: pm25_distribution.png")

    # --------------------------------------------------------
    # AQI by hour
    # --------------------------------------------------------

    hourly_aqi = (
        df.groupby("hour")["aqi"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_aqi.index,
        hourly_aqi.values,
        marker="o"
    )

    plt.title("Average AQI by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average AQI")

    plt.xticks(range(24))

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_by_hour.png",
        dpi=150
    )

    plt.close()

    print("Saved: aqi_by_hour.png")

    # --------------------------------------------------------
    # AQI by month
    # --------------------------------------------------------

    monthly_aqi = (
        df.groupby("month")["aqi"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        monthly_aqi.index,
        monthly_aqi.values,
        marker="o"
    )

    plt.title("Average AQI by Month")
    plt.xlabel("Month")
    plt.ylabel("Average AQI")

    plt.xticks(range(1, 13))

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_by_month.png",
        dpi=150
    )

    plt.close()

    print("Saved: aqi_by_month.png")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    validate_dataset(df)

    analyze_aqi(df)

    correlation_analysis(df)

    create_visualizations(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nEDA outputs saved to:")
    print(EDA_DIR)


if __name__ == "__main__":
    main()