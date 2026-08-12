from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "historical_features.parquet"
)

EDA_DIR = BASE_DIR / "data" / "eda"

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 60)
    print("LOADING HISTORICAL DATASET")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


# ============================================================
# BASIC DATASET ANALYSIS
# ============================================================

def basic_analysis(df):

    print("\n" + "=" * 60)
    print("BASIC DATASET INFORMATION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDate range:")
    print(df["timestamp"].min())
    print("to")
    print(df["timestamp"].max())


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(df):

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numeric_df = df.select_dtypes(
        include="number"
    )

    print(
        numeric_df.describe().T
    )

    numeric_df.describe().T.to_csv(
        EDA_DIR / "descriptive_statistics.csv"
    )


# ============================================================
# AQI DISTRIBUTION
# ============================================================

def plot_aqi_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["aqi"],
        bins=50
    )

    plt.title(
        "AQI Distribution"
    )

    plt.xlabel("US AQI")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_distribution.png"
    )

    plt.close()


# ============================================================
# AQI OVER TIME
# ============================================================

def plot_aqi_over_time(df):

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp"],
        df["aqi"],
        linewidth=0.8
    )

    plt.title(
        "AQI Over Time — Islamabad"
    )

    plt.xlabel("Date")
    plt.ylabel("US AQI")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_over_time.png"
    )

    plt.close()


# ============================================================
# HOURLY AQI PATTERN
# ============================================================

def analyze_hourly_pattern(df):

    hourly_aqi = (
        df.groupby("hour")["aqi"]
        .mean()
    )

    print("\n" + "=" * 60)
    print("AVERAGE AQI BY HOUR")
    print("=" * 60)

    print(hourly_aqi)

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_aqi.index,
        hourly_aqi.values,
        marker="o"
    )

    plt.title(
        "Average AQI by Hour of Day"
    )

    plt.xlabel("Hour")
    plt.ylabel("Average AQI")

    plt.xticks(range(24))

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_by_hour.png"
    )

    plt.close()


# ============================================================
# MONTHLY AQI PATTERN
# ============================================================

def analyze_monthly_pattern(df):

    monthly_aqi = (
        df.groupby("month")["aqi"]
        .mean()
    )

    print("\n" + "=" * 60)
    print("AVERAGE AQI BY MONTH")
    print("=" * 60)

    print(monthly_aqi)

    plt.figure(figsize=(10, 6))

    plt.plot(
        monthly_aqi.index,
        monthly_aqi.values,
        marker="o"
    )

    plt.title(
        "Average AQI by Month"
    )

    plt.xlabel("Month")
    plt.ylabel("Average AQI")

    plt.xticks(range(1, 13))

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_by_month.png"
    )

    plt.close()


# ============================================================
# WEEKDAY VS WEEKEND
# ============================================================

def analyze_weekend_pattern(df):

    weekend_aqi = (
        df.groupby("is_weekend")["aqi"]
        .mean()
    )

    print("\n" + "=" * 60)
    print("WEEKDAY VS WEEKEND AQI")
    print("=" * 60)

    print(
        "Weekday:",
        weekend_aqi.get(0)
    )

    print(
        "Weekend:",
        weekend_aqi.get(1)
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = (
        numeric_df.corr()["aqi"]
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 60)
    print("FEATURE CORRELATION WITH AQI")
    print("=" * 60)

    print(correlation)

    correlation.to_csv(
        EDA_DIR / "aqi_correlations.csv"
    )

    # Correlation plot
    plt.figure(figsize=(10, 8))

    correlation.drop(
        labels=["aqi"],
        errors="ignore"
    ).sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Feature Correlation with AQI"
    )

    plt.xlabel("Correlation")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_correlations.png"
    )

    plt.close()


# ============================================================
# AQI CHANGE RATE
# ============================================================

def analyze_aqi_change(df):

    change = (
        df["aqi_change_rate"]
        .dropna()
    )

    print("\n" + "=" * 60)
    print("AQI CHANGE RATE")
    print("=" * 60)

    print(
        change.describe()
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        change,
        bins=50
    )

    plt.title(
        "AQI Change Rate Distribution"
    )

    plt.xlabel(
        "AQI Change Rate"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "aqi_change_rate.png"
    )

    plt.close()


# ============================================================
# POLLUTANT DISTRIBUTIONS
# ============================================================

def analyze_pollutants(df):

    pollutants = [
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "o3",
        "so2",
    ]

    for column in pollutants:

        plt.figure(figsize=(10, 6))

        plt.hist(
            df[column].dropna(),
            bins=50
        )

        plt.title(
            f"{column.upper()} Distribution"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            EDA_DIR
            / f"{column}_distribution.png"
        )

        plt.close()


# ============================================================
# OUTLIER SUMMARY
# ============================================================

def outlier_analysis(df):

    numeric_columns = [
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
    ]

    print("\n" + "=" * 60)
    print("OUTLIER SUMMARY")
    print("=" * 60)

    results = []

    for column in numeric_columns:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (df[column] < lower)
            | (df[column] > upper)
        ).sum()

        results.append({
            "feature": column,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": count,
        })

    outlier_df = pd.DataFrame(
        results
    )

    print(outlier_df)

    outlier_df.to_csv(
        EDA_DIR / "outlier_summary.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    basic_analysis(df)

    descriptive_statistics(df)

    plot_aqi_distribution(df)

    plot_aqi_over_time(df)

    analyze_hourly_pattern(df)

    analyze_monthly_pattern(df)

    analyze_weekend_pattern(df)

    correlation_analysis(df)

    analyze_aqi_change(df)

    analyze_pollutants(df)

    outlier_analysis(df)

    print("\n" + "=" * 60)
    print("✅ EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nEDA outputs saved to:\n{EDA_DIR}"
    )


if __name__ == "__main__":
    main()