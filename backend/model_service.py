"""
Core prediction service used by the FastAPI backend.

Handles:
    - Connecting to Hopsworks and loading the latest
      registered aqi_ridge_hourly model
    - Pulling recent feature data and building the current
      feature row (same lag/rolling/cyclical feature logic
      used during training)
    - Running the 72-hour forecast
    - Computing a SHAP explanation for the 24-hour-ahead
      prediction
    - Flagging hazardous AQI hours using standard US AQI
      breakpoints
"""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap

import hopsworks

from ml_pipeline.feature_store.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)


MODEL_REGISTRY_NAME = "aqi_ridge_hourly"

TARGET_COLUMN = "aqi"

FORECAST_HORIZON_HOURS = 72

NON_FEATURE_COLUMNS = ["city", "timestamp"]

# Standard US AQI category breakpoints, used for hazard alerts
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]

# Any category at or above this index is treated as a
# "hazard alert" worth flagging prominently on the dashboard.
HAZARD_ALERT_THRESHOLD_AQI = 151  # "Unhealthy" and above


def categorize_aqi(aqi_value: float) -> str:
    """
    Categorize an AQI value using standard US AQI breakpoints.

    Uses cumulative (<=) comparisons rather than separate
    (low, high) ranges, so that fractional predicted values
    (e.g. 50.9, 100.4) are never accidentally skipped between
    two integer bounds and mis-classified as "Hazardous" by
    falling through every range.
    """

    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:
        return "Unhealthy"
    if aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def build_feature_row(raw_df: pd.DataFrame):
    """
    Given a chronologically sorted DataFrame of raw feature-
    store rows (city, timestamp, weather, pollutants, aqi,
    hour/day/month/day_of_week/is_weekend), compute the SAME
    lag/rolling/cyclical features used during training, and
    return:
        - the full engineered DataFrame (used as SHAP
          background)
        - the single most recent complete feature row (used
          for prediction)
        - the list of feature column names, in the exact
          order the model expects
    """

    df = raw_df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0)

    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df[TARGET_COLUMN].shift(lag)

    previous_aqi = df[TARGET_COLUMN].shift(1)

    for window in [3, 6, 24]:
        df[f"aqi_rolling_mean_{window}h"] = (
            previous_aqi.rolling(window=window).mean()
        )
        df[f"aqi_rolling_std_{window}h"] = (
            previous_aqi.rolling(window=window).std()
        )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    engineered_df = df.dropna(subset=feature_columns).reset_index(
        drop=True
    )

    if len(engineered_df) == 0:
        raise ValueError(
            "Not enough recent history to compute features "
            "(need at least 24+ hours of prior data)."
        )

    current_row = engineered_df.iloc[[-1]]
    current_timestamp = current_row["timestamp"].values[0]

    return engineered_df, current_row, feature_columns, current_timestamp


class ModelService:

    def __init__(self):
        self._model = None
        self._feature_columns = None
        self._project = None
        self.model_version = None

    def is_ready(self) -> bool:
        return self._model is not None

    def load(self):
        """
        Connect to Hopsworks once, download the latest
        registered model, and keep the connection open for
        reuse across requests.
        """

        self._project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY,
        )

        model_registry = self._project.get_model_registry()

        # NOTE: passing version=None to get_model() does NOT
        # return the latest version - Hopsworks silently
        # defaults to version 1 instead (confirmed via
        # VersionWarning during testing). We must explicitly
        # find and request the highest version number.
        all_versions = model_registry.get_models(
            name=MODEL_REGISTRY_NAME,
        )

        if not all_versions:
            raise RuntimeError(
                f"No versions found for model "
                f"'{MODEL_REGISTRY_NAME}' in the registry."
            )

        model_meta = max(all_versions, key=lambda m: m.version)

        self.model_version = model_meta.version

        model_dir = model_meta.download()

        # The registered artifact is a single .pkl file inside
        # this downloaded directory.
        import os

        pkl_files = [
            f for f in os.listdir(model_dir) if f.endswith(".pkl")
        ]

        if not pkl_files:
            raise RuntimeError(
                f"No .pkl file found in downloaded model dir: "
                f"{model_dir}"
            )

        model_path = os.path.join(model_dir, pkl_files[0])

        self._model = joblib.load(model_path)

    def _fetch_recent_features(self) -> pd.DataFrame:
        """
        Pull the most recent ~500 hours of raw feature rows
        from the Hopsworks feature store. 500 hours is far
        more than the 24 hours strictly needed for lag/rolling
        features, giving us a reasonable SHAP background
        sample too.

        This read has been observed to intermittently fail
        with a FlightUnavailableError on some networks (the
        Arrow Flight service uses a dedicated port, 5005,
        which can be unreliable through certain ISPs/routers
        even when not permanently blocked). We retry a few
        times with a short delay before giving up, since the
        exact same call has been observed to succeed on a
        later attempt with no code changes.
        """

        import time

        feature_store = self._project.get_feature_store()

        feature_group = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )

        max_attempts = 3
        last_error = None

        for attempt in range(1, max_attempts + 1):

            try:
                df = feature_group.read(
                    read_options={"use_hive": True},
                )
                break  # success, stop retrying

            except Exception as exc:
                last_error = exc
                print(
                    f"⚠️  Feature store read failed "
                    f"(attempt {attempt}/{max_attempts}): {exc}"
                )

                if attempt < max_attempts:
                    wait_seconds = attempt * 5  # 5s, then 10s
                    print(f"Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)

        else:
            # Loop completed without a successful break -
            # every attempt failed.
            raise RuntimeError(
                f"Could not read from Hopsworks Feature Store "
                f"after {max_attempts} attempts. This is likely "
                f"an intermittent network issue reaching "
                f"Hopsworks' Arrow Flight service (port 5005). "
                f"Last error: {last_error}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Keep only the most recent window
        df = df.tail(500).reset_index(drop=True)

        return df

    def predict(self) -> dict:

        raw_df = self._fetch_recent_features()

        (
            engineered_df,
            current_row,
            feature_columns,
            current_timestamp,
        ) = build_feature_row(raw_df)

        X_current = current_row[feature_columns]

        # ------------------------------------------------
        # 72-hour forecast
        # ------------------------------------------------

        predictions = self._model.predict(X_current)[0]
        predictions = np.clip(predictions, 0, None)  # AQI can't be negative

        current_ts = pd.Timestamp(current_timestamp)

        forecast = []

        for h in range(1, FORECAST_HORIZON_HOURS + 1):

            forecast_time = current_ts + pd.Timedelta(hours=h)

            aqi_value = float(predictions[h - 1])

            forecast.append(
                {
                    "hour_offset": h,
                    "timestamp": forecast_time.isoformat(),
                    "predicted_aqi": round(aqi_value, 1),
                    "category": categorize_aqi(aqi_value),
                }
            )

        # ------------------------------------------------
        # Hazard alerts: any hour predicted at "Unhealthy"
        # (AQI 151+) or worse
        # ------------------------------------------------

        hazard_hours = [
            entry
            for entry in forecast
            if entry["predicted_aqi"] >= HAZARD_ALERT_THRESHOLD_AQI
        ]

        peak_entry = max(forecast, key=lambda e: e["predicted_aqi"])

        # ------------------------------------------------
        # SHAP explanation for the 24-hour-ahead prediction
        # (chosen as the most human-relevant single horizon
        # to explain - "why is tomorrow's AQI predicted this
        # way")
        # ------------------------------------------------

        explanation_horizon = 24
        target_idx = explanation_horizon - 1

        background_sample = engineered_df[feature_columns].iloc[:-1]

        # Ridge stores one coefficient row + intercept per
        # output. LinearExplainer accepts (coef, intercept)
        # directly for a single output, avoiding the need to
        # wrap the whole multi-output model.
        explainer = shap.LinearExplainer(
            (
                self._model.coef_[target_idx],
                self._model.intercept_[target_idx],
            ),
            background_sample,
            feature_names=feature_columns,
        )

        shap_values = explainer.shap_values(X_current)[0]

        contributions = sorted(
            zip(feature_columns, shap_values),
            key=lambda pair: -abs(pair[1]),
        )[:8]

        shap_explanation = [
            {
                "feature": name,
                "contribution": round(float(value), 3),
            }
            for name, value in contributions
        ]

        return {
            "city": "Islamabad",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "last_known_timestamp": current_ts.isoformat(),
            "current_aqi": float(current_row["aqi"].values[0]),
            "forecast": forecast,
            "peak": peak_entry,
            "hazard_alert": {
                "has_alert": len(hazard_hours) > 0,
                "hours": hazard_hours,
            },
            "explanation": {
                "horizon_hours": explanation_horizon,
                "top_features": shap_explanation,
            },
        }