from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "training"
)

TRAIN_PATH = TRAIN_DIR / "train.parquet"
VALIDATION_PATH = TRAIN_DIR / "validation.parquet"

MODEL_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
#
# We train ONE multi-output Ridge model that predicts AQI
# for every hour from t+1 to t+72 (the full next 3 days) in
# a single .predict() call. Ridge was chosen because it
# outperformed XGBoost and Random Forest on TEST data at
# every horizon we evaluated (24h/48h/72h checkpoints),
# generalizing better than the tree-based models on this
# dataset size.
# ============================================================

FORECAST_HORIZON_HOURS = 72

TARGET_COLUMNS = [
    f"target_aqi_h{h}"
    for h in range(1, FORECAST_HORIZON_HOURS + 1)
]

NON_FEATURE_COLUMNS = [
    "city",
    "timestamp",
] + TARGET_COLUMNS

MODEL_FILENAME = "ridge_aqi_model_72h_hourly.pkl"
REGISTRY_NAME = "aqi_ridge_hourly"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING FINAL TRAINING DATA")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)

    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")

    return train_df, validation_df


# ============================================================
# PREPARE FEATURES / MULTI-OUTPUT TARGET
# ============================================================

def prepare_features(train_df, validation_df):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMNS]   # shape (n, 72)

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET_COLUMNS]

    print(f"\nNumber of features: {len(feature_columns)}")
    print(f"Number of target hours: {len(TARGET_COLUMNS)}")

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_model(X_train, y_train):

    print("\n" + "=" * 60)
    print("TRAINING FINAL MULTI-OUTPUT RIDGE MODEL (72 HOURS)")
    print("=" * 60)

    # Ridge natively supports multi-output regression:
    # a single model learns to predict all 72 hourly targets
    # at once, sharing information across nearby horizons.
    model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)

    print("✅ Training complete.")

    return model


# ============================================================
# EVALUATE FINAL MODEL
# ============================================================

def evaluate_model(model, X_validation, y_validation):

    predictions = model.predict(X_validation)   # shape (n, 72)

    # Overall metrics across all 72 hours combined
    overall_rmse = np.sqrt(
        mean_squared_error(y_validation, predictions)
    )
    overall_mae = mean_absolute_error(y_validation, predictions)
    overall_r2 = r2_score(y_validation, predictions)

    print("\nOverall performance (all 72 hours combined):")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"R²:   {overall_r2:.4f}")

    # Per-hour metrics at key checkpoints, useful for the report
    print("\nPer-hour performance at key checkpoints:")

    checkpoint_metrics = {}

    for h in [1, 6, 12, 24, 48, 72]:

        col_index = h - 1

        y_true_h = y_validation.iloc[:, col_index]
        y_pred_h = predictions[:, col_index]

        rmse_h = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        mae_h = mean_absolute_error(y_true_h, y_pred_h)
        r2_h = r2_score(y_true_h, y_pred_h)

        print(
            f"  hour {h:>2}: "
            f"RMSE={rmse_h:.3f}  MAE={mae_h:.3f}  R²={r2_h:.3f}"
        )

        checkpoint_metrics[f"rmse_h{h}"] = float(rmse_h)
        checkpoint_metrics[f"mae_h{h}"] = float(mae_h)
        checkpoint_metrics[f"r2_h{h}"] = float(r2_h)

    metrics = {
        "rmse": float(overall_rmse),
        "mae": float(overall_mae),
        "r2": float(overall_r2),
    }

    metrics.update(checkpoint_metrics)

    return metrics


# ============================================================
# SAVE MODEL ARTIFACT
# ============================================================

def save_model(model, model_path):

    joblib.dump(model, model_path)

    print(f"✅ Model saved to:\n{model_path}")


# ============================================================
# REGISTER MODEL IN HOPSWORKS
# ============================================================

def register_model(metrics, feature_columns, model_path):

    print("\n" + "=" * 60)
    print(f"REGISTERING MODEL IN HOPSWORKS: {REGISTRY_NAME}")
    print("=" * 60)

    import hopsworks

    from ..feature_store.config import (
        HOPSWORKS_API_KEY,
        HOPSWORKS_PROJECT,
    )

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
    )

    print("✅ Connected to Hopsworks Model Registry!")

    model_registry = project.get_model_registry()

    # Only pass simple top-level metrics to Hopsworks
    # (it expects scalar values per metric key).
    registry_metrics = {
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "rmse_h24": metrics["rmse_h24"],
        "rmse_h48": metrics["rmse_h48"],
        "rmse_h72": metrics["rmse_h72"],
        "r2_h24": metrics["r2_h24"],
        "r2_h48": metrics["r2_h48"],
        "r2_h72": metrics["r2_h72"],
    }

    registered_model = (
        model_registry.python.create_model(
            name=REGISTRY_NAME,
            metrics=registry_metrics,
            description=(
                "Multi-output Ridge Regression model predicting "
                "hourly AQI for the next 72 hours (3 days) in a "
                "single prediction call. Selected over XGBoost "
                "and Random Forest after those models showed "
                "weaker generalization on the test set at every "
                "forecast horizon evaluated."
            ),
        )
    )

    # IMPORTANT: model_path must be a string, not a
    # pathlib.Path object, or hsml raises:
    #   AttributeError: 'WindowsPath' object has no
    #   attribute 'startswith'
    registered_model.save(str(model_path))

    print(f"\n✅ {REGISTRY_NAME} successfully registered in Hopsworks!")
    print(f"Features registered: {len(feature_columns)}")
    print(f"Overall RMSE: {metrics['rmse']:.4f}")
    print(f"Overall MAE: {metrics['mae']:.4f}")
    print(f"Overall R²: {metrics['r2']:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():

    train_df, validation_df = load_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_features(train_df, validation_df)

    model = train_model(X_train, y_train)

    print("\n" + "=" * 60)
    print("FINAL MODEL VALIDATION")
    print("=" * 60)

    metrics = evaluate_model(model, X_validation, y_validation)

    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACT")
    print("=" * 60)

    model_path = MODEL_DIR / MODEL_FILENAME

    save_model(model, model_path)

    register_model(metrics, feature_columns, model_path)

    print("\n" + "=" * 60)
    print("✅ MODEL REGISTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()