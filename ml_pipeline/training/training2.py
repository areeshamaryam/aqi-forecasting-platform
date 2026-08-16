from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
# ============================================================
# PROJECT PATHS
# Multi-horizon version: trains Ridge / RF / XGBoost for
# EACH forecast horizon (24h, 48h, 72h) instead of a single
# 1-hour-ahead target.
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
TEST_PATH = TRAIN_DIR / "test.parquet"

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_results"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Multi-horizon targets, must match prepare_training_data.py
TARGET_COLUMNS = [
    "target_aqi_24h",
    "target_aqi_48h",
    "target_aqi_72h",
]

# Columns that should NOT be used as numerical ML features
NON_FEATURE_COLUMNS = [
    "city",
    "timestamp",
] + TARGET_COLUMNS


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():
    """
    Load chronological training, validation and test datasets.
    """

    print("=" * 60)
    print("LOADING TRAINING DATASETS")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    return train_df, validation_df, test_df


# ============================================================
# PREPARE FEATURES AND TARGET (for one horizon)
# ============================================================

def prepare_features(train_df, validation_df, test_df, target_column):
    """
    Separate features and target for a SINGLE horizon while
    keeping the chronological train/validation/test split intact.
    """

    feature_columns = [
        column
        for column in train_df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    horizon_label,
):
    """
    Train a model and evaluate it on validation and test data.
    """

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}  [{horizon_label}]")
    print("=" * 60)

    print("Training model...")

    # ----------------------------------------------------------
    # XGBoost uses early stopping against the validation set.
    # Ridge / Random Forest don't accept eval_set, so they are
    # trained the normal way.
    # ----------------------------------------------------------

    if model_name == "XGBoost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )

        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        if best_iteration is not None:
            print(
                f"Best iteration (early stopping): "
                f"{best_iteration}"
            )

    else:
        model.fit(
            X_train,
            y_train,
        )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions,
        )
    )

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    validation_r2 = r2_score(
        y_validation,
        validation_predictions,
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions,
        )
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    test_r2 = r2_score(
        y_test,
        test_predictions,
    )

    print("\nValidation Performance:")
    print(f"RMSE: {validation_rmse:.4f}")
    print(f"MAE:  {validation_mae:.4f}")
    print(f"R²:   {validation_r2:.4f}")

    print("\nTest Performance:")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE:  {test_mae:.4f}")
    print(f"R²:   {test_r2:.4f}")

    return {
        "horizon": horizon_label,
        "model": model_name,

        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae,
        "validation_r2": validation_r2,

        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():
    """
    Create the classical ML models for experimentation.
    A fresh set is created per horizon so no model state
    leaks between horizons.
    """

    models = {

        # ----------------------------------------------------
        # 1. Ridge Regression
        # ----------------------------------------------------

        "Ridge Regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0
                    ),
                ),
            ]
        ),

        # ----------------------------------------------------
        # 2. Random Forest
        # ----------------------------------------------------

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),

        # ----------------------------------------------------
        # 3. XGBoost (with early stopping)
        # ----------------------------------------------------

        "XGBoost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
        ),
    }

    return models


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    """
    Save model comparison results (across all horizons) to CSV.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    # Sort by horizon, then by test RMSE within each horizon
    results_df = results_df.sort_values(
        by=["horizon", "test_rmse"],
        ascending=True,
    )

    output_path = (
        RESULTS_DIR
        / "classical_model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (ALL HORIZONS)")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\n✅ Results saved to:\n{output_path}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = load_datasets()

    all_results = []

    # --------------------------------------------------------
    # Loop over each forecast horizon
    # --------------------------------------------------------

    for target_column in TARGET_COLUMNS:

        horizon_label = target_column.replace("target_aqi_", "")

        print("\n" + "#" * 60)
        print(f"# HORIZON: {horizon_label}")
        print("#" * 60)

        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            feature_columns,
        ) = prepare_features(
            train_df,
            validation_df,
            test_df,
            target_column,
        )

        models = create_models()

        for model_name, model in models.items():

            result = evaluate_model(
                model_name,
                model,
                X_train,
                y_train,
                X_validation,
                y_validation,
                X_test,
                y_test,
                horizon_label,
            )

            all_results.append(result)

    # --------------------------------------------------------
    # Save comparison across all horizons and models
    # --------------------------------------------------------

    save_results(
        all_results
    )

    print("\n" + "=" * 60)
    print("✅ MULTI-HORIZON MODEL EXPERIMENTATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()