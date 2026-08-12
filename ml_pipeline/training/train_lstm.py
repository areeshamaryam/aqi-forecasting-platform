from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "ml_pipeline"
    / "data"
    / "processed"
    / "training"
    / "train.parquet"
)

VALIDATION_PATH = (
    BASE_DIR
    / "ml_pipeline"
    / "data"
    / "processed"
    / "training"
    / "validation.parquet"
)

TEST_PATH = (
    BASE_DIR
    / "ml_pipeline"
    / "data"
    / "processed"
    / "training"
    / "test.parquet"
)

RESULTS_DIR = (
    BASE_DIR
    / "ml_pipeline"
    / "data"
    / "processed"
    / "model_results"
)

MODEL_DIR = (
    BASE_DIR
    / "ml_pipeline"
    / "models"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "target_aqi"

SEQUENCE_LENGTH = 24

EPOCHS = 30

BATCH_SIZE = 64


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "aqi_change_rate",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_rolling_mean_3h",
    "aqi_rolling_std_3h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_std_6h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_std_24h",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():

    print("=" * 60)
    print("LOADING LSTM DATASETS")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    return train_df, validation_df, test_df


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(df, feature_columns, target_column, scaler):

    X_values = df[feature_columns].values
    y_values = df[target_column].values

    X_scaled = scaler.transform(X_values)

    X = []
    y = []

    for i in range(SEQUENCE_LENGTH, len(df)):

        X.append(
            X_scaled[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y.append(
            y_values[i]
        )

    return np.array(X), np.array(y)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_model(input_shape):

    model = Sequential(
        [
            LSTM(
                64,
                input_shape=input_shape,
                return_sequences=False,
            ),

            Dropout(0.2),

            Dense(32, activation="relu"),

            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(
        X,
        verbose=0,
    ).flatten()

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y,
        predictions,
    )

    r2 = r2_score(
        y,
        predictions,
    )

    print(f"\n{dataset_name} Performance:")

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df, validation_df, test_df = load_datasets()

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in train_df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # --------------------------------------------------------
    # Prepare scaler
    #
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SCALING FEATURES")
    print("=" * 60)

    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CREATING 24-HOUR SEQUENCES")
    print("=" * 60)

    X_train, y_train = create_sequences(
        train_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        scaler,
    )

    X_validation, y_validation = create_sequences(
        validation_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        scaler,
    )

    X_test, y_test = create_sequences(
        test_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        scaler,
    )

    print(
        f"Training sequences:   {X_train.shape}"
    )

    print(
        f"Validation sequences: {X_validation.shape}"
    )

    print(
        f"Test sequences:       {X_test.shape}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING LSTM MODEL")
    print("=" * 60)

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2],
        )
    )

    model.summary()

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_validation,
            y_validation,
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping
        ],

        verbose=1,
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    validation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "lstm_aqi_model.keras"
    )

    model.save(model_path)

    print(
        f"\n✅ LSTM model saved to:"
    )

    print(model_path)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "validation_rmse": validation_results["rmse"],
                "validation_mae": validation_results["mae"],
                "validation_r2": validation_results["r2"],
                "test_rmse": test_results["rmse"],
                "test_mae": test_results["mae"],
                "test_r2": test_results["r2"],
            }
        ]
    )

    results_path = (
        RESULTS_DIR
        / "lstm_model_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print(
        f"\n✅ LSTM results saved to:"
    )

    print(results_path)

    print("\n" + "=" * 60)
    print("LSTM TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()