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
    / "data"
    / "processed"
    / "models"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
#
# This trains ONE LSTM model that predicts AQI for all 72
# hours (t+1 ... t+72) in a single forward pass, using the
# same target structure as prepare_training_data.py and
# regsiter_model.py (Ridge). This is used as the "deep
# learning" comparison point required by the project brief
# ("support multiple forecasting models from statistical to
# deep learning") — it is NOT expected to beat Ridge, since
# the dataset (~12k sequences) is small for an LSTM and the
# engineered lag/rolling features already capture most of
# the time-series signal directly. Keep these results (and
# the comparison against Ridge) in your report.
# ============================================================

FORECAST_HORIZON_HOURS = 72

TARGET_COLUMNS = [
    f"target_aqi_h{h}"
    for h in range(1, FORECAST_HORIZON_HOURS + 1)
]

SEQUENCE_LENGTH = 24   # use the past 24 hours to predict the next 72

EPOCHS = 30

BATCH_SIZE = 64


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
# DETERMINE FEATURE COLUMNS
# ============================================================

def get_feature_columns(train_df):
    """
    Use every column that is not an identifier/timestamp and
    not one of the 72 target columns. This keeps the LSTM's
    inputs consistent with whatever prepare_training_data.py
    produced, instead of hard-coding a feature list that could
    drift out of sync.
    """

    non_feature_columns = [
        "city",
        "timestamp",
    ] + TARGET_COLUMNS

    feature_columns = [
        column
        for column in train_df.columns
        if column not in non_feature_columns
    ]

    return feature_columns


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(df, feature_columns, scaler):

    X_values = df[feature_columns].values
    y_values = df[TARGET_COLUMNS].values

    X_scaled = scaler.transform(X_values)

    X = []
    y = []

    for i in range(SEQUENCE_LENGTH, len(df)):

        X.append(
            X_scaled[i - SEQUENCE_LENGTH:i]
        )

        y.append(
            y_values[i]
        )

    return np.array(X), np.array(y)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_model(input_shape, output_size):

    model = Sequential(
        [
            LSTM(
                64,
                input_shape=input_shape,
                return_sequences=False,
            ),

            Dropout(0.2),

            Dense(32, activation="relu"),

            # One output neuron per forecast hour (72 total)
            Dense(output_size),
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
    )

    overall_rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    overall_mae = mean_absolute_error(y, predictions)

    overall_r2 = r2_score(y, predictions)

    print(f"\n{dataset_name} performance (all 72 hours combined):")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"R²:   {overall_r2:.4f}")

    checkpoint_metrics = {}

    print(f"\n{dataset_name} performance at key checkpoints:")

    for h in [1, 6, 12, 24, 48, 72]:

        col_index = h - 1

        rmse_h = np.sqrt(
            mean_squared_error(y[:, col_index], predictions[:, col_index])
        )

        mae_h = mean_absolute_error(
            y[:, col_index], predictions[:, col_index]
        )

        r2_h = r2_score(
            y[:, col_index], predictions[:, col_index]
        )

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
# MAIN
# ============================================================

def main():

    train_df, validation_df, test_df = load_datasets()

    feature_columns = get_feature_columns(train_df)

    print(f"\nNumber of features: {len(feature_columns)}")
    print(f"Number of target hours: {len(TARGET_COLUMNS)}")

    # --------------------------------------------------------
    # Scale features. Fit ONLY on training data to avoid
    # leaking validation/test information into the scaler.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SCALING FEATURES")
    print("=" * 60)

    scaler = StandardScaler()

    scaler.fit(train_df[feature_columns])

    # --------------------------------------------------------
    # Build 24-hour input sequences for every split
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CREATING 24-HOUR INPUT SEQUENCES")
    print("=" * 60)

    X_train, y_train = create_sequences(train_df, feature_columns, scaler)
    X_validation, y_validation = create_sequences(
        validation_df, feature_columns, scaler
    )
    X_test, y_test = create_sequences(test_df, feature_columns, scaler)

    print(f"Training sequences:   {X_train.shape} -> targets {y_train.shape}")
    print(f"Validation sequences: {X_validation.shape}")
    print(f"Test sequences:       {X_test.shape}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUILDING LSTM MODEL (72-HOUR MULTI-OUTPUT)")
    print("=" * 60)

    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_size=len(TARGET_COLUMNS),
    )

    model.summary()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING LSTM")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    # --------------------------------------------------------
    # Evaluate on validation and test
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATING LSTM")
    print("=" * 60)

    validation_metrics = evaluate_model(
        model, X_validation, y_validation, "Validation"
    )

    test_metrics = evaluate_model(
        model, X_test, y_test, "Test"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / "lstm_aqi_model_72h_hourly.keras"

    model.save(model_path)

    print(f"\n✅ LSTM model saved to:\n{model_path}")

    # --------------------------------------------------------
    # Save results for the report
    # --------------------------------------------------------

    results_row = {"model": "LSTM"}

    for key, value in validation_metrics.items():
        results_row[f"validation_{key}"] = value

    for key, value in test_metrics.items():
        results_row[f"test_{key}"] = value

    results_df = pd.DataFrame([results_row])

    results_path = RESULTS_DIR / "lstm_model_results.csv"

    results_df.to_csv(results_path, index=False)

    print(f"\n✅ LSTM results saved to:\n{results_path}")

    print("\n" + "=" * 60)
    print("✅ LSTM TRAINING COMPLETED")
    print("=" * 60)
    print(
        "\nNote: compare test_r2 / test_rmse against the Ridge "
        "model's results (from regsiter_model.py) for your "
        "report's model-comparison section."
    )


if __name__ == "__main__":
    main()