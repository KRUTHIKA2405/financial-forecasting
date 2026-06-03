"""Ablation study for the hybrid CNN-LSTM architecture."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import directional_accuracy
from src.models.hybrid_cnn_lstm import AttentionLayer, build_sequences, load_config, load_feature_csv
from src.training.train_hybrid import split_sequences


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "ablation_results.csv"


def load_aapl_frame(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """Load the AAPL feature dataframe for the ablation study."""
    feature_csv_path = processed_dir / "AAPL_features.csv"
    if not feature_csv_path.exists():
        feature_csv_path = processed_dir / "AAPL.csv"
    if not feature_csv_path.exists():
        raise FileNotFoundError("AAPL feature CSV was not found in data/processed/")
    return load_feature_csv(feature_csv_path)


def build_ablation_data(config_path: Path = CONFIG_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Prepare train/validation/test sequences for AAPL."""
    config = load_config(config_path)
    frame = load_aapl_frame()
    sequences, targets, feature_columns = build_sequences(frame, config_path=config_path)
    x_train, y_train, x_val, y_val, x_test, y_test = split_sequences(
        sequences,
        targets,
        train_ratio=config["split"]["train"],
        validation_ratio=config["split"]["validation"],
        test_ratio=config["split"]["test"],
    )
    return x_train, y_train, x_val, y_val, x_test, y_test, feature_columns, int(config["window_size"])


def build_cnn_only_model(window_size: int, num_features: int) -> Model:
    """Build a CNN-only ablation model."""
    inputs = Input(shape=(window_size, num_features))
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name="cnn_only")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_lstm_only_model(window_size: int, num_features: int) -> Model:
    """Build an LSTM-only ablation model."""
    inputs = Input(shape=(window_size, num_features))
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(64)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs, name="lstm_only")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_cnn_lstm_model(window_size: int, num_features: int, attention: bool = False) -> Model:
    """Build the CNN+LSTM model with or without attention."""
    inputs = Input(shape=(window_size, num_features))
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True if attention else False)(x)
    if attention:
        x = AttentionLayer()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    name = "cnn_lstm_attention" if attention else "cnn_lstm"
    model = Model(inputs, outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute the standard regression metrics used in the ablation table."""
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    mse = mean_squared_error(actuals, predictions)
    return {
        "mae": float(mean_absolute_error(actuals, predictions)),
        "rmse": float(np.sqrt(mse)),
        "mape": float(mean_absolute_percentage_error(actuals, predictions)),
        "directional_accuracy": float(directional_accuracy(predictions, actuals)),
    }


def train_and_evaluate_variant(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Train a model variant and evaluate it on the test set."""
    callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )

    predictions = model.predict(x_test, verbose=0).reshape(-1)
    metrics = compute_metrics(y_test, predictions)
    return {
        "predictions": predictions,
        "actuals": y_test,
        **metrics,
    }


def run_ablation_study(config_path: Path = CONFIG_PATH, results_path: Path = RESULTS_TABLE_PATH) -> pd.DataFrame:
    """Train and compare four hybrid variants on AAPL."""
    x_train, y_train, x_val, y_val, x_test, y_test, feature_columns, window_size = build_ablation_data(config_path=config_path)
    num_features = len(feature_columns)

    variants = {
        "CNN only": build_cnn_only_model(window_size, num_features),
        "LSTM only": build_lstm_only_model(window_size, num_features),
        "CNN+LSTM without attention": build_cnn_lstm_model(window_size, num_features, attention=False),
        "CNN+LSTM+Attention": build_cnn_lstm_model(window_size, num_features, attention=True),
    }

    rows: list[dict[str, Any]] = []
    for variant_name, model in variants.items():
        result = train_and_evaluate_variant(model, x_train, y_train, x_val, y_val, x_test, y_test)
        rows.append(
            {
                "variant": variant_name,
                "ticker": "AAPL",
                "mae": result["mae"],
                "rmse": result["rmse"],
                "mape": result["mape"],
                "directional_accuracy": result["directional_accuracy"],
            }
        )

    results_df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(results_df.to_string(index=False))
    return results_df


def main() -> None:
    """Run the ablation study for AAPL."""
    run_ablation_study()


if __name__ == "__main__":
    main()