"""Hybrid CNN-LSTM model for time series forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    MaxPooling1D,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "hybrid_cnn_lstm_summary.txt"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the project configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_feature_csv(path: Path) -> pd.DataFrame:
    """Load a feature CSV and normalize the date column."""
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"Expected a Date column in {path}")

    frame["Date"] = pd.to_datetime(frame["Date"], utc=False)
    return frame.sort_values("Date").reset_index(drop=True)


def build_sequences(frame: pd.DataFrame, window_size: int | None = None, config_path: Path = CONFIG_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build supervised sequences from a feature dataframe using a rolling window."""
    config = load_config(config_path)
    window_size = window_size or int(config["window_size"])

    target_column = "Adj Close" if "Adj Close" in frame.columns else "Close"
    feature_columns = [
        column
        for column in frame.columns
        if column != "Date" and column != target_column and pd.api.types.is_numeric_dtype(frame[column])
    ]

    if target_column not in frame.columns:
        raise ValueError(f"Expected an {target_column} column in the feature dataframe")

    values = frame[feature_columns].astype(float).to_numpy()
    targets = frame[target_column].astype(float).to_numpy()

    sequences: list[np.ndarray] = []
    sequence_targets: list[float] = []

    for index in range(window_size, len(frame)):
        sequences.append(values[index - window_size : index])
        sequence_targets.append(targets[index])

    return np.asarray(sequences, dtype=np.float32), np.asarray(sequence_targets, dtype=np.float32), feature_columns


class AttentionLayer(Layer):
    """Simple attention layer that learns softmax weights over time steps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.score_weight = self.add_weight(
            name="score_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.score_bias = self.add_weight(
            name="score_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.tensordot(inputs, self.score_weight, axes=1) + self.score_bias
        weights = tf.nn.softmax(scores, axis=1)
        context_vector = tf.reduce_sum(weights * inputs, axis=1)
        return context_vector


def build_model(num_features: int, window_size: int | None = None, config_path: Path = CONFIG_PATH) -> Model:
    """Build the hybrid CNN-LSTM model."""
    config = load_config(config_path)
    window_size = window_size or int(config["window_size"])

    inputs = Input(shape=(window_size, num_features))

    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = AttentionLayer()(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_lstm")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def save_model_summary(model: Model, summary_path: Path = SUMMARY_PATH) -> Path:
    """Save the model architecture summary to disk."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_model_from_file(feature_csv_path: Path, config_path: Path = CONFIG_PATH) -> tuple[Model, np.ndarray, np.ndarray, list[str]]:
    """Load a feature CSV, build sequences, and construct the model."""
    frame = load_feature_csv(feature_csv_path)
    sequences, targets, feature_columns = build_sequences(frame, config_path=config_path)
    model = build_model(num_features=len(feature_columns), config_path=config_path)
    save_model_summary(model)
    return model, sequences, targets, feature_columns


def build_model_for_ticker(ticker: str, processed_dir: Path = PROCESSED_DATA_DIR, config_path: Path = CONFIG_PATH) -> tuple[Model, np.ndarray, np.ndarray, list[str]]:
    """Convenience helper to build the model from a ticker's feature file."""
    feature_csv_path = processed_dir / f"{ticker}_features.csv"
    if not feature_csv_path.exists():
        feature_csv_path = processed_dir / f"{ticker}.csv"
    if not feature_csv_path.exists():
        raise FileNotFoundError(f"No feature CSV found for ticker {ticker}")

    return build_model_from_file(feature_csv_path, config_path=config_path)


def main() -> None:
    """Build the model summary using the first available processed ticker file."""
    processed_files = sorted(PROCESSED_DATA_DIR.glob("*_features.csv"))
    if not processed_files:
        processed_files = sorted(PROCESSED_DATA_DIR.glob("*.csv"))
    if not processed_files:
        raise FileNotFoundError(f"No processed feature files found in {PROCESSED_DATA_DIR}")

    feature_csv_path = processed_files[0]
    model, _, _, feature_columns = build_model_from_file(feature_csv_path)
    print(f"Built hybrid CNN-LSTM model for {feature_csv_path.stem} with {len(feature_columns)} features.")
    print(f"Saved model summary to {SUMMARY_PATH}")
    return model


if __name__ == "__main__":
    main()