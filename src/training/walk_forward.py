"""Rolling walk-forward retraining for the hybrid CNN-LSTM model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid_cnn_lstm import build_model, build_sequences, load_config, load_feature_csv
from src.training.train_hybrid import split_sequences


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_rolling_window_frame(frame: pd.DataFrame, end_index: int, lookback_days: int = 252) -> pd.DataFrame:
    """Return the trailing rolling window of raw feature rows up to end_index."""
    start_index = max(0, end_index - lookback_days)
    return frame.iloc[start_index:end_index].copy().reset_index(drop=True)


def walk_forward_hybrid(
    feature_csv_path: Path,
    config_path: Path = CONFIG_PATH,
    lookback_days: int = 252,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run a rolling 252-day retraining loop and collect one-step-ahead predictions."""
    config = load_config(config_path)
    frame = load_feature_csv(feature_csv_path)
    ticker = feature_csv_path.stem.replace("_features", "")

    target_column = "Adj Close" if "Adj Close" in frame.columns else "Close"
    predictions: list[float] = []
    actuals: list[float] = []
    prediction_dates: list[pd.Timestamp] = []

    window_size = int(config["window_size"])
    start_index = lookback_days
    total_windows = max(0, len(frame) - start_index)

    for end_index in range(start_index, len(frame)):
        window_number = end_index - start_index + 1
        print(f"{ticker} walk-forward window {window_number}/{total_windows}")

        rolling_frame = build_rolling_window_frame(frame, end_index=end_index, lookback_days=lookback_days)
        if len(rolling_frame) <= window_size:
            continue

        sequences, targets, feature_columns = build_sequences(rolling_frame, config_path=config_path)
        if len(sequences) < 5:
            continue

        x_train, y_train, x_val, y_val, _, _, _, _ = split_sequences(
            sequences,
            targets,
            train_ratio=config["split"]["train"],
            validation_ratio=config["split"]["validation"],
            test_ratio=config["split"]["test"],
        )

        model = build_model(num_features=len(feature_columns), window_size=window_size, config_path=config_path)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
        )

        latest_window = rolling_frame.iloc[-window_size:]
        latest_sequence = latest_window[[column for column in feature_columns]].astype(float).to_numpy(dtype=np.float32)
        latest_sequence = np.expand_dims(latest_sequence, axis=0)
        predicted_value = float(model.predict(latest_sequence, verbose=0).reshape(-1)[0])

        actual_value = float(frame.iloc[end_index][target_column])
        predictions.append(predicted_value)
        actuals.append(actual_value)
        prediction_dates.append(pd.to_datetime(frame.iloc[end_index]["Date"]))

    return {
        "ticker": ticker,
        "dates": prediction_dates,
        "predictions": predictions,
        "actuals": actuals,
        "lookback_days": lookback_days,
        "window_size": window_size,
    }


def run_all_tickers(processed_dir: Path = PROCESSED_DATA_DIR, config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    """Run rolling walk-forward retraining for each ticker."""
    config = load_config(config_path)
    results: list[dict[str, Any]] = []

    for ticker in config["tickers"]:
        feature_csv_path = processed_dir / f"{ticker}_features.csv"
        if not feature_csv_path.exists():
            feature_csv_path = processed_dir / f"{ticker}.csv"
        if not feature_csv_path.exists():
            continue

        results.append(walk_forward_hybrid(feature_csv_path, config_path=config_path))

    return results


def main() -> None:
    """Run the rolling walk-forward workflow for all available tickers."""
    run_all_tickers()


if __name__ == "__main__":
    main()