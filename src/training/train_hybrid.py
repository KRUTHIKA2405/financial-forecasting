"""Train the hybrid CNN-LSTM model for each ticker."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid_cnn_lstm import build_model, build_sequences, load_feature_csv, load_config
from src.evaluation.metrics import directional_accuracy, mae, mape, rmse, sharpe_ratio


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
HYBRID_RESULTS_PATH = TABLES_DIR / "hybrid_results.csv"


def load_processed_feature_csv(path: Path) -> pd.DataFrame:
    """Load a processed feature CSV for training."""
    return load_feature_csv(path)


def split_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequence samples chronologically into train, validation, and test sets."""
    total_samples = len(sequences)
    train_end = int(total_samples * train_ratio)
    validation_end = train_end + int(total_samples * validation_ratio)

    x_train = sequences[:train_end]
    y_train = targets[:train_end]
    x_val = sequences[train_end:validation_end]
    y_val = targets[train_end:validation_end]
    x_test = sequences[validation_end:]
    y_test = targets[validation_end:]
    return x_train, y_train, x_val, y_val, x_test, y_test


def build_training_data(feature_csv_path: Path, config_path: Path = CONFIG_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Load a feature CSV and create model-ready sequence splits."""
    config = load_config(config_path)
    frame = load_processed_feature_csv(feature_csv_path)
    sequences, targets, feature_columns = build_sequences(frame, config_path=config_path)

    x_train, y_train, x_val, y_val, x_test, y_test = split_sequences(
        sequences,
        targets,
        train_ratio=config["split"]["train"],
        validation_ratio=config["split"]["validation"],
        test_ratio=config["split"]["test"],
    )
    return x_train, y_train, x_val, y_val, x_test, y_test, feature_columns, int(config["window_size"])


def save_training_curve(history: tf.keras.callbacks.History, ticker: str, output_dir: Path = FIGURES_DIR) -> Path:
    """Plot and save the training and validation loss curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{ticker}_hybrid_loss.png"

    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get("loss", []), label="Training Loss")
    plt.plot(history.history.get("val_loss", []), label="Validation Loss")
    plt.title(f"Hybrid CNN-LSTM Loss Curve - {ticker}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return figure_path


def train_hybrid_model(
    feature_csv_path: Path,
    config_path: Path = CONFIG_PATH,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Train the hybrid CNN-LSTM model for a single ticker and return test predictions."""
    config = load_config(config_path)
    ticker = feature_csv_path.stem.replace("_features", "")

    print(f"\n=== Training hybrid model for {ticker} ===")

    x_train, y_train, x_val, y_val, x_test, y_test, feature_columns, window_size = build_training_data(
        feature_csv_path,
        config_path=config_path,
    )

    model = build_model(num_features=len(feature_columns), window_size=window_size, config_path=config_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / f"{ticker}_hybrid.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(filepath=best_model_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )

    print(f"=== Finished training {ticker} at epoch {len(history.history.get('loss', []))} ===")

    save_training_curve(history, ticker)

    test_predictions = model.predict(x_test, verbose=0).reshape(-1)
    results_row = {
        "ticker": ticker,
        "feature_csv": str(feature_csv_path),
        "model_path": str(best_model_path),
        "predictions": test_predictions.tolist(),
        "actuals": y_test.tolist(),
        "mae": float(mae(test_predictions, y_test)),
        "rmse": float(rmse(test_predictions, y_test)),
        "mape": float(mape(test_predictions, y_test)),
        "directional_accuracy": float(directional_accuracy(test_predictions, y_test)),
        "sharpe_ratio": float(sharpe_ratio(test_predictions, y_test)),
    }

    return {
        **results_row,
        "history": history.history,
        "x_test": x_test,
        "y_test": y_test,
        "predictions": test_predictions,
        "feature_columns": feature_columns,
    }


def run_all_tickers(processed_dir: Path = PROCESSED_DATA_DIR, config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    """Train the hybrid model for every available ticker and return test predictions."""
    config = load_config(config_path)
    results: list[dict[str, Any]] = []

    for ticker in config["tickers"]:
        feature_csv_path = processed_dir / f"{ticker}_features.csv"
        if not feature_csv_path.exists():
            feature_csv_path = processed_dir / f"{ticker}.csv"
        if not feature_csv_path.exists():
            continue

        result = train_hybrid_model(feature_csv_path, config_path=config_path)
        results.append(result)

    if results:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(
            [
                {
                    "ticker": row["ticker"],
                    "feature_csv": row["feature_csv"],
                    "model_path": row["model_path"],
                    "predictions": row["predictions"].tolist() if hasattr(row["predictions"], "tolist") else row["predictions"],
                    "actuals": row["actuals"].tolist() if hasattr(row["actuals"], "tolist") else row["actuals"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mape": row["mape"],
                    "directional_accuracy": row["directional_accuracy"],
                    "sharpe_ratio": row["sharpe_ratio"],
                }
                for row in results
            ]
        )
        results_df.to_csv(HYBRID_RESULTS_PATH, index=False)

    return results


def main() -> None:
    """Train the hybrid model for all tickers in the configuration."""
    run_all_tickers()


if __name__ == "__main__":
    main()