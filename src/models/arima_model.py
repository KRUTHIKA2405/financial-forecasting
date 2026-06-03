"""ARIMA baseline model for ticker forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pmdarima as pm
import yaml
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import directional_accuracy, sharpe_ratio


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "arima_results.csv"


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
    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame


def train_test_split_series(series: pd.Series, train_ratio: float, validation_ratio: float, test_ratio: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a series chronologically into train, validation, and test sets."""
    total_length = len(series)
    train_end = int(total_length * train_ratio)
    validation_end = train_end + int(total_length * validation_ratio)

    train_series = series.iloc[:train_end]
    validation_series = series.iloc[train_end:validation_end]
    test_series = series.iloc[validation_end:]
    return train_series, validation_series, test_series


def calculate_directional_accuracy(actuals: pd.Series, predictions: pd.Series) -> float:
    """Measure how often predicted and actual return directions match."""
    if len(actuals) < 2 or len(predictions) < 2:
        return float("nan")

    actual_direction = np.sign(actuals.diff().fillna(0))
    predicted_direction = np.sign(predictions.diff().fillna(0))
    return float((actual_direction == predicted_direction).mean())


def build_metrics(actuals: pd.Series, predictions: pd.Series) -> dict[str, float]:
    """Compute standard forecast metrics."""
    mae = mean_absolute_error(actuals, predictions)
    rmse = mean_squared_error(actuals, predictions, squared=False)
    mape = mean_absolute_percentage_error(actuals, predictions)
    directional_accuracy_value = calculate_directional_accuracy(actuals, predictions)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "directional_accuracy": float(directional_accuracy_value),
        "sharpe_ratio": float(sharpe_ratio(predictions, actuals)),
    }


class ARIMAModel:
    """ARIMA baseline model using auto_arima and walk-forward forecasting."""

    def __init__(self, feature_csv_path: Path | str, config_path: Path = CONFIG_PATH) -> None:
        self.feature_csv_path = Path(feature_csv_path)
        self.config = load_config(config_path)
        self.frame = load_feature_csv(self.feature_csv_path)
        self.ticker = self.feature_csv_path.stem.replace("_features", "")
        self.model = None
        self.train_series: pd.Series | None = None
        self.validation_series: pd.Series | None = None
        self.test_series: pd.Series | None = None
        self.predictions: pd.Series | None = None

    def prepare_series(self) -> pd.Series:
        """Extract the adjusted close series from the feature dataframe."""
        if "Adj Close" not in self.frame.columns:
            raise ValueError(f"Expected an Adj Close column in {self.feature_csv_path}")

        series = self.frame["Adj Close"].astype(float).reset_index(drop=True)
        self.train_series, self.validation_series, self.test_series = train_test_split_series(
            series,
            train_ratio=self.config["split"]["train"],
            validation_ratio=self.config["split"]["validation"],
            test_ratio=self.config["split"]["test"],
        )
        return series

    def fit(self) -> pm.arima.ARIMA:
        """Fit auto_arima on the training portion of the series."""
        if self.train_series is None:
            self.prepare_series()

        self.model = pm.auto_arima(
            self.train_series,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            approximation=False,
        )
        return self.model

    def forecast_walk_forward(self) -> pd.Series:
        """Generate one-step-ahead forecasts across the test set with walk-forward updating."""
        if self.model is None:
            self.fit()
        if self.train_series is None or self.test_series is None:
            self.prepare_series()

        history = self.train_series.tolist()
        predictions: list[float] = []

        for actual_value in self.test_series.tolist():
            forecast = self.model.predict(n_periods=1)
            predicted_value = float(np.asarray(forecast).reshape(-1)[0])
            predictions.append(predicted_value)
            history.append(actual_value)
            self.model.update(actual_value)

        self.predictions = pd.Series(predictions, index=self.test_series.index, name="prediction")
        return self.predictions

    def evaluate(self) -> dict[str, Any]:
        """Return predictions, actuals, and evaluation metrics for the test set."""
        if self.test_series is None:
            self.prepare_series()
        if self.predictions is None:
            self.forecast_walk_forward()

        actuals = self.test_series.reset_index(drop=True)
        predictions = self.predictions.reset_index(drop=True)
        metrics = build_metrics(actuals, predictions)

        return {
            "ticker": self.ticker,
            "predictions": predictions.tolist(),
            "actuals": actuals.tolist(),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "directional_accuracy": metrics["directional_accuracy"],
            "sharpe_ratio": metrics["sharpe_ratio"],
        }

    def run(self) -> dict[str, Any]:
        """Run the full ARIMA workflow for a single ticker."""
        self.prepare_series()
        self.fit()
        self.forecast_walk_forward()
        return self.evaluate()


def run_all_tickers(
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_path: Path = RESULTS_TABLE_PATH,
    config_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Run ARIMA forecasting for all tickers and save the summary results."""
    config = load_config(config_path)
    results: list[dict[str, Any]] = []

    for ticker in config["tickers"]:
        feature_csv = processed_dir / f"{ticker}_features.csv"
        if not feature_csv.exists():
            feature_csv = processed_dir / f"{ticker}.csv"
        if not feature_csv.exists():
            continue

        model = ARIMAModel(feature_csv, config_path=config_path)
        result = model.run()
        result["feature_csv"] = str(feature_csv)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    return results_df


def main() -> None:
    """Run ARIMA for all tickers as a standalone script."""
    run_all_tickers()


if __name__ == "__main__":
    main()