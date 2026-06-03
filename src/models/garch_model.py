"""GARCH forecasting model for ticker returns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pmdarima as pm
import yaml
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import sharpe_ratio


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "garch_results.csv"


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


def train_test_split_series(
    series: pd.Series,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split a series chronologically into train, validation, and test sets."""
    total_length = len(series)
    train_end = int(total_length * train_ratio)
    validation_end = train_end + int(total_length * validation_ratio)

    train_series = series.iloc[:train_end]
    validation_series = series.iloc[train_end:validation_end]
    test_series = series.iloc[validation_end:]
    return train_series, validation_series, test_series


def calculate_directional_accuracy(actuals: pd.Series, predictions: pd.Series) -> float:
    """Measure how often predicted and actual directions match."""
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


class GARCHModel:
    """ARIMA-GARCH hybrid model built on adjusted close log returns."""

    def __init__(self, feature_csv_path: Path | str, config_path: Path = CONFIG_PATH) -> None:
        self.feature_csv_path = Path(feature_csv_path)
        self.config = load_config(config_path)
        self.frame = load_feature_csv(self.feature_csv_path)
        self.ticker = self.feature_csv_path.stem.replace("_features", "")
        self.arima_model = None
        self.garch_model = None
        self.log_returns: pd.Series | None = None
        self.train_returns: pd.Series | None = None
        self.validation_returns: pd.Series | None = None
        self.test_returns: pd.Series | None = None
        self.mean_forecast: pd.Series | None = None
        self.volatility_forecast: pd.Series | None = None
        self.combined_forecast: pd.DataFrame | None = None

    def prepare_returns(self) -> pd.Series:
        """Extract adjusted close log returns from the feature dataframe."""
        if "Adj Close" not in self.frame.columns:
            raise ValueError(f"Expected an Adj Close column in {self.feature_csv_path}")

        adj_close = self.frame["Adj Close"].astype(float).reset_index(drop=True)
        log_returns = np.log(adj_close / adj_close.shift(1)).dropna().reset_index(drop=True)
        self.log_returns = log_returns
        self.train_returns, self.validation_returns, self.test_returns = train_test_split_series(
            log_returns,
            train_ratio=self.config["split"]["train"],
            validation_ratio=self.config["split"]["validation"],
            test_ratio=self.config["split"]["test"],
        )
        return log_returns

    def fit_arima_mean(self) -> pm.arima.ARIMA:
        """Fit an ARIMA model to the training log returns."""
        if self.train_returns is None:
            self.prepare_returns()

        self.arima_model = pm.auto_arima(
            self.train_returns,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            approximation=False,
        )
        return self.arima_model

    def fit_garch_volatility(self) -> Any:
        """Fit a GARCH(1,1) model to the ARIMA residuals."""
        if self.train_returns is None:
            self.prepare_returns()
        if self.arima_model is None:
            self.fit_arima_mean()

        arima_fitted = pd.Series(self.arima_model.predict_in_sample(), index=self.train_returns.index)
        residuals = self.train_returns - arima_fitted
        scaled_residuals = residuals * 100

        self.garch_model = arch_model(
            scaled_residuals,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        ).fit(disp="off")
        return self.garch_model

    def forecast_walk_forward(self) -> pd.DataFrame:
        """Generate one-step-ahead mean forecasts and test-horizon volatility forecasts."""
        if self.train_returns is None or self.test_returns is None:
            self.prepare_returns()
        if self.arima_model is None:
            self.fit_arima_mean()
        if self.garch_model is None:
            self.fit_garch_volatility()

        mean_predictions: list[float] = []
        actuals = self.test_returns.copy()
        volatility_path = self.garch_model.forecast(horizon=len(actuals), reindex=False).variance.iloc[-1]
        volatility_predictions = np.sqrt(np.maximum(volatility_path.to_numpy(dtype=float), 0.0)) / 100.0

        for actual_value in actuals.tolist():
            arima_forecast = self.arima_model.predict(n_periods=1)
            arima_mean = float(np.asarray(arima_forecast).reshape(-1)[0])
            mean_predictions.append(arima_mean)
            self.arima_model.update(actual_value)

        self.mean_forecast = pd.Series(mean_predictions, index=actuals.index, name="arima_mean_forecast")
        self.volatility_forecast = pd.Series(volatility_predictions, index=actuals.index, name="garch_volatility")
        self.combined_forecast = pd.DataFrame(
            {
                "arima_mean_forecast": self.mean_forecast,
                "garch_volatility": self.volatility_forecast,
                "forecast_upper": self.mean_forecast + self.volatility_forecast,
                "forecast_lower": self.mean_forecast - self.volatility_forecast,
            }
        )
        return self.combined_forecast

    def evaluate(self) -> dict[str, Any]:
        """Return forecasts and evaluation metrics for the test set."""
        if self.test_returns is None:
            self.prepare_returns()
        if self.combined_forecast is None:
            self.forecast_walk_forward()

        actuals = self.test_returns.reset_index(drop=True)
        predictions = self.mean_forecast.reset_index(drop=True)
        metrics = build_metrics(actuals, predictions)

        return {
            "ticker": self.ticker,
            "predictions": predictions.tolist(),
            "actuals": actuals.tolist(),
            "volatility_forecast": self.volatility_forecast.reset_index(drop=True).tolist(),
            "forecast_upper": self.combined_forecast["forecast_upper"].reset_index(drop=True).tolist(),
            "forecast_lower": self.combined_forecast["forecast_lower"].reset_index(drop=True).tolist(),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "directional_accuracy": metrics["directional_accuracy"],
            "sharpe_ratio": metrics["sharpe_ratio"],
        }

    def run(self) -> dict[str, Any]:
        """Run the full GARCH workflow for a single ticker."""
        self.prepare_returns()
        self.fit_arima_mean()
        self.fit_garch_volatility()
        self.forecast_walk_forward()
        return self.evaluate()


def run_all_tickers(
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_path: Path = RESULTS_TABLE_PATH,
    config_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Run GARCH forecasting for all tickers and save the summary results."""
    config = load_config(config_path)
    results: list[dict[str, Any]] = []

    for ticker in config["tickers"]:
        feature_csv = processed_dir / f"{ticker}_features.csv"
        if not feature_csv.exists():
            feature_csv = processed_dir / f"{ticker}.csv"
        if not feature_csv.exists():
            continue

        model = GARCHModel(feature_csv, config_path=config_path)
        result = model.run()
        result["feature_csv"] = str(feature_csv)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    return results_df


def main() -> None:
    """Run GARCH for all tickers as a standalone script."""
    run_all_tickers()


if __name__ == "__main__":
    main()