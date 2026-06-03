"""Diebold-Mariano test utilities for comparing forecast accuracy."""

from __future__ import annotations

import ast
import sys
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import mae, mape, rmse


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float).reshape(-1)


def _loss_differential(actuals: Iterable[float], predictions_a: Iterable[float], predictions_b: Iterable[float], loss: str = "mse") -> np.ndarray:
    actual = _to_numpy(actuals)
    a = _to_numpy(predictions_a)
    b = _to_numpy(predictions_b)

    if loss == "mae":
        return np.abs(actual - a) - np.abs(actual - b)
    return (actual - a) ** 2 - (actual - b) ** 2


def _acf_lag1(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered = values - np.mean(values)
    denom = np.sum(centered**2)
    if denom == 0:
        return 0.0
    return float(np.sum(centered[1:] * centered[:-1]) / denom)


def diebold_mariano_test(
    actuals: Iterable[float],
    hybrid_predictions: Iterable[float],
    baseline_predictions: Iterable[float],
    loss: str = "mse",
) -> dict[str, float]:
    """Compute a simple Diebold-Mariano test comparing hybrid and baseline forecasts."""
    differential = _loss_differential(actuals, hybrid_predictions, baseline_predictions, loss=loss)
    n_obs = len(differential)
    if n_obs < 2:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "mean_diff": float("nan")}

    mean_diff = float(np.mean(differential))
    gamma0 = float(np.var(differential, ddof=1))
    rho1 = _acf_lag1(differential)
    long_run_variance = gamma0 * (1.0 + 2.0 * rho1)
    if long_run_variance <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "mean_diff": mean_diff}

    dm_stat = mean_diff / math.sqrt(long_run_variance / n_obs)
    p_value = math.erfc(abs(dm_stat) / math.sqrt(2.0))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "mean_diff": mean_diff}


def _parse_series_cell(value: object) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, (list, tuple, np.ndarray)):
        return [float(item) for item in parsed]
    return [float(parsed)]


def load_result_table(path: Path) -> pd.DataFrame:
    """Load a results table that stores predictions and actuals as CSV columns."""
    frame = pd.read_csv(path)
    for column in ("predictions", "actuals"):
        if column in frame.columns:
            frame[column] = frame[column].apply(_parse_series_cell)
    return frame


def compare_hybrid_to_baselines(
    hybrid_results_path: Path,
    baseline_results_paths: dict[str, Path],
    loss: str = "mse",
) -> pd.DataFrame:
    """Compare hybrid forecasts against each baseline and print a summary table."""
    hybrid_results = load_result_table(hybrid_results_path)
    rows: list[dict[str, object]] = []

    for baseline_name, baseline_path in baseline_results_paths.items():
        baseline_results = load_result_table(baseline_path)
        merged = hybrid_results.merge(
            baseline_results,
            on="ticker",
            suffixes=("_hybrid", "_baseline"),
        )

        hybrid_predictions: list[float] = []
        baseline_predictions: list[float] = []
        actuals: list[float] = []

        for _, row in merged.iterrows():
            hybrid_predictions.extend(row["predictions_hybrid"])
            baseline_predictions.extend(row["predictions_baseline"])
            actuals.extend(row["actuals_hybrid"])

        dm_result = diebold_mariano_test(actuals, hybrid_predictions, baseline_predictions, loss=loss)
        better = dm_result["p_value"] < 0.05 and dm_result["mean_diff"] < 0.0
        rows.append(
            {
                "baseline": baseline_name,
                "dm_stat": dm_result["dm_stat"],
                "p_value": dm_result["p_value"],
                "hybrid_significantly_better": bool(better),
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def main() -> None:
    """Run a DM comparison if the required result tables are available."""
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "results" / "tables"
    hybrid_results_path = results_dir / "hybrid_results.csv"
    baseline_paths = {
        "ARIMA": results_dir / "arima_results.csv",
        "GARCH": results_dir / "garch_results.csv",
    }

    if hybrid_results_path.exists() and all(path.exists() for path in baseline_paths.values()):
        compare_hybrid_to_baselines(hybrid_results_path, baseline_paths)
    else:
        print("Required result tables were not found. Provide hybrid_results.csv and baseline tables in results/tables/.")


if __name__ == "__main__":
    main()
