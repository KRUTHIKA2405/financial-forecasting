"""Evaluation metrics for forecasting models."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    """Convert values to a flat NumPy array of floats."""
    return np.asarray(list(values), dtype=float).reshape(-1)


def mae(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    """Mean absolute error."""
    predicted = _to_numpy(predictions)
    observed = _to_numpy(actuals)
    return float(np.mean(np.abs(predicted - observed)))


def rmse(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    """Root mean squared error."""
    predicted = _to_numpy(predictions)
    observed = _to_numpy(actuals)
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def mape(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    """Mean absolute percentage error with zero-safe handling."""
    predicted = _to_numpy(predictions)
    observed = _to_numpy(actuals)
    nonzero_mask = observed != 0
    if not np.any(nonzero_mask):
        return float("nan")
    return float(np.mean(np.abs((observed[nonzero_mask] - predicted[nonzero_mask]) / observed[nonzero_mask])) * 100.0)


def directional_accuracy(predictions: Iterable[float], actuals: Iterable[float]) -> float:
    """Share of periods where predicted and actual directions match."""
    predicted = _to_numpy(predictions)
    observed = _to_numpy(actuals)
    if len(predicted) < 2 or len(observed) < 2:
        return float("nan")

    predicted_direction = np.sign(np.diff(predicted))
    actual_direction = np.sign(np.diff(observed))
    return float(np.mean(predicted_direction == actual_direction))


def sharpe_ratio(predictions: Iterable[float], actuals: Iterable[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio of a simple sign-based strategy.

    The strategy return is sign(prediction) * actual return, so positive
    predictions are treated as long exposure and negative predictions as short.
    """
    predicted = _to_numpy(predictions)
    observed = _to_numpy(actuals)
    if len(predicted) == 0 or len(observed) == 0:
        return float("nan")

    strategy_returns = np.sign(predicted) * observed
    excess_returns = strategy_returns - (risk_free_rate / 252.0)
    return_std = np.std(excess_returns, ddof=1)
    if return_std == 0:
        return float("nan")
    return float(np.sqrt(252.0) * np.mean(excess_returns) / return_std)
