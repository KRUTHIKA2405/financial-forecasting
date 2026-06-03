"""Feature engineering for processed ticker data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_processed_csv(path: Path) -> pd.DataFrame:
    """Load a processed ticker CSV and parse the date column."""
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"Expected a Date column in {path}")

    frame["Date"] = pd.to_datetime(frame["Date"], utc=False)
    return frame.sort_values("Date").reset_index(drop=True)


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add core price-based features."""
    frame = frame.copy()
    frame["price_log_return"] = np.log(frame["Adj Close"] / frame["Adj Close"].shift(1))
    frame["ohlc_ratio"] = (frame["High"] - frame["Low"]) / frame["Open"]
    frame["pct_change"] = frame["Adj Close"].pct_change()
    return frame


def add_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following indicators."""
    frame = frame.copy()
    frame["sma_10"] = ta.sma(frame["Close"], length=10)
    frame["sma_20"] = ta.sma(frame["Close"], length=20)
    frame["sma_50"] = ta.sma(frame["Close"], length=50)
    frame["sma_200"] = ta.sma(frame["Close"], length=200)
    frame["ema_12"] = ta.ema(frame["Close"], length=12)
    frame["ema_26"] = ta.ema(frame["Close"], length=26)

    macd = ta.macd(frame["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        frame["macd"] = macd.iloc[:, 0]
        frame["macd_signal"] = macd.iloc[:, 2]
    else:
        frame["macd"] = np.nan
        frame["macd_signal"] = np.nan

    return frame


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators."""
    frame = frame.copy()
    frame["rsi_14"] = ta.rsi(frame["Close"], length=14)

    stochastic = ta.stoch(frame["High"], frame["Low"], frame["Close"], k=14, d=3, smooth_k=3)
    if stochastic is not None and not stochastic.empty:
        frame["stoch_k"] = stochastic.iloc[:, 0]
        frame["stoch_d"] = stochastic.iloc[:, 1]
    else:
        frame["stoch_k"] = np.nan
        frame["stoch_d"] = np.nan

    frame["roc_10"] = ta.roc(frame["Close"], length=10)
    frame["williams_r"] = ta.willr(frame["High"], frame["Low"], frame["Close"], length=14)
    return frame


def add_volatility_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add volatility and band-based indicators."""
    frame = frame.copy()
    atr = ta.atr(frame["High"], frame["Low"], frame["Close"], length=14)
    bbands = ta.bbands(frame["Close"], length=20, std=2)

    frame["atr_14"] = atr
    if bbands is not None and not bbands.empty:
        frame["bb_lower"] = bbands.iloc[:, 0]
        frame["bb_upper"] = bbands.iloc[:, 2]
        frame["bb_width"] = bbands.iloc[:, 3]
    else:
        frame["bb_upper"] = np.nan
        frame["bb_lower"] = np.nan
        frame["bb_width"] = np.nan

    frame["hist_vol_20"] = frame["price_log_return"].rolling(window=20).std() * np.sqrt(252)
    return frame


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lagged return features."""
    frame = frame.copy()
    for lag in (1, 2, 5, 10, 20):
        frame[f"lag_return_{lag}"] = frame["price_log_return"].shift(lag)
    return frame


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal encodings."""
    frame = frame.copy()
    dates = pd.to_datetime(frame["Date"])

    day_of_week = dates.dt.dayofweek
    month = dates.dt.month

    frame["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return frame


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature groups to a single ticker dataframe."""
    frame = add_price_features(frame)
    frame = add_trend_features(frame)
    frame = add_momentum_features(frame)
    frame = add_volatility_features(frame)
    frame = add_lag_features(frame)
    frame = add_temporal_features(frame)
    frame = frame.dropna().reset_index(drop=True)
    return frame


def process_processed_file(processed_path: Path, output_dir: Path = PROCESSED_DATA_DIR) -> Path:
    """Create features for a processed ticker file and save the result."""
    frame = load_processed_csv(processed_path)
    featured_frame = build_feature_frame(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{processed_path.stem}_features.csv"
    featured_frame.to_csv(output_path, index=False)
    return output_path


def process_all_processed_files(
    input_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> list[Path]:
    """Create feature files for all processed ticker CSVs."""
    saved_paths: list[Path] = []
    for processed_path in sorted(input_dir.glob("*.csv")):
        if processed_path.stem.endswith("_features"):
            continue
        saved_paths.append(process_processed_file(processed_path, output_dir=output_dir))
    return saved_paths


def main() -> None:
    """Run feature engineering as a standalone script."""
    process_all_processed_files()


if __name__ == "__main__":
    main()