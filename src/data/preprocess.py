"""Preprocess raw ticker and macro data for the forecasting study."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MACRO_DATA_DIR = PROJECT_ROOT / "data" / "macro"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the project configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and parse its date column."""
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"Expected a Date column in {path}")

    frame["Date"] = pd.to_datetime(frame["Date"], utc=False)
    return frame


def set_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and use Date as the index."""
    frame = frame.copy()
    frame = frame.sort_values("Date")
    frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index)
    return frame


def forward_fill_trading_days(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Reindex to business days and forward-fill missing trading days."""
    full_index = pd.bdate_range(start=start_date, end=end_date)
    frame = frame.reindex(full_index)
    frame.index.name = "Date"
    return frame.ffill()


def calculate_adjusted_log_returns(frame: pd.DataFrame, price_column: str = "Adj Close") -> pd.DataFrame:
    """Add adjusted log returns based on the adjusted close series."""
    frame = frame.copy()
    if price_column not in frame.columns:
        price_column = "Close"

    frame["adjusted_log_return"] = np.log(frame[price_column] / frame[price_column].shift(1))
    return frame


def run_adf_test(series: pd.Series, ticker: str) -> dict:
    """Run and print an augmented Dickey-Fuller stationarity test."""
    clean_series = series.dropna()
    if clean_series.empty:
        result = {"ticker": ticker, "error": "not enough data"}
        print(f"ADF test for {ticker}: not enough data")
        return result

    try:
        statistic, p_value, used_lag, n_obs, critical_values, icbest = adfuller(clean_series, autolag="AIC")
    except ValueError as exc:
        result = {"ticker": ticker, "error": str(exc)}
        print(f"ADF test for {ticker}: {exc}")
        return result

    is_stationary = p_value < 0.05
    result = {
        "ticker": ticker,
        "statistic": statistic,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": critical_values,
        "icbest": icbest,
        "is_stationary": is_stationary,
    }

    print(
        f"ADF test for {ticker}: statistic={statistic:.6f}, p_value={p_value:.6f}, "
        f"stationary={is_stationary}"
    )
    print(f"Critical values for {ticker}: {critical_values}")
    return result


def _prefix_macro_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix macro columns so each series remains distinct after merging."""
    frame = frame.copy()
    renamed_columns = {}
    for column in frame.columns:
        if column != "Date":
            renamed_columns[column] = f"{prefix}_{column.replace(' ', '_')}"
    return frame.rename(columns=renamed_columns)


def load_macro_data(macro_dir: Path = MACRO_DATA_DIR) -> pd.DataFrame:
    """Load and merge all macro CSV files on the Date index."""
    macro_frames: list[pd.DataFrame] = []
    for macro_path in sorted(macro_dir.glob("*.csv")):
        macro_name = macro_path.stem.lower()
        macro_frame = load_csv(macro_path)
        macro_frame = set_date_index(macro_frame)
        macro_frame = _prefix_macro_columns(macro_frame.reset_index(), macro_name)
        macro_frame = set_date_index(macro_frame)
        macro_frames.append(macro_frame)

    if not macro_frames:
        return pd.DataFrame()

    merged_macro = macro_frames[0]
    for macro_frame in macro_frames[1:]:
        merged_macro = merged_macro.join(macro_frame, how="outer")

    return merged_macro.sort_index().ffill()


def align_macro_data(ticker_frame: pd.DataFrame, macro_frame: pd.DataFrame) -> pd.DataFrame:
    """Align macro data to the ticker's date index."""
    if macro_frame.empty:
        return ticker_frame

    aligned_macro = macro_frame.reindex(ticker_frame.index).ffill()
    return ticker_frame.join(aligned_macro, how="left")


def clean_ticker_frame(ticker_frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Apply the core cleaning steps to a single ticker's raw data."""
    ticker_frame = set_date_index(ticker_frame)
    ticker_frame = forward_fill_trading_days(ticker_frame, start_date=start_date, end_date=end_date)
    ticker_frame = calculate_adjusted_log_returns(ticker_frame)
    ticker_frame = ticker_frame[ticker_frame["Volume"].fillna(0) != 0]
    ticker_frame = ticker_frame.dropna(subset=["adjusted_log_return"])
    return ticker_frame


def process_ticker_file(
    ticker_path: Path,
    macro_frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Process one ticker CSV and save the cleaned, merged dataframe."""
    ticker = ticker_path.stem
    raw_ticker_frame = load_csv(ticker_path)
    cleaned_ticker_frame = clean_ticker_frame(raw_ticker_frame, start_date=start_date, end_date=end_date)
    run_adf_test(cleaned_ticker_frame["adjusted_log_return"], ticker)

    merged_frame = align_macro_data(cleaned_ticker_frame, macro_frame)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_data_dir / f"{ticker}.csv"
    merged_frame.reset_index().to_csv(output_path, index=False)
    return output_path


def process_all_tickers(
    raw_data_dir: Path = RAW_DATA_DIR,
    macro_data_dir: Path = MACRO_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """Process every raw ticker file and save cleaned outputs."""
    config = load_config()
    start_date = start_date or config["start_date"]
    end_date = end_date or config["end_date"]

    macro_frame = load_macro_data(macro_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for ticker_path in sorted(raw_data_dir.glob("*.csv")):
        saved_paths.append(
            process_ticker_file(
                ticker_path,
                macro_frame,
                start_date,
                end_date,
                processed_data_dir=processed_data_dir,
            )
        )

    return saved_paths


def main() -> None:
    """Run preprocessing as a standalone script."""
    config = load_config()
    process_all_tickers(start_date=config["start_date"], end_date=config["end_date"])


if __name__ == "__main__":
    main()