"""Download daily market and macro data for the forecasting study."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MACRO_DATA_DIR = PROJECT_ROOT / "data" / "macro"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the project configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _standardize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded OHLCV data to a flat, CSV-friendly shape."""
    if frame.empty:
        return frame

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] if isinstance(column, tuple) else column for column in frame.columns]

    if frame.index.name is None:
        frame.index.name = "Date"

    frame = frame.reset_index()

    preferred_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_columns = [column for column in preferred_columns if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in available_columns]

    return frame[available_columns + remaining_columns]


def download_ticker_data(ticker: str, start_date: str, end_date: str, output_dir: Path = RAW_DATA_DIR) -> Path:
    """Download daily OHLCV data for a single ticker and save it as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False, progress=False)
    standardized = _standardize_ohlcv(data)
    output_path = output_dir / f"{ticker}.csv"
    standardized.to_csv(output_path, index=False)
    return output_path


def download_macro_data(start_date: str, end_date: str, output_dir: Path = MACRO_DATA_DIR) -> list[Path]:
    """Download macro market series and save them as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_series = {
        "vix": "^VIX",
        "tnx": "^TNX",
        "dxy": "DX-Y.NYB",
    }

    saved_paths: list[Path] = []
    for file_stem, ticker in macro_series.items():
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False, progress=False)
        standardized = _standardize_ohlcv(data)
        output_path = output_dir / f"{file_stem}.csv"
        standardized.to_csv(output_path, index=False)
        saved_paths.append(output_path)

    return saved_paths


def download_all_tickers(tickers: Iterable[str], start_date: str, end_date: str) -> list[Path]:
    """Download and persist all requested ticker series."""
    saved_paths: list[Path] = []
    for ticker in tickers:
        saved_paths.append(download_ticker_data(ticker, start_date, end_date))
    return saved_paths


def main() -> None:
    """Run the downloader as a standalone script."""
    config = load_config()
    tickers = config["tickers"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    download_all_tickers(tickers, start_date, end_date)
    download_macro_data(start_date, end_date)


if __name__ == "__main__":
    main()