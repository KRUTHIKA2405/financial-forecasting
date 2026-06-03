"""Generate the complete comparative results report for all forecasting models."""

from __future__ import annotations

import ast
import base64
import io
import sys
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import directional_accuracy, mae, mape, rmse, sharpe_ratio


CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SUMMARY_HTML_PATH = PROJECT_ROOT / "results" / "summary_report.html"

ARIMA_RESULTS_PATH = RESULTS_TABLES_DIR / "arima_results.csv"
GARCH_RESULTS_PATH = RESULTS_TABLES_DIR / "garch_results.csv"
HYBRID_RESULTS_PATH = RESULTS_TABLES_DIR / "hybrid_results.csv"

MODEL_FILES: dict[str, Path] = {
    "ARIMA": ARIMA_RESULTS_PATH,
    "GARCH": GARCH_RESULTS_PATH,
    "CNN+LSTM": HYBRID_RESULTS_PATH,
}

MODEL_COLORS = {
    "ARIMA": "#1f77b4",
    "GARCH": "#ff7f0e",
    "CNN+LSTM": "#2ca02c",
}

METRIC_COLUMNS = ["MAE", "RMSE", "MAPE", "Directional Accuracy", "Sharpe Ratio"]
LOWER_IS_BETTER = {"MAE", "RMSE", "MAPE"}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the project configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ticker_order() -> list[str]:
    """Return ticker order from the project configuration."""
    config = load_config()
    return list(config.get("tickers", []))


def parse_series_cell(value: object) -> list[float]:
    """Parse a CSV cell that may contain a serialized list of numbers."""
    if isinstance(value, list):
        return [float(item) for item in value]
    if pd.isna(value):
        return []

    value_str = str(value).strip()
    try:
        parsed_value = ast.literal_eval(value_str)
        if isinstance(parsed_value, (list, tuple)):
            return [float(item) for item in parsed_value]
        return [float(parsed_value)]
    except (SyntaxError, ValueError):
        cleaned = value_str.strip("[]")
        if not cleaned:
            return []
        numeric_values = [item for item in cleaned.replace("\n", " ").split() if item]
        return [float(item) for item in numeric_values]


def load_result_table(path: Path, model_name: str) -> pd.DataFrame:
    """Load a model result table and normalize column names."""
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise ValueError(f"Expected a ticker column in {path}")

    rename_map = {
        "mae": "MAE",
        "rmse": "RMSE",
        "mape": "MAPE",
        "directional_accuracy": "Directional Accuracy",
        "directional accuracy": "Directional Accuracy",
        "sharpe_ratio": "Sharpe Ratio",
        "sharpe ratio": "Sharpe Ratio",
    }
    frame = frame.rename(columns=rename_map)
    frame["Model"] = model_name

    for column in ("predictions", "actuals"):
        if column in frame.columns:
            frame[column] = frame[column].apply(parse_series_cell)

    return frame


def compute_row_metrics(row: pd.Series) -> dict[str, float]:
    """Compute metrics from stored predictions and actuals when needed."""
    predictions = row.get("predictions", [])
    actuals = row.get("actuals", [])

    if not isinstance(predictions, list):
        predictions = parse_series_cell(predictions)
    if not isinstance(actuals, list):
        actuals = parse_series_cell(actuals)

    return {
        "MAE": float(row["MAE"]) if "MAE" in row and pd.notna(row["MAE"]) else mae(predictions, actuals),
        "RMSE": float(row["RMSE"]) if "RMSE" in row and pd.notna(row["RMSE"]) else rmse(predictions, actuals),
        "MAPE": float(row["MAPE"]) if "MAPE" in row and pd.notna(row["MAPE"]) else mape(predictions, actuals),
        "Directional Accuracy": float(row["Directional Accuracy"]) if "Directional Accuracy" in row and pd.notna(row["Directional Accuracy"]) else directional_accuracy(predictions, actuals),
        "Sharpe Ratio": float(row["Sharpe Ratio"]) if "Sharpe Ratio" in row and pd.notna(row["Sharpe Ratio"]) else sharpe_ratio(predictions, actuals),
    }


def load_all_results() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load ARIMA, GARCH, and Hybrid result tables into a single dataframe."""
    loaded_frames: list[pd.DataFrame] = []
    source_frames: dict[str, pd.DataFrame] = {}
    missing_models: list[str] = []

    for model_name, results_path in MODEL_FILES.items():
        if not results_path.exists():
            missing_models.append(model_name)
            continue
        frame = load_result_table(results_path, model_name)
        loaded_frames.append(frame)
        source_frames[model_name] = frame

    if missing_models:
        missing_files = ", ".join(f"{model}: {MODEL_FILES[model].name}" for model in missing_models)
        raise FileNotFoundError(f"Missing required result CSVs: {missing_files}")

    if not loaded_frames:
        raise FileNotFoundError(f"No results CSVs were found in {RESULTS_TABLES_DIR}")

    combined = pd.concat(loaded_frames, ignore_index=True, sort=False)
    combined["Model"] = pd.Categorical(combined["Model"], categories=list(MODEL_FILES.keys()), ordered=True)

    comparison_rows: list[dict[str, object]] = []
    for _, row in combined.iterrows():
        metrics = compute_row_metrics(row)
        comparison_rows.append(
            {
                "Model": str(row["Model"]),
                "Ticker": str(row["ticker"]),
                **metrics,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison["Ticker"] = pd.Categorical(comparison["Ticker"], categories=ticker_order(), ordered=True)
    comparison = comparison.sort_values(["Ticker", "Model"]).reset_index(drop=True)
    return comparison, source_frames


def save_master_comparison(comparison: pd.DataFrame) -> Path:
    """Save the master comparison table to disk."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_TABLES_DIR / "master_comparison.csv"
    comparison.to_csv(output_path, index=False)
    return output_path


def save_summary_avg(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean of each metric per model across all tickers."""
    summary = comparison.groupby("Model", as_index=False)[METRIC_COLUMNS].mean(numeric_only=True)
    summary["Model"] = pd.Categorical(summary["Model"], categories=list(MODEL_FILES.keys()), ordered=True)
    summary = summary.sort_values("Model").reset_index(drop=True)
    output_path = RESULTS_TABLES_DIR / "summary_avg.csv"
    summary.to_csv(output_path, index=False)
    return summary


def compute_best_model_per_ticker(comparison: pd.DataFrame) -> pd.DataFrame:
    """For each ticker, record the model that wins on each metric."""
    rows: list[dict[str, str]] = []

    for ticker, ticker_frame in comparison.groupby("Ticker", sort=False):
        row = {"Ticker": str(ticker)}
        for metric in METRIC_COLUMNS:
            metric_values = pd.to_numeric(ticker_frame[metric], errors="coerce")
            if metric in LOWER_IS_BETTER:
                winner = ticker_frame.loc[metric_values.idxmin(), "Model"]
            else:
                winner = ticker_frame.loc[metric_values.idxmax(), "Model"]
            row[metric] = str(winner)
        rows.append(row)

    best_df = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    output_path = RESULTS_TABLES_DIR / "best_model_per_ticker.csv"
    best_df.to_csv(output_path, index=False)
    return best_df


def figure_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def save_figure(fig: plt.Figure, path: Path) -> str:
    """Save a figure to disk and return its base64 representation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return figure_to_base64(fig)


def create_grouped_bar_chart(comparison: pd.DataFrame, metric: str, filename: str, title: str) -> tuple[Path, str]:
    """Create grouped bar charts for ticker-level metric comparison."""
    pivot = comparison.pivot(index="Ticker", columns="Model", values=metric)
    pivot = pivot.reindex(index=ticker_order(), columns=list(MODEL_FILES.keys()))

    fig, ax = plt.subplots(figsize=(18, 8))
    pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS[model] for model in pivot.columns])
    ax.set_title(title)
    ax.set_xlabel("Ticker")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Model")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / filename
    encoded = save_figure(fig, output_path)
    return output_path, encoded


def create_heatmap(comparison: pd.DataFrame, metric: str, filename: str, cmap: str, title: str) -> tuple[Path, str]:
    """Create model-by-ticker heatmaps for a metric."""
    pivot = comparison.pivot(index="Model", columns="Ticker", values=metric)
    pivot = pivot.reindex(index=list(MODEL_FILES.keys()), columns=ticker_order())

    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax, cbar_kws={"label": metric})
    ax.set_title(title)
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Model")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / filename
    encoded = save_figure(fig, output_path)
    return output_path, encoded


def plot_actual_vs_predicted_for_ticker(source_frames: dict[str, pd.DataFrame], ticker: str, filename: str) -> tuple[Path, str]:
    """Plot actual vs predicted close prices for all three models on the same chart."""
    fig, ax = plt.subplots(figsize=(18, 8))

    actual_series: list[float] | None = None
    for model_name in MODEL_FILES.keys():
        model_frame = source_frames.get(model_name)
        if model_frame is None:
            continue

        ticker_rows = model_frame[model_frame["ticker"] == ticker]
        if ticker_rows.empty:
            continue

        row = ticker_rows.iloc[0]
        predictions = parse_series_cell(row.get("predictions", []))
        actuals = parse_series_cell(row.get("actuals", []))
        if actual_series is None and actuals:
            actual_series = actuals

        ax.plot(predictions, label=f"{model_name} Predicted", linewidth=2, color=MODEL_COLORS[model_name])

    if actual_series is None:
        raise ValueError(f"No actual series found for {ticker}")

    ax.plot(actual_series, label=f"{ticker} Actual", color="black", linewidth=2.5, linestyle="--")
    ax.set_title(f"{ticker}: Actual vs Predicted Close Price")
    ax.set_xlabel("Test Set Step")
    ax.set_ylabel("Close Price")
    ax.legend(loc="best")
    plt.tight_layout()

    output_path = RESULTS_FIGURES_DIR / filename
    encoded = save_figure(fig, output_path)
    return output_path, encoded


def build_summary_avg_html(summary: pd.DataFrame) -> str:
    """Build a Bootstrap-styled summary average table with best/worst highlighting."""
    summary = summary.copy().reset_index(drop=True)
    for metric in METRIC_COLUMNS:
        summary[metric] = pd.to_numeric(summary[metric], errors="coerce")

    html_parts = [
        '<table class="table table-striped table-bordered table-hover align-middle">',
        "<thead><tr><th>Model</th>" + "".join(f"<th>{escape(column)}</th>" for column in METRIC_COLUMNS) + "</tr></thead>",
        "<tbody>",
    ]

    for _, row in summary.iterrows():
        html_parts.append("<tr>")
        html_parts.append(f"<td><strong>{escape(str(row['Model']))}</strong></td>")
        for metric in METRIC_COLUMNS:
            value = float(row[metric])
            if metric in LOWER_IS_BETTER:
                best_value = summary[metric].min()
                worst_value = summary[metric].max()
                if value == best_value:
                    style = ' style="background-color:#d4edda;color:#155724;font-weight:600;"'
                elif value == worst_value:
                    style = ' style="background-color:#f8d7da;color:#721c24;font-weight:600;"'
                else:
                    style = ""
            else:
                best_value = summary[metric].max()
                worst_value = summary[metric].min()
                if value == best_value:
                    style = ' style="background-color:#d4edda;color:#155724;font-weight:600;"'
                elif value == worst_value:
                    style = ' style="background-color:#f8d7da;color:#721c24;font-weight:600;"'
                else:
                    style = ""
            html_parts.append(f"<td{style}>{value:.4f}</td>")
        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")
    return "".join(html_parts)


def build_results_report() -> dict[str, object]:
    """Generate all tables, plots, and the HTML report."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison, source_frames = load_all_results()
    save_master_comparison(comparison)

    summary_avg = save_summary_avg(comparison)
    best_model_per_ticker = compute_best_model_per_ticker(comparison)

    figure_assets: dict[str, dict[str, str]] = {}

    for metric, filename, title in [
        ("RMSE", "rmse_grouped_bar.png", "Grouped RMSE Comparison"),
        ("MAE", "mae_grouped_bar.png", "Grouped MAE Comparison"),
        ("Directional Accuracy", "directional_accuracy_grouped_bar.png", "Grouped Directional Accuracy Comparison"),
        ("Sharpe Ratio", "sharpe_ratio_grouped_bar.png", "Grouped Sharpe Ratio Comparison"),
    ]:
        path, encoded = create_grouped_bar_chart(comparison, metric, filename, title)
        figure_assets[metric] = {"path": str(path), "base64": encoded}

    for metric, filename, cmap, title in [
        ("RMSE", "rmse_heatmap.png", "magma", "RMSE Heatmap"),
        ("Directional Accuracy", "directional_accuracy_heatmap.png", "Greens", "Directional Accuracy Heatmap"),
        ("MAPE", "mape_heatmap.png", "viridis", "MAPE Heatmap"),
    ]:
        path, encoded = create_heatmap(comparison, metric, filename, cmap, title)
        figure_assets[f"heatmap_{metric}"] = {"path": str(path), "base64": encoded}

    for ticker, filename in [("AAPL", "aapl_actual_vs_predicted.png"), ("SPY", "spy_actual_vs_predicted.png")]:
        path, encoded = plot_actual_vs_predicted_for_ticker(source_frames, ticker, filename)
        figure_assets[f"{ticker}_line"] = {"path": str(path), "base64": encoded}

    summary_html = build_summary_avg_html(summary_avg)
    best_model_html = best_model_per_ticker.to_html(index=False, classes="table table-striped table-bordered table-hover align-middle", border=0)

    html_parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Comparative Study of Time Series Forecasting in Financial Markets</title>",
        '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">',
        "<style>",
        "body { background: #f8f9fa; }",
        ".report-section { margin-bottom: 2.5rem; }",
        ".figure-card img { width: 100%; height: auto; border: 1px solid #dee2e6; border-radius: .5rem; background: white; }",
        ".figure-title { margin-bottom: .75rem; }",
        ".summary-card { background: white; border-radius: 1rem; padding: 1rem; box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,.075); }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="container-fluid py-4">',
        '<div class="row"><div class="col-12">',
        '<div class="p-4 mb-4 bg-white rounded-3 shadow-sm">',
        '<h1 class="display-6 mb-2">Comparative Study of Time Series Forecasting in Financial Markets</h1>',
        '<p class="text-muted mb-0">Master comparison, summary averages, best model per ticker, and all requested figures.</p>',
        "</div>",
        "</div></div>",
        '<div class="report-section">',
        '<h2 class="h4 mb-3">Summary Average Metrics</h2>',
        '<div class="summary-card">',
        summary_html,
        "</div>",
        "</div>",
    ]

    def figure_block(title: str, image_key: str) -> str:
        return (
            '<div class="col-12 col-lg-6 report-section">'
            f'<h2 class="h5 figure-title">{escape(title)}</h2>'
            '<div class="figure-card shadow-sm bg-white p-2 rounded-3">'
            f'<img src="data:image/png;base64,{figure_assets[image_key]["base64"]}" alt="{escape(title)}">'
            "</div></div>"
        )

    html_parts.extend([
        '<div class="row g-4">',
        figure_block("Grouped RMSE Comparison", "RMSE"),
        figure_block("Grouped MAE Comparison", "MAE"),
        figure_block("Grouped Directional Accuracy Comparison", "Directional Accuracy"),
        figure_block("Grouped Sharpe Ratio Comparison", "Sharpe Ratio"),
        "</div>",
        '<div class="row g-4 mt-1">',
        figure_block("RMSE Heatmap", "heatmap_RMSE"),
        figure_block("Directional Accuracy Heatmap", "heatmap_Directional Accuracy"),
        figure_block("MAPE Heatmap", "heatmap_MAPE"),
        "</div>",
        '<div class="row g-4 mt-1">',
        figure_block("AAPL Actual vs Predicted Close Price", "AAPL_line"),
        figure_block("SPY Actual vs Predicted Close Price", "SPY_line"),
        "</div>",
        '<div class="report-section mt-4">',
        '<h2 class="h4 mb-3">Best Model Per Ticker</h2>',
        best_model_html,
        "</div>",
        "</div>",
        "</body>",
        "</html>",
    ])

    SUMMARY_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_HTML_PATH.write_text("\n".join(html_parts), encoding="utf-8")

    return {
        "comparison": comparison,
        "summary_avg": summary_avg,
        "best_model_per_ticker": best_model_per_ticker,
        "figure_assets": figure_assets,
        "summary_html_path": SUMMARY_HTML_PATH,
    }


def main() -> None:
    """Generate the full comparative report."""
    try:
        report = build_results_report()
    except FileNotFoundError as exc:
        print(str(exc))
        return

    print(report["comparison"].to_string(index=False))
    print(f"Saved master comparison table to {RESULTS_TABLES_DIR / 'master_comparison.csv'}")
    print(f"Saved summary averages to {RESULTS_TABLES_DIR / 'summary_avg.csv'}")
    print(f"Saved best model per ticker table to {RESULTS_TABLES_DIR / 'best_model_per_ticker.csv'}")
    print(f"Saved summary report to {SUMMARY_HTML_PATH}")


if __name__ == "__main__":
    main()