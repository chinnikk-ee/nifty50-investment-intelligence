"""MODULE 12 — PDF report generation.

Renders a multi-page investment report with matplotlib's PdfPages (no extra
dependencies): cover summary, EDA charts, forecast table, portfolio
allocation, risk metrics and recommendations.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ml import eda
from ml.utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "generated"


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(df) + 2.2))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=18)
    table = ax.table(cellText=df.round(4).astype(str).values,
                     colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")
    if note:
        ax.text(0, -0.02, note, transform=ax.transAxes, fontsize=7, color="gray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _cover_page(pdf: PdfPages, n_symbols: int, date_range: str, focus: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis("off")
    ax.text(0.5, 0.72, "NIFTY-50 Investment Intelligence Report",
            ha="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.62, f"Generated {datetime.now():%d %b %Y %H:%M}", ha="center", fontsize=11)
    ax.text(0.5, 0.50,
            f"Universe: {n_symbols} stocks   •   History: {date_range}   •   Focus: {focus}",
            ha="center", fontsize=11)
    ax.text(0.5, 0.18,
            "All analytics are derived exclusively from the supplied historical dataset.\n"
            "Forecasts are validated walk-forward; nothing here is investment advice.",
            ha="center", fontsize=9, color="gray")
    pdf.savefig(fig)
    plt.close(fig)


def generate_report(
    panel: pd.DataFrame,
    out_path: str | Path | None = None,
    focus_symbol: str | None = None,
    forecasts: list[dict] | None = None,
    portfolio: dict | None = None,
    risk_table: pd.DataFrame | None = None,
    recommendations: list[dict] | None = None,
) -> Path:
    """Assemble the full PDF. Any analytics section that isn't supplied is
    simply skipped, so the report works at any stage of the pipeline."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(panel["Symbol"].unique())
    focus = focus_symbol or symbols[0]
    out_path = Path(out_path or REPORTS_DIR / f"investment_report_{datetime.now():%Y%m%d_%H%M%S}.pdf")

    with PdfPages(out_path) as pdf:
        date_range = f"{panel['Date'].min():%Y-%m-%d} → {panel['Date'].max():%Y-%m-%d}"
        _cover_page(pdf, len(symbols), date_range, focus)

        top = symbols[: min(8, len(symbols))]
        for fig in (
            eda.plot_price_trends(panel, top),
            eda.plot_sector_comparison(panel),
            eda.plot_volatility(panel, top),
            eda.plot_drawdown(panel, focus),
            eda.plot_correlation_heatmap(panel),
        ):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if forecasts:
            df = pd.DataFrame(forecasts)[
                ["symbol", "model", "horizon", "last_close", "predicted_price",
                 "predicted_return", "directional_accuracy"]
            ]
            _table_page(pdf, "Price Forecasts (walk-forward validated)", df,
                        "directional_accuracy is out-of-sample; predicted values are model outputs, not advice.")

        if portfolio:
            alloc = pd.Series(portfolio["allocation"], name="weight_pct").sort_values(ascending=False)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
            alloc.plot.pie(ax=ax1, autopct="%.1f%%", textprops={"fontsize": 7}, ylabel="")
            ax1.set_title(f"{portfolio['profile_label']} Portfolio — {portfolio['method']}")
            ax2.axis("off")
            ax2.text(0.05, 0.8, f"Expected return: {portfolio['expected_return']:.2%}", fontsize=12)
            ax2.text(0.05, 0.65, f"Volatility: {portfolio['volatility']:.2%}", fontsize=12)
            ax2.text(0.05, 0.5, f"Sharpe ratio: {portfolio['sharpe']:.2f}", fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if risk_table is not None and not risk_table.empty:
            _table_page(pdf, "Risk Metrics (annualized)", risk_table.reset_index())

        if recommendations:
            df = pd.DataFrame(recommendations)[["symbol", "action", "score", "reasoning"]]
            df["reasoning"] = df["reasoning"].str.wrap(80)
            _table_page(pdf, "AI Recommendations", df,
                        "Composite of forecast, momentum, trend, risk-adjusted and sector-relative signals.")

    logger.info("report written to %s", out_path)
    return out_path
