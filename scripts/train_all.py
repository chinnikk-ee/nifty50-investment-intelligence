"""End-to-end offline pipeline: ingest -> EDA -> train/compare -> recommend ->
report. Warms every cache the API uses, so the dashboard is instant afterwards.

Usage:
  python scripts/train_all.py                # full universe, default models
  python scripts/train_all.py --symbols RELIANCE TCS --horizons 5 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from backend.data_loader import load_processed  # noqa: E402
from ml.eda import generate_eda_suite  # noqa: E402
from ml.evaluation.compare import compare_models  # noqa: E402
from ml.recommendation.engine import recommend_all  # noqa: E402
from ml.reports import generate_report  # noqa: E402
from ml.risk.metrics import market_returns, stock_risk  # noqa: E402
from ml.training.trainer import train_and_forecast  # noqa: E402
from ml.utils import ARTIFACTS_DIR, PROJECT_ROOT, get_logger  # noqa: E402

logger = get_logger("train_all")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None, help="subset of symbols")
    ap.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 20])
    ap.add_argument("--model", default="random_forest", help="forecast model for recommendations")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    data = load_processed()
    panel, close_wide = data["panel"], data["close_wide"]
    symbols = args.symbols or sorted(panel["Symbol"].unique())
    logger.info("universe: %d symbols", len(symbols))

    out_dir = PROJECT_ROOT / "reports" / "generated"
    generate_eda_suite(panel, out_dir / "eda")

    market = market_returns(close_wide)
    forecasts, comparison_rows, risk_rows = {}, [], []
    for i, symbol in enumerate(symbols, 1):
        df = panel[panel["Symbol"] == symbol].sort_values("Date")
        logger.info("[%d/%d] %s", i, len(symbols), symbol)
        try:
            fc = train_and_forecast(df, args.model, horizon=20, save_as=f"{symbol}_ret20")
            forecasts[symbol] = fc["predicted_return"]
            cmp_table = compare_models(df, horizons=tuple(args.horizons), n_splits=3)
            cmp_table.insert(0, "symbol", symbol)
            comparison_rows.append(cmp_table)
            risk_rows.append({"symbol": symbol,
                              **stock_risk(close_wide[symbol].dropna(), market)})
        except Exception as exc:
            logger.error("skipping %s: %s", symbol, exc)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.concat(comparison_rows, ignore_index=True)
    comparison.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    risk_df = pd.DataFrame(risk_rows).set_index("symbol").round(4)
    risk_df.to_csv(ARTIFACTS_DIR / "risk_table.csv")

    recs = recommend_all(panel, forecasts)
    pd.DataFrame(recs).to_json(ARTIFACTS_DIR / "recommendations.json",
                               orient="records", indent=2)
    logger.info("recommendations: %d BUY / %d HOLD / %d SELL",
                *[sum(1 for r in recs if r["action"] == a) for a in ("BUY", "HOLD", "SELL")])

    if not args.skip_report:
        forecast_rows = [
            {"symbol": s, "model": args.model, "horizon": 20,
             "last_close": float(close_wide[s].dropna().iloc[-1]),
             "predicted_price": round(float(close_wide[s].dropna().iloc[-1]) * (1 + r), 2),
             "predicted_return": round(r, 4),
             "directional_accuracy": None}
            for s, r in list(forecasts.items())[:10]
        ]
        generate_report(panel, forecasts=forecast_rows, risk_table=risk_df,
                        recommendations=recs[:15])
    logger.info("done — artifacts in %s", ARTIFACTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
