"""Anomaly, recommendation, backtest, simulation and analytics engines."""
import numpy as np

from ml.analytics import correlation_network, sector_rotation
from ml.anomaly.detectors import anomaly_features, detect_anomalies
from ml.backtest import backtest_portfolio
from ml.recommendation.engine import recommend_all, recommend_stock
from ml.simulation import monte_carlo_forecast, scenario_simulate


def test_anomaly_features_no_nans(stock_df):
    feats = anomaly_features(stock_df)
    assert not feats.drop(columns=["Date"]).isna().any().any()


def test_detect_anomalies_flags_crash(stock_df):
    # The synthetic generator injects a 2020 crash window — detectors should fire.
    report = detect_anomalies(stock_df, "ALPHA")
    assert not report.empty
    assert set(report["type"]).issubset(
        {"volatility_spike", "unusual_return", "volume_surge", "extreme_drawdown"}
    )
    assert (report["votes"] >= 2).all()


def test_recommend_stock_structure(stock_df):
    rec = recommend_stock("ALPHA", stock_df, predicted_return=0.04, sector_return_21d=0.01)
    assert rec["action"] in {"BUY", "HOLD", "SELL"}
    assert -1 <= rec["score"] <= 1
    assert set(rec["components"]) == {"forecast", "momentum", "trend", "risk_adj", "sector_rel"}
    assert rec["symbol"] in rec["reasoning"]


def test_recommend_strong_signals_say_buy(stock_df):
    rec = recommend_stock("ALPHA", stock_df, predicted_return=0.50, sector_return_21d=-0.50)
    assert rec["components"]["forecast"] > 0.9  # saturated tanh


def test_recommend_all_sorted(panel):
    recs = recommend_all(panel)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert len(recs) == panel["Symbol"].nunique()


def test_backtest_runs_and_reports(close_wide):
    weights = {s: 0.25 for s in close_wide.columns}
    out = backtest_portfolio(close_wide, weights, rebalance_days=21)
    assert out["final_value"] > 0
    assert "max_drawdown" in out["metrics"]
    assert len(out["equity_curve"]) > 10
    # Transaction costs make the first equity point below initial capital.
    assert out["equity_curve"][0]["portfolio"] < out["initial_capital"]


def test_monte_carlo_bands_ordered(close_wide):
    out = monte_carlo_forecast(close_wide.iloc[:, 0].dropna(), horizon_days=60, n_sims=300)
    for row in out["paths_summary"]:
        assert row["p5"] <= row["p25"] <= row["p50"] <= row["p75"] <= row["p95"]
    assert 0 <= out["prob_positive"] <= 1


def test_scenario_negative_shock_hurts(close_wide, panel):
    sectors = panel.drop_duplicates("Symbol").set_index("Symbol")["Sector"]
    weights = {s: 0.25 for s in close_wide.columns}
    out = scenario_simulate(close_wide, weights, sectors, market_shock=-0.15)
    assert out["portfolio_impact"] < 0
    assert out["impact_band_low"] < out["portfolio_impact"] < out["impact_band_high"]


def test_sector_rotation_and_network(panel):
    rot = sector_rotation(panel)
    assert rot["leaders"] and rot["laggards"]
    net = correlation_network(panel, threshold=0.0)  # threshold 0 -> fully connected
    n = panel["Symbol"].nunique()
    assert len(net["nodes"]) == n
    assert len(net["edges"]) == n * (n - 1) // 2
