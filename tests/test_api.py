"""MODULE 13 — API integration tests over the full app with synthetic data."""


def test_health(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}


def test_stocks_list_and_detail(api_client):
    stocks = api_client.get("/stocks").json()
    assert len(stocks) >= 2
    symbol = stocks[0]["symbol"]

    detail = api_client.get(f"/stocks/{symbol}").json()
    assert detail["symbol"] == symbol
    assert len(detail["prices"]) > 100
    assert "rsi14" in detail["indicators"]

    assert api_client.get("/stocks/NOPE").status_code == 404


def test_stocks_filters(api_client):
    sectors = api_client.get("/sectors").json()
    assert sectors
    filtered = api_client.get(f"/stocks?sector={sectors[0]}").json()
    assert all(s["sector"] == sectors[0] for s in filtered)


def test_forecast_endpoint(api_client):
    symbol = api_client.get("/stocks").json()[0]["symbol"]
    res = api_client.post("/forecast", json={"symbol": symbol, "horizon": 5,
                                             "model": "linear_regression"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["predicted_price"] is not None
    assert "directional_accuracy" in body["metrics"]


def test_portfolio_and_questionnaire(api_client):
    res = api_client.post("/portfolio", json={"profile": "balanced", "lookback_days": 500})
    assert res.status_code == 200, res.text
    body = res.json()
    assert abs(sum(body["allocation"].values()) - 100) < 0.5

    quiz = api_client.post("/portfolio/questionnaire", json={
        "horizon_years": 10, "loss_tolerance": 4, "experience": 4,
        "income_stability": 4, "goal": 4,
    }).json()
    assert quiz["profile"] == "aggressive"


def test_risk_endpoints(api_client):
    table = api_client.get("/risk").json()
    assert table and "sharpe" in table[0]
    symbol = table[0]["symbol"]
    one = api_client.get(f"/risk/{symbol}").json()
    assert "max_drawdown" in one

    weights = {row["symbol"]: 1 / len(table) for row in table}
    port = api_client.post("/risk/portfolio", json={"weights": weights}).json()
    assert port["volatility"] > 0


def test_anomalies_endpoint(api_client):
    symbol = api_client.get("/stocks").json()[0]["symbol"]
    res = api_client.get(f"/anomalies?symbol={symbol}&limit=10")
    assert res.status_code == 200
    for rec in res.json():
        assert rec["votes"] >= 2


def test_recommendations_and_explainability(api_client):
    recs = api_client.get("/recommendations").json()
    assert recs and recs[0]["action"] in {"BUY", "HOLD", "SELL"}
    assert "reasoning" in recs[0]

    symbol = recs[0]["symbol"]
    one = api_client.get(f"/recommendations/{symbol}").json()
    assert one["symbol"] == symbol

    exp = api_client.post("/explainability", json={"symbol": symbol, "horizon": 5,
                                                   "model": "random_forest"})
    assert exp.status_code == 200, exp.text
    body = exp.json()
    assert body["top_features"]
    assert 0 <= body["confidence"] <= 1


def test_analytics_endpoints(api_client):
    summary = api_client.get("/analytics/summary").json()
    assert summary["n_stocks"] >= 2

    rot = api_client.get("/analytics/sector-rotation").json()
    assert rot["leaders"]

    net = api_client.get("/analytics/network?threshold=0.2").json()
    assert net["nodes"]

    symbols = [s["symbol"] for s in api_client.get("/stocks").json()]
    weights = {s: 1 / len(symbols) for s in symbols}
    bt = api_client.post("/analytics/backtest", json={"weights": weights}).json()
    assert bt["final_value"] > 0

    mc = api_client.post("/analytics/montecarlo", json={
        "symbol": symbols[0], "horizon_days": 30, "n_sims": 200}).json()
    assert 0 <= mc["prob_positive"] <= 1


def test_chat_endpoint(api_client):
    symbol = api_client.get("/stocks").json()[0]["symbol"]
    res = api_client.post("/chat", json={"question": f"how risky is {symbol}?"}).json()
    assert res["intent"] == "risk"
    assert symbol in res["answer"]

    res = api_client.post("/chat", json={"question": "hello there"}).json()
    assert res["intent"] == "help"
