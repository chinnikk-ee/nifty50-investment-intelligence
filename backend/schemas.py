"""MODULE 10 — Pydantic request/response schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Horizon = Literal[1, 5, 20]
Profile = Literal["conservative", "balanced", "aggressive"]


# ----------------------------------------------------------------- stocks
class StockInfo(BaseModel):
    symbol: str
    company: str
    sector: str
    last_close: float
    return_1y: float | None = None
    volatility_1y: float | None = None


class PricePoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class StockDetail(BaseModel):
    symbol: str
    company: str
    sector: str
    start: str
    end: str
    prices: list[PricePoint]
    indicators: dict[str, list[float | None]] = Field(default_factory=dict)


# ----------------------------------------------------------------- forecast
class ForecastRequest(BaseModel):
    symbol: str
    horizon: Horizon = 20
    model: str = "random_forest"
    task: Literal["regression", "classification"] = "regression"


class ForecastResponse(BaseModel):
    symbol: str
    model: str
    task: str
    horizon: int
    as_of: str
    last_close: float
    predicted_return: float | None = None
    predicted_price: float | None = None
    direction: str | None = None
    probability_up: float | None = None
    metrics: dict[str, float]


class ModelComparisonRow(BaseModel):
    model: str
    horizon: int
    task: str
    metrics: dict[str, float]


# ----------------------------------------------------------------- portfolio
class PortfolioRequest(BaseModel):
    profile: Profile = "balanced"
    method: str | None = None
    symbols: list[str] | None = None
    lookback_days: int = Field(756, ge=126, le=5000)


class PortfolioResponse(BaseModel):
    profile: str
    profile_label: str
    profile_description: str
    method: str
    lookback_days: int
    allocation: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float


class QuestionnaireRequest(BaseModel):
    """Investor risk questionnaire: each answer scored 1 (cautious) – 4 (bold)."""
    horizon_years: int = Field(ge=0, le=50)
    loss_tolerance: int = Field(ge=1, le=4, description="Reaction to a 20% drawdown")
    experience: int = Field(ge=1, le=4)
    income_stability: int = Field(ge=1, le=4)
    goal: int = Field(ge=1, le=4, description="1=preserve capital … 4=maximize growth")


class QuestionnaireResponse(BaseModel):
    profile: Profile
    score: int
    rationale: str


# ----------------------------------------------------------------- risk
class RiskMetrics(BaseModel):
    annual_return: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    alpha: float | None = None
    beta: float | None = None
    diversification_ratio: float | None = None


class PortfolioRiskRequest(BaseModel):
    weights: dict[str, float]


# ----------------------------------------------------------------- anomalies
class AnomalyRecord(BaseModel):
    date: str
    symbol: str
    close: float
    ret_1d: float
    volume_ratio: float
    drawdown: float
    votes: int
    methods: str
    type: str


# ----------------------------------------------------------------- intelligence
class Recommendation(BaseModel):
    symbol: str
    sector: str | None = None
    action: Literal["BUY", "HOLD", "SELL"]
    score: float
    components: dict[str, float]
    predicted_return: float | None = None
    horizon: int
    last_close: float
    reasoning: str


class ExplainRequest(BaseModel):
    symbol: str
    horizon: Horizon = 20
    model: str = "random_forest"
    task: Literal["regression", "classification"] = "regression"


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    value: float
    direction: str


class ExplainResponse(BaseModel):
    symbol: str
    model: str
    horizon: int
    prediction: float
    probability_up: float | None = None
    confidence: float
    method: str
    top_features: list[FeatureContribution]
    global_importances: dict[str, float]


# ----------------------------------------------------------------- bonus
class BacktestRequest(BaseModel):
    weights: dict[str, float]
    start: str | None = None
    rebalance_days: int = Field(21, ge=1, le=252)
    transaction_cost_bps: float = Field(10.0, ge=0, le=200)
    initial_capital: float = Field(1_000_000, gt=0)


class SimulationRequest(BaseModel):
    symbol: str
    horizon_days: int = Field(252, ge=5, le=1260)
    n_sims: int = Field(2000, ge=100, le=20000)
    method: Literal["bootstrap", "gbm"] = "bootstrap"


class ScenarioRequest(BaseModel):
    weights: dict[str, float]
    market_shock: float = Field(-0.10, ge=-0.6, le=0.6)
    sector_shocks: dict[str, float] = Field(default_factory=dict)
    vol_multiplier: float = Field(1.0, ge=0.1, le=5.0)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    symbol: str | None = None
    intent: str
