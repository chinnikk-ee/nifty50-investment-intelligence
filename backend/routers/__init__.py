from backend.routers import (  # noqa: F401
    analytics,
    anomalies,
    chat,
    explainability,
    forecast,
    portfolio,
    recommendations,
    reports,
    risk,
    stocks,
)

ALL_ROUTERS = [
    stocks.router,
    forecast.router,
    portfolio.router,
    risk.router,
    anomalies.router,
    recommendations.router,
    explainability.router,
    analytics.router,
    chat.router,
    reports.router,
]
