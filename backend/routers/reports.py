import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.services import get_platform
from ml.reports import generate_report

router = APIRouter(tags=["reports"])


@router.post("/reports/generate")
def generate(focus_symbol: str | None = None, profile: str = "balanced"):
    """Render the full PDF report and return it for download."""
    p = get_platform()
    try:
        recs = p.recommendations()
        forecasts = []
        for rec in recs[:10]:  # forecast detail for the top-scoring names
            f = p.forecast(rec["symbol"])
            forecasts.append({
                "symbol": rec["symbol"], "model": f["model"], "horizon": f["horizon"],
                "last_close": f["last_close"],
                "predicted_price": round(f["predicted_price"], 2),
                "predicted_return": round(f["predicted_return"], 4),
                "directional_accuracy": round(f["metrics"]["directional_accuracy"], 4),
            })
        risk_df = pd.DataFrame(p.risk_table()).set_index("symbol")
        path = generate_report(
            p.panel,
            focus_symbol=focus_symbol,
            forecasts=forecasts,
            portfolio=p.portfolio(profile),
            risk_table=risk_df,
            recommendations=recs[:15],
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc))
    return FileResponse(path, media_type="application/pdf", filename=path.name)
