"use client";

import { BandChart } from "@/components/charts";
import { Button, Card, CardBody, CardHeader, ErrorNote, Select, Skeleton, StatCard, Table } from "@/components/ui";
import { useApi, useApiPost } from "@/lib/api";
import { fmtInr, fmtPct } from "@/lib/utils";
import { useState } from "react";

type StockInfo = { symbol: string };
type Forecast = {
  symbol: string; model: string; horizon: number; as_of: string; last_close: number;
  predicted_return: number; predicted_price: number; metrics: Record<string, number>;
};
type CompareRow = { model: string; horizon: number; metrics: Record<string, number | null> };
type MonteCarlo = {
  expected_terminal: number; prob_positive: number; var_95_terminal: number;
  paths_summary: { day: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
};

const HORIZONS = [1, 5, 20] as const;

export default function ForecastingPage() {
  const { data: stocks } = useApi<StockInfo[]>("/stocks");
  const { data: models } = useApi<{ regressors: string[] }>("/forecast/models");
  const [symbol, setSymbol] = useState("");
  const [horizon, setHorizon] = useState<(typeof HORIZONS)[number]>(20);
  const [model, setModel] = useState("random_forest");

  const forecast = useApiPost<Forecast>();
  const montecarlo = useApiPost<MonteCarlo>();
  const sym = symbol || stocks?.[0]?.symbol || "";
  const { data: comparison, loading: cmpLoading } = useApi<CompareRow[]>(
    forecast.data ? `/forecast/compare/${forecast.data.symbol}` : null
  );

  const run = async () => {
    await Promise.all([
      forecast.run("/forecast", { symbol: sym, horizon, model }),
      montecarlo.run("/analytics/montecarlo", { symbol: sym, horizon_days: 126, n_sims: 2000, method: "bootstrap" }),
    ]);
  };

  const f = forecast.data;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Forecasting</h1>
        <p className="text-sm text-muted-foreground">
          Walk-forward-validated price forecasts plus Monte-Carlo path simulation
        </p>
      </header>

      <Card>
        <CardBody className="flex flex-wrap items-end gap-3">
          <label className="text-sm">
            <span className="mb-1 block text-xs text-muted-foreground">Stock</span>
            <Select value={sym} onChange={(e) => setSymbol(e.target.value)} className="w-44">
              {(stocks ?? []).map((s) => (
                <option key={s.symbol}>{s.symbol}</option>
              ))}
            </Select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs text-muted-foreground">Horizon (days)</span>
            <Select value={horizon} onChange={(e) => setHorizon(Number(e.target.value) as 1 | 5 | 20)}>
              {HORIZONS.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </Select>
          </label>
          <label className="text-sm">
            <span className="mb-1 block text-xs text-muted-foreground">Model</span>
            <Select value={model} onChange={(e) => setModel(e.target.value)} className="w-44">
              {(models?.regressors ?? ["random_forest"]).map((m) => (
                <option key={m}>{m}</option>
              ))}
            </Select>
          </label>
          <Button onClick={run} disabled={forecast.loading || !sym}>
            {forecast.loading ? "Training…" : "Run Forecast"}
          </Button>
          {forecast.error && <ErrorNote message={forecast.error} />}
        </CardBody>
      </Card>

      {f && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label={`${f.symbol} · last close`} value={fmtInr(f.last_close)} hint={`as of ${f.as_of}`} />
          <StatCard
            label={`Predicted ${f.horizon}d return`}
            value={fmtPct(f.predicted_return)}
            tone={f.predicted_return >= 0 ? "up" : "down"}
            hint={`→ ${fmtInr(f.predicted_price)}`}
          />
          <StatCard label="Directional accuracy" value={fmtPct(f.metrics.directional_accuracy, 0)} hint="out-of-sample, walk-forward" />
          <StatCard label="RMSE" value={f.metrics.rmse?.toFixed(4) ?? "—"} hint={`MAE ${f.metrics.mae?.toFixed(4)} · R² ${f.metrics.r2?.toFixed(3)}`} />
        </div>
      )}

      {montecarlo.data && (
        <Card>
          <CardHeader
            title="Monte-Carlo Simulation (126 days, 2000 paths, bootstrap)"
            subtitle={`P(gain) ${fmtPct(montecarlo.data.prob_positive, 0)} · terminal VaR95 ${fmtPct(montecarlo.data.var_95_terminal)} · expected terminal ${fmtInr(montecarlo.data.expected_terminal)}`}
          />
          <CardBody>
            <BandChart data={montecarlo.data.paths_summary} />
          </CardBody>
        </Card>
      )}

      {f && (
        <Card>
          <CardHeader title="Model Comparison" subtitle="Every model × horizon, ranked by walk-forward RMSE" />
          <CardBody>
            {cmpLoading ? (
              <Skeleton className="h-48" />
            ) : (
              <Table headers={["Model", "Horizon", "RMSE", "MAE", "MAPE %", "R²", "Dir. Acc."]}>
                {(comparison ?? []).map((r, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2 font-medium">{r.model}</td>
                    <td className="px-3 py-2">{r.horizon}d</td>
                    <td className="px-3 py-2 tabular-nums">{r.metrics.rmse?.toFixed(4)}</td>
                    <td className="px-3 py-2 tabular-nums">{r.metrics.mae?.toFixed(4)}</td>
                    <td className="px-3 py-2 tabular-nums">{r.metrics.mape?.toFixed(1)}</td>
                    <td className="px-3 py-2 tabular-nums">{r.metrics.r2?.toFixed(3)}</td>
                    <td className="px-3 py-2 tabular-nums">{fmtPct(r.metrics.directional_accuracy ?? null, 1)}</td>
                  </tr>
                ))}
              </Table>
            )}
          </CardBody>
        </Card>
      )}
    </div>
  );
}
