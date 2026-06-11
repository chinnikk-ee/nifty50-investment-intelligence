"use client";

import { Button, Card, CardBody, CardHeader, ErrorNote, Select, Skeleton, Table } from "@/components/ui";
import { useApi, useApiPost } from "@/lib/api";
import { cn, fmtPct } from "@/lib/utils";
import { useMemo, useState } from "react";

type RiskRow = {
  symbol: string; annual_return: number; volatility: number; sharpe: number; sortino: number;
  calmar: number; max_drawdown: number; var_95: number; cvar_95: number; alpha?: number; beta?: number;
};
type Scenario = {
  portfolio_impact: number; impact_band_low: number; impact_band_high: number;
  positions: { symbol: string; sector: string; beta: number; stock_impact: number; portfolio_contribution: number }[];
};

const COLUMNS: { key: keyof RiskRow; label: string; pct?: boolean }[] = [
  { key: "annual_return", label: "Ann. Return", pct: true },
  { key: "volatility", label: "Volatility", pct: true },
  { key: "sharpe", label: "Sharpe" },
  { key: "sortino", label: "Sortino" },
  { key: "calmar", label: "Calmar" },
  { key: "max_drawdown", label: "Max DD", pct: true },
  { key: "var_95", label: "VaR 95", pct: true },
  { key: "cvar_95", label: "CVaR 95", pct: true },
  { key: "alpha", label: "Alpha", pct: true },
  { key: "beta", label: "Beta" },
];

export default function RiskPage() {
  const { data: rows, error, loading } = useApi<RiskRow[]>("/risk");
  const [sortKey, setSortKey] = useState<keyof RiskRow>("sharpe");
  const [shock, setShock] = useState(-10);
  const scenario = useApiPost<Scenario>();

  const sorted = useMemo(
    () => [...(rows ?? [])].sort((a, b) => Number(b[sortKey] ?? 0) - Number(a[sortKey] ?? 0)),
    [rows, sortKey]
  );

  const runScenario = () => {
    if (!rows) return;
    // Stress an equal-weight portfolio of the whole universe.
    const weights = Object.fromEntries(rows.map((r) => [r.symbol, 1 / rows.length]));
    scenario.run("/analytics/scenario", { weights, market_shock: shock / 100, sector_shocks: {}, vol_multiplier: 1.5 });
  };

  if (error) return <ErrorNote message={error} />;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Risk Analytics</h1>
          <p className="text-sm text-muted-foreground">Stock-level risk metrics, annualized · sortable</p>
        </div>
        <label className="text-sm">
          <span className="mb-1 block text-xs text-muted-foreground">Sort by</span>
          <Select value={sortKey} onChange={(e) => setSortKey(e.target.value as keyof RiskRow)}>
            {COLUMNS.map((c) => (
              <option key={c.key} value={c.key}>{c.label}</option>
            ))}
          </Select>
        </label>
      </header>

      <Card>
        <CardHeader title="Scenario Simulator" subtitle="Instantaneous market-shock stress test of an equal-weight universe portfolio (beta-scaled)" />
        <CardBody className="flex flex-wrap items-end gap-3">
          <label className="text-sm">
            <span className="mb-1 block text-xs text-muted-foreground">Market shock</span>
            <Select value={shock} onChange={(e) => setShock(+e.target.value)}>
              {[-30, -20, -10, -5, 5, 10].map((s) => (
                <option key={s} value={s}>{s}%</option>
              ))}
            </Select>
          </label>
          <Button onClick={runScenario} disabled={scenario.loading || !rows}>
            {scenario.loading ? "Simulating…" : "Run Stress Test"}
          </Button>
          {scenario.data && (
            <p className="text-sm">
              Estimated portfolio impact:{" "}
              <span className={cn("font-bold", scenario.data.portfolio_impact >= 0 ? "text-emerald-500" : "text-red-500")}>
                {fmtPct(scenario.data.portfolio_impact)}
              </span>{" "}
              <span className="text-xs text-muted-foreground">
                (band {fmtPct(scenario.data.impact_band_low)} … {fmtPct(scenario.data.impact_band_high)})
              </span>
            </p>
          )}
          {scenario.error && <ErrorNote message={scenario.error} />}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Universe Risk Table" subtitle="Computed over each stock's full history vs the equal-weight market proxy" />
        <CardBody>
          {loading || !rows ? (
            <Skeleton className="h-96" />
          ) : (
            <Table headers={["Symbol", ...COLUMNS.map((c) => c.label)]}>
              {sorted.map((r) => (
                <tr key={r.symbol}>
                  <td className="px-3 py-2 font-semibold">{r.symbol}</td>
                  {COLUMNS.map((c) => {
                    const v = r[c.key] as number | undefined;
                    return (
                      <td key={c.key} className="px-3 py-2 tabular-nums">
                        {v == null ? "—" : c.pct ? fmtPct(v, 1) : v.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </Table>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
