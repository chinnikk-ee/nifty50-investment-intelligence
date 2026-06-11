"use client";

import { SimpleBars } from "@/components/charts";
import { Badge, Card, CardBody, CardHeader, ErrorNote, Skeleton, StatCard, Table } from "@/components/ui";
import { useApi } from "@/lib/api";
import { fmtPct } from "@/lib/utils";
import Link from "next/link";

type Summary = {
  n_stocks: number;
  date_range: { start: string; end: string };
  buy_count: number;
  sell_count: number;
  top_movers_21d: { symbol: string; return: number }[];
  bottom_movers_21d: { symbol: string; return: number }[];
  leading_sectors: { sector: string; momentum: number }[];
  lagging_sectors: { sector: string; momentum: number }[];
};

type Rec = { symbol: string; sector?: string; action: "BUY" | "HOLD" | "SELL"; score: number; reasoning: string };

export default function DashboardPage() {
  const { data: summary, error, loading } = useApi<Summary>("/analytics/summary");
  const { data: recs } = useApi<Rec[]>("/recommendations");

  if (error) return <ErrorNote message={`Backend unreachable — ${error}. Start it with: uvicorn backend.main:app`} />;
  if (loading || !summary)
    return (
      <div className="grid gap-4 md:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
    );

  const movers = [...summary.top_movers_21d, ...summary.bottom_movers_21d.slice().reverse()];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Market Dashboard</h1>
        <p className="text-sm text-muted-foreground">
          {summary.n_stocks} NIFTY stocks · history {summary.date_range.start} → {summary.date_range.end}
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Universe" value={summary.n_stocks} hint="stocks analyzed" />
        <StatCard label="Buy Signals" value={summary.buy_count} tone="up" hint="composite score ≥ +0.20" />
        <StatCard label="Sell Signals" value={summary.sell_count} tone="down" hint="composite score ≤ −0.20" />
        <StatCard
          label="Leading Sector"
          value={summary.leading_sectors[0]?.sector ?? "—"}
          hint={`momentum ${fmtPct(summary.leading_sectors[0]?.momentum)}`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="21-Day Movers" subtitle="Best and worst performers over the last month" />
          <CardBody>
            <SimpleBars data={movers.map((m) => ({ ...m, pct: +(m.return * 100).toFixed(2) }))} xKey="symbol" yKey="pct" colorBySign />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Sector Rotation" subtitle="63-day momentum leaders vs laggards" right={<Link className="text-xs text-primary hover:underline" href="/insights">Details →</Link>} />
          <CardBody className="space-y-3">
            {summary.leading_sectors.map((s) => (
              <div key={s.sector} className="flex items-center justify-between text-sm">
                <span>{s.sector}</span>
                <span className="font-semibold text-emerald-500">{fmtPct(s.momentum)}</span>
              </div>
            ))}
            <div className="border-t border-border" />
            {summary.lagging_sectors.map((s) => (
              <div key={s.sector} className="flex items-center justify-between text-sm">
                <span>{s.sector}</span>
                <span className="font-semibold text-red-500">{fmtPct(s.momentum)}</span>
              </div>
            ))}
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Top AI Recommendations"
          subtitle="Composite of forecast, momentum, trend, risk-adjusted and sector signals"
          right={<Link className="text-xs text-primary hover:underline" href="/insights">All insights →</Link>}
        />
        <CardBody>
          <Table headers={["Symbol", "Sector", "Action", "Score", "Reasoning"]}>
            {(recs ?? []).slice(0, 8).map((r) => (
              <tr key={r.symbol}>
                <td className="px-3 py-2 font-semibold">
                  <Link className="hover:text-primary" href={`/stocks?symbol=${r.symbol}`}>{r.symbol}</Link>
                </td>
                <td className="px-3 py-2 text-muted-foreground">{r.sector ?? "—"}</td>
                <td className="px-3 py-2">
                  <Badge tone={r.action.toLowerCase() as "buy" | "sell" | "hold"}>{r.action}</Badge>
                </td>
                <td className="px-3 py-2 tabular-nums">{r.score.toFixed(2)}</td>
                <td className="max-w-xl px-3 py-2 text-xs text-muted-foreground">{r.reasoning}</td>
              </tr>
            ))}
          </Table>
        </CardBody>
      </Card>
    </div>
  );
}
