"use client";

import { Badge, Card, CardBody, CardHeader, ErrorNote, Select, Skeleton, Table } from "@/components/ui";
import { useApi } from "@/lib/api";
import { fmtPct } from "@/lib/utils";
import { useState } from "react";

type Anomaly = {
  date: string; symbol: string; close: number; ret_1d: number; volume_ratio: number;
  drawdown: number; votes: number; methods: string; type: string;
};
type StockInfo = { symbol: string };

const TYPE_LABEL: Record<string, string> = {
  volatility_spike: "Volatility Spike",
  unusual_return: "Unusual Return",
  volume_surge: "Volume Surge",
  extreme_drawdown: "Extreme Drawdown",
};
const TYPE_TONE: Record<string, "buy" | "sell" | "hold" | "neutral"> = {
  volatility_spike: "hold",
  unusual_return: "neutral",
  volume_surge: "buy",
  extreme_drawdown: "sell",
};

export default function AnomaliesPage() {
  const { data: stocks } = useApi<StockInfo[]>("/stocks");
  const [symbol, setSymbol] = useState("");
  const path = symbol ? `/anomalies?symbol=${symbol}&limit=100` : "/anomalies?limit=100";
  const { data, error, loading } = useApi<Anomaly[]>(path);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Anomaly Detection</h1>
          <p className="text-sm text-muted-foreground">
            Ensemble of Isolation Forest, DBSCAN and Autoencoder — a day is flagged when ≥2 detectors agree
          </p>
        </div>
        <Select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          <option value="">All stocks (top per stock)</option>
          {(stocks ?? []).map((s) => (
            <option key={s.symbol}>{s.symbol}</option>
          ))}
        </Select>
      </header>

      {error && <ErrorNote message={error} />}
      <Card>
        <CardHeader title={`Detected Anomalies ${symbol ? `— ${symbol}` : ""}`} subtitle="Most recent first · 'methods' lists which detectors voted" />
        <CardBody>
          {loading || !data ? (
            <Skeleton className="h-96" />
          ) : data.length === 0 ? (
            <p className="py-12 text-center text-sm text-muted-foreground">No anomalies detected for this selection.</p>
          ) : (
            <Table headers={["Date", "Symbol", "Type", "1d Return", "Volume ×", "Drawdown", "Votes", "Detectors"]}>
              {data.map((a, i) => (
                <tr key={i}>
                  <td className="px-3 py-2 tabular-nums">{a.date}</td>
                  <td className="px-3 py-2 font-semibold">{a.symbol}</td>
                  <td className="px-3 py-2">
                    <Badge tone={TYPE_TONE[a.type] ?? "neutral"}>{TYPE_LABEL[a.type] ?? a.type}</Badge>
                  </td>
                  <td className={`px-3 py-2 tabular-nums ${a.ret_1d >= 0 ? "text-emerald-500" : "text-red-500"}`}>{fmtPct(a.ret_1d)}</td>
                  <td className="px-3 py-2 tabular-nums">{a.volume_ratio.toFixed(1)}×</td>
                  <td className="px-3 py-2 tabular-nums">{fmtPct(a.drawdown, 1)}</td>
                  <td className="px-3 py-2">{a.votes}/3</td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">{a.methods}</td>
                </tr>
              ))}
            </Table>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
