"use client";

import { PriceChart } from "@/components/charts";
import { Card, CardBody, CardHeader, ErrorNote, Input, Select, Skeleton, StatCard } from "@/components/ui";
import { useApi } from "@/lib/api";
import { cn, fmtInr, fmtPct } from "@/lib/utils";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";

type StockInfo = { symbol: string; company: string; sector: string; last_close: number; return_1y: number; volatility_1y: number };
type Detail = {
  symbol: string;
  company: string;
  sector: string;
  prices: { date: string; close: number; volume: number }[];
  indicators: Record<string, (number | null)[]>;
};

function Explorer() {
  const params = useSearchParams();
  const [search, setSearch] = useState("");
  const [sector, setSector] = useState("");
  const [selected, setSelected] = useState(params.get("symbol") ?? "");

  const { data: stocks, error } = useApi<StockInfo[]>("/stocks");
  const { data: sectors } = useApi<string[]>("/sectors");
  const symbol = selected || stocks?.[0]?.symbol || null;
  const { data: detail, loading: detailLoading } = useApi<Detail>(symbol ? `/stocks/${symbol}` : null);

  const filtered = useMemo(
    () =>
      (stocks ?? []).filter(
        (s) =>
          (!sector || s.sector === sector) &&
          (!search || s.symbol.toLowerCase().includes(search.toLowerCase()) || s.company.toLowerCase().includes(search.toLowerCase()))
      ),
    [stocks, search, sector]
  );

  const chartData = useMemo(() => {
    if (!detail) return [];
    return detail.prices.map((p, i) => ({
      date: p.date,
      close: p.close,
      ma50: detail.indicators.ma50?.[i] ?? null,
      ma200: detail.indicators.ma200?.[i] ?? null,
      bb_upper: detail.indicators.bb_upper?.[i] ?? null,
      bb_lower: detail.indicators.bb_lower?.[i] ?? null,
    }));
  }, [detail]);

  const rsiData = useMemo(() => {
    if (!detail) return [];
    return detail.prices.map((p, i) => ({ date: p.date, rsi: detail.indicators.rsi14?.[i] ?? null }));
  }, [detail]);

  if (error) return <ErrorNote message={error} />;

  const info = stocks?.find((s) => s.symbol === symbol);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Stock Explorer</h1>
          <p className="text-sm text-muted-foreground">Search the universe, filter by sector, inspect indicators</p>
        </div>
        <div className="flex gap-2">
          <Input placeholder="Search symbol or company…" value={search} onChange={(e) => setSearch(e.target.value)} className="w-56" />
          <Select value={sector} onChange={(e) => setSector(e.target.value)}>
            <option value="">All sectors</option>
            {(sectors ?? []).map((s) => (
              <option key={s}>{s}</option>
            ))}
          </Select>
        </div>
      </header>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Card className="max-h-[75vh] overflow-y-auto">
          <CardHeader title={`Stocks (${filtered.length})`} />
          <div className="divide-y divide-border/60">
            {filtered.map((s) => (
              <button
                key={s.symbol}
                onClick={() => setSelected(s.symbol)}
                className={cn(
                  "flex w-full items-center justify-between px-4 py-2.5 text-left text-sm hover:bg-muted",
                  s.symbol === symbol && "bg-primary/10"
                )}
              >
                <span>
                  <span className="font-semibold">{s.symbol}</span>
                  <span className="block text-[11px] text-muted-foreground">{s.sector}</span>
                </span>
                <span className={cn("text-xs font-semibold tabular-nums", s.return_1y >= 0 ? "text-emerald-500" : "text-red-500")}>
                  {fmtPct(s.return_1y, 1)}
                </span>
              </button>
            ))}
          </div>
        </Card>

        <div className="space-y-4">
          {info && (
            <div className="grid gap-3 sm:grid-cols-3">
              <StatCard label={info.symbol} value={fmtInr(info.last_close)} hint={info.company} />
              <StatCard label="1Y Return" value={fmtPct(info.return_1y)} tone={info.return_1y >= 0 ? "up" : "down"} />
              <StatCard label="1Y Volatility" value={fmtPct(info.volatility_1y)} hint="annualized" />
            </div>
          )}
          <Card>
            <CardHeader title={`${symbol ?? ""} — Price, MAs & Bollinger Bands`} subtitle="Last 3 trading years" />
            <CardBody>
              {detailLoading || !detail ? (
                <Skeleton className="h-80" />
              ) : (
                <PriceChart
                  data={chartData}
                  series={[
                    { key: "close", label: "Close", color: "#3b82f6" },
                    { key: "ma50", label: "MA50", color: "#f59e0b" },
                    { key: "ma200", label: "MA200", color: "#8b5cf6" },
                    { key: "bb_upper", label: "BB upper", color: "#475569" },
                    { key: "bb_lower", label: "BB lower", color: "#475569" },
                  ]}
                />
              )}
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="RSI (14)" subtitle="Above 70 overbought · below 30 oversold" />
            <CardBody>
              {detailLoading || !detail ? <Skeleton className="h-40" /> : <PriceChart data={rsiData} series={[{ key: "rsi", label: "RSI", color: "#10b981" }]} height={160} />}
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default function StocksPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96" />}>
      <Explorer />
    </Suspense>
  );
}
