"use client";

/** Recharts wrappers with consistent theming across the dashboard. */
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"];

// Theme-aware: reads the same CSS variables the rest of the app uses, so the
// tooltip follows light/dark instead of being hardcoded dark.
const tooltipStyle = {
  backgroundColor: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 8,
  fontSize: 12,
  color: "hsl(var(--card-foreground))",
};
// contentStyle.color does not reach the label/value rows — set them explicitly
// or the label text falls back to a dark default and vanishes on the dark box.
const tooltipLabelStyle = { color: "hsl(var(--card-foreground))", fontWeight: 600 };
const tooltipItemStyle = { color: "hsl(var(--card-foreground))" };

export function PriceChart({
  data,
  series,
  xKey = "date",
  height = 320,
}: {
  data: Record<string, unknown>[];
  series: { key: string; label?: string; color?: string }[];
  xKey?: string;
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
        <XAxis dataKey={xKey} tick={{ fontSize: 10 }} minTickGap={48} />
        <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} width={70} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
        {series.map((s, i) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label ?? s.key}
            stroke={s.color ?? PALETTE[i % PALETTE.length]}
            strokeWidth={1.6}
            dot={false}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function BandChart({
  data,
  height = 320,
}: {
  data: { day: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
  height?: number;
}) {
  // Stacked-band trick: invisible base + deltas render percentile fans.
  const shaped = data.map((d) => ({
    ...d,
    base: d.p5,
    inner: d.p75 - d.p25,
    outer: d.p95 - d.p5,
    innerBase: d.p25,
  }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={shaped} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
        <XAxis dataKey="day" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} width={70} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
        <Area dataKey="base" stackId="o" stroke="none" fill="transparent" />
        <Area dataKey="outer" stackId="o" stroke="none" fill="#3b82f6" fillOpacity={0.12} name="5–95%" />
        <Area dataKey="innerBase" stackId="i" stroke="none" fill="transparent" />
        <Area dataKey="inner" stackId="i" stroke="none" fill="#3b82f6" fillOpacity={0.25} name="25–75%" />
        <Line type="monotone" dataKey="p50" stroke="#3b82f6" strokeWidth={2} dot={false} name="median" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function AllocationPie({ allocation, height = 300 }: { allocation: Record<string, number>; height?: number }) {
  const data = Object.entries(allocation).map(([name, value]) => ({ name, value }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius="45%" outerRadius="78%" paddingAngle={1.5}>
          {data.map((_, i) => (
            <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={tooltipLabelStyle}
          itemStyle={tooltipItemStyle}
          formatter={(v: number) => `${v.toFixed(2)}%`}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

export function SimpleBars({
  data,
  xKey,
  yKey,
  height = 280,
  colorBySign = false,
}: {
  data: Record<string, unknown>[];
  xKey: string;
  yKey: string;
  height?: number;
  colorBySign?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.15} />
        <XAxis dataKey={xKey} tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" height={60} />
        <YAxis tick={{ fontSize: 10 }} width={70} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
        <Bar dataKey={yKey} radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={colorBySign ? (Number(d[yKey]) >= 0 ? "#10b981" : "#ef4444") : PALETTE[i % PALETTE.length]}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
