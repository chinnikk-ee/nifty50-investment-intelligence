"use client";

import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  BrainCircuit,
  Briefcase,
  CandlestickChart,
  LayoutDashboard,
  LineChart,
  Moon,
  Settings,
  ShieldAlert,
  Sun,
  TrendingUp,
} from "lucide-react";
import { useTheme } from "next-themes";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/stocks", label: "Stock Explorer", icon: CandlestickChart },
  { href: "/forecasting", label: "Forecasting", icon: LineChart },
  { href: "/portfolio", label: "Portfolio Builder", icon: Briefcase },
  { href: "/risk", label: "Risk Analytics", icon: ShieldAlert },
  { href: "/anomalies", label: "Anomaly Detection", icon: AlertTriangle },
  { href: "/insights", label: "AI Insights", icon: BrainCircuit },
  { href: "/settings", label: "Settings", icon: Settings },
];

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="h-9 w-9" />;
  return (
    <button
      aria-label="Toggle dark mode"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      className="flex h-9 w-9 items-center justify-center rounded-lg border border-border hover:bg-muted"
    >
      {resolvedTheme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="flex h-screen w-60 shrink-0 flex-col border-r border-border bg-card max-md:hidden">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
          <TrendingUp size={18} />
        </span>
        <div>
          <p className="text-sm font-bold leading-tight">NIFTY-50 Intel</p>
          <p className="text-[11px] text-muted-foreground">Investment Intelligence</p>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-3">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
              pathname === href && "bg-primary/10 text-primary"
            )}
          >
            <Icon size={16} />
            {label}
          </Link>
        ))}
      </nav>
      <div className="flex items-center justify-between border-t border-border px-5 py-4">
        <p className="text-[11px] text-muted-foreground">Historical data only</p>
        <ThemeToggle />
      </div>
    </aside>
  );
}
