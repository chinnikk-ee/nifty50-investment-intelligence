"use client";

import { AllocationPie, PriceChart } from "@/components/charts";
import { Button, Card, CardBody, CardHeader, ErrorNote, Select, StatCard } from "@/components/ui";
import { useApiPost } from "@/lib/api";
import { fmtInr, fmtPct } from "@/lib/utils";
import { useState } from "react";

type Portfolio = {
  profile: string; profile_label: string; profile_description: string; method: string;
  allocation: Record<string, number>; expected_return: number; volatility: number; sharpe: number;
};
type Backtest = {
  final_value: number; total_return: number; cagr: number; benchmark_final_value: number;
  metrics: Record<string, number>;
  equity_curve: { date: string; portfolio: number; benchmark: number }[];
};
type QuizResult = { profile: string; score: number; rationale: string };

const METHODS = ["", "equal_weight", "risk_parity", "mean_variance", "max_sharpe", "min_volatility"];

const QUESTIONS = [
  { key: "loss_tolerance", label: "If your portfolio dropped 20%, you would…", opts: ["Sell everything", "Sell some", "Hold", "Buy more"] },
  { key: "experience", label: "Your investing experience", opts: ["None", "Beginner", "Intermediate", "Advanced"] },
  { key: "income_stability", label: "Your income is…", opts: ["Unstable", "Somewhat stable", "Stable", "Very stable"] },
  { key: "goal", label: "Your primary goal", opts: ["Preserve capital", "Steady income", "Balanced growth", "Maximize growth"] },
] as const;

export default function PortfolioPage() {
  const [profile, setProfile] = useState("balanced");
  const [method, setMethod] = useState("");
  const [answers, setAnswers] = useState<Record<string, number>>({ horizon_years: 5, loss_tolerance: 2, experience: 2, income_stability: 3, goal: 3 });

  const portfolio = useApiPost<Portfolio>();
  const backtest = useApiPost<Backtest>();
  const quiz = useApiPost<QuizResult>();

  const build = (p = profile, m = method) =>
    portfolio.run("/portfolio", { profile: p, method: m || null });

  const runQuiz = async () => {
    const res = await quiz.run("/portfolio/questionnaire", answers);
    if (res) {
      setProfile(res.profile);
      await build(res.profile, "");
    }
  };

  const runBacktest = () => {
    if (!portfolio.data) return;
    const weights = Object.fromEntries(Object.entries(portfolio.data.allocation).map(([s, w]) => [s, w / 100]));
    backtest.run("/analytics/backtest", { weights, rebalance_days: 21, transaction_cost_bps: 10, initial_capital: 1000000 });
  };

  const p = portfolio.data;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Portfolio Builder</h1>
        <p className="text-sm text-muted-foreground">Profile-driven optimization with backtesting</p>
      </header>

      <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader title="Risk Questionnaire" subtitle="Answer five questions to find your investor profile" />
            <CardBody className="space-y-3">
              <label className="block text-sm">
                <span className="mb-1 block text-xs text-muted-foreground">Investment horizon (years)</span>
                <Select value={answers.horizon_years} onChange={(e) => setAnswers((a) => ({ ...a, horizon_years: +e.target.value }))} className="w-full">
                  {[1, 3, 5, 10, 20].map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </Select>
              </label>
              {QUESTIONS.map((q) => (
                <label key={q.key} className="block text-sm">
                  <span className="mb-1 block text-xs text-muted-foreground">{q.label}</span>
                  <Select value={answers[q.key]} onChange={(e) => setAnswers((a) => ({ ...a, [q.key]: +e.target.value }))} className="w-full">
                    {q.opts.map((o, i) => (
                      <option key={o} value={i + 1}>{o}</option>
                    ))}
                  </Select>
                </label>
              ))}
              <Button onClick={runQuiz} disabled={quiz.loading} className="w-full">
                {quiz.loading ? "Scoring…" : "Find My Profile"}
              </Button>
              {quiz.data && <p className="text-xs text-muted-foreground">{quiz.data.rationale}</p>}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Build Portfolio" />
            <CardBody className="space-y-3">
              <label className="block text-sm">
                <span className="mb-1 block text-xs text-muted-foreground">Investor profile</span>
                <Select value={profile} onChange={(e) => setProfile(e.target.value)} className="w-full">
                  <option value="conservative">Conservative</option>
                  <option value="balanced">Balanced</option>
                  <option value="aggressive">Aggressive</option>
                </Select>
              </label>
              <label className="block text-sm">
                <span className="mb-1 block text-xs text-muted-foreground">Method (blank = profile default)</span>
                <Select value={method} onChange={(e) => setMethod(e.target.value)} className="w-full">
                  {METHODS.map((m) => (
                    <option key={m} value={m}>{m || "profile default"}</option>
                  ))}
                </Select>
              </label>
              <Button onClick={() => build()} disabled={portfolio.loading} className="w-full">
                {portfolio.loading ? "Optimizing…" : "Build Portfolio"}
              </Button>
              {portfolio.error && <ErrorNote message={portfolio.error} />}
            </CardBody>
          </Card>
        </div>

        <div className="space-y-4">
          {p ? (
            <>
              <div className="grid gap-3 sm:grid-cols-3">
                <StatCard label="Expected Return" value={fmtPct(p.expected_return)} tone="up" hint="annualized, shrunk estimate" />
                <StatCard label="Volatility" value={fmtPct(p.volatility)} hint="annualized" />
                <StatCard label="Sharpe Ratio" value={p.sharpe.toFixed(2)} hint="risk-free 6%" />
              </div>
              <Card>
                <CardHeader title={`${p.profile_label} Allocation — ${p.method.replace(/_/g, " ")}`} subtitle={p.profile_description} right={<Button variant="outline" onClick={runBacktest} disabled={backtest.loading}>{backtest.loading ? "Backtesting…" : "Backtest"}</Button>} />
                <CardBody>
                  <AllocationPie allocation={p.allocation} />
                </CardBody>
              </Card>
            </>
          ) : (
            <Card>
              <CardBody className="py-16 text-center text-sm text-muted-foreground">
                Build a portfolio (or take the questionnaire) to see allocations here.
              </CardBody>
            </Card>
          )}

          {backtest.data && (
            <Card>
              <CardHeader
                title="Backtest vs Equal-Weight Benchmark"
                subtitle={`Final ${fmtInr(backtest.data.final_value)} (${fmtPct(backtest.data.total_return)}, CAGR ${fmtPct(backtest.data.cagr)}) · benchmark ${fmtInr(backtest.data.benchmark_final_value)} · max DD ${fmtPct(backtest.data.metrics.max_drawdown)}`}
              />
              <CardBody>
                <PriceChart
                  data={backtest.data.equity_curve}
                  series={[
                    { key: "portfolio", label: "Portfolio", color: "#3b82f6" },
                    { key: "benchmark", label: "Equal-weight", color: "#94a3b8" },
                  ]}
                />
              </CardBody>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
