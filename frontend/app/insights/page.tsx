"use client";

import { SimpleBars } from "@/components/charts";
import { Badge, Button, Card, CardBody, CardHeader, ErrorNote, Input, Select, Skeleton } from "@/components/ui";
import { useApi, useApiPost } from "@/lib/api";
import { cn, fmtPct } from "@/lib/utils";
import { Send } from "lucide-react";
import { useRef, useState } from "react";

type Rec = {
  symbol: string; sector?: string; action: "BUY" | "HOLD" | "SELL"; score: number;
  components: Record<string, number>; reasoning: string;
};
type Explain = {
  symbol: string; model: string; prediction: number; confidence: number; method: string;
  top_features: { feature: string; contribution: number; value: number; direction: string }[];
};
type ChatMsg = { role: "user" | "assistant"; text: string };

export default function InsightsPage() {
  const { data: recs, error, loading } = useApi<Rec[]>("/recommendations");
  const [selected, setSelected] = useState<Rec | null>(null);
  const explain = useApiPost<Explain>();

  const [messages, setMessages] = useState<ChatMsg[]>([
    { role: "assistant", text: "Ask me about the platform's analysis — e.g. 'should I buy INFY?', 'how risky is TCS?', 'build me a conservative portfolio'." },
  ]);
  const [question, setQuestion] = useState("");
  const chat = useApiPost<{ answer: string }>();
  const chatEnd = useRef<HTMLDivElement>(null);

  const pick = async (rec: Rec) => {
    setSelected(rec);
    await explain.run("/explainability", { symbol: rec.symbol, horizon: 20, model: "random_forest" });
  };

  const ask = async () => {
    const q = question.trim();
    if (!q) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setQuestion("");
    const res = await chat.run("/chat", { question: q });
    setMessages((m) => [...m, { role: "assistant", text: res?.answer ?? `Error: ${chat.error}` }]);
    setTimeout(() => chatEnd.current?.scrollIntoView({ behavior: "smooth" }), 50);
  };

  if (error) return <ErrorNote message={error} />;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">AI Insights</h1>
        <p className="text-sm text-muted-foreground">Explainable recommendations, SHAP attributions and the insight assistant</p>
      </header>

      <div className="grid gap-4 lg:grid-cols-[1fr_400px]">
        <div className="space-y-4">
          <Card>
            <CardHeader title="Recommendations" subtitle="Click a stock for the SHAP explanation of its forecast" />
            <div className="max-h-[60vh] divide-y divide-border/60 overflow-y-auto">
              {loading || !recs
                ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="m-3 h-14" />)
                : recs.map((r) => (
                    <button
                      key={r.symbol}
                      onClick={() => pick(r)}
                      className={cn("w-full px-5 py-3 text-left hover:bg-muted", selected?.symbol === r.symbol && "bg-primary/10")}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{r.symbol}</span>
                        <span className="flex items-center gap-2">
                          <span className="text-xs tabular-nums text-muted-foreground">{r.score.toFixed(2)}</span>
                          <Badge tone={r.action.toLowerCase() as "buy" | "sell" | "hold"}>{r.action}</Badge>
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">{r.reasoning}</p>
                    </button>
                  ))}
            </div>
          </Card>

          {selected && (
            <Card>
              <CardHeader
                title={`Why? — ${selected.symbol}`}
                subtitle={
                  explain.data
                    ? `${explain.data.method.toUpperCase()} attribution · 20d predicted return ${fmtPct(explain.data.prediction)} · confidence ${fmtPct(explain.data.confidence, 0)}`
                    : "Signal components and model attribution"
                }
              />
              <CardBody className="space-y-4">
                <SimpleBars
                  data={Object.entries(selected.components).map(([k, v]) => ({ signal: k, value: v }))}
                  xKey="signal"
                  yKey="value"
                  colorBySign
                  height={200}
                />
                {explain.loading && <Skeleton className="h-48" />}
                {explain.error && <ErrorNote message={explain.error} />}
                {explain.data && (
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Top feature contributions to the model forecast
                    </p>
                    <SimpleBars
                      data={explain.data.top_features.map((f) => ({ feature: f.feature, contribution: f.contribution }))}
                      xKey="feature"
                      yKey="contribution"
                      colorBySign
                      height={220}
                    />
                  </div>
                )}
              </CardBody>
            </Card>
          )}
        </div>

        <Card className="flex h-[78vh] flex-col">
          <CardHeader title="Insight Assistant" subtitle="Grounded only in this platform's computed analytics" />
          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
            {messages.map((m, i) => (
              <div
                key={i}
                className={cn(
                  "max-w-[90%] rounded-xl px-3.5 py-2.5 text-sm",
                  m.role === "user" ? "ml-auto bg-primary text-primary-foreground" : "bg-muted"
                )}
              >
                {m.text}
              </div>
            ))}
            {chat.loading && <div className="rounded-xl bg-muted px-3.5 py-2.5 text-sm text-muted-foreground">Thinking…</div>}
            <div ref={chatEnd} />
          </div>
          <div className="flex gap-2 border-t border-border p-3">
            <Input
              placeholder="e.g. forecast RELIANCE"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
            />
            <Button onClick={ask} disabled={chat.loading} aria-label="Send">
              <Send size={15} />
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
