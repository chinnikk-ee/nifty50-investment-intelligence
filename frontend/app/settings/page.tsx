"use client";

import { Button, Card, CardBody, CardHeader, ErrorNote, Select } from "@/components/ui";
import { API_URL, useApi } from "@/lib/api";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const { data: health, error } = useApi<{ status: string }>("/health");
  const { data: models } = useApi<{ regressors: string[]; classifiers: string[] }>("/forecast/models");
  const [downloading, setDownloading] = useState(false);
  useEffect(() => setMounted(true), []);

  const exportReport = async () => {
    setDownloading(true);
    try {
      const res = await fetch(`${API_URL}/reports/generate`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "investment_report.pdf";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Report generation failed: ${e}`);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="max-w-2xl space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground">Appearance, backend status and report export</p>
      </header>

      <Card>
        <CardHeader title="Appearance" />
        <CardBody>
          {mounted && (
            <label className="block text-sm">
              <span className="mb-1 block text-xs text-muted-foreground">Theme</span>
              <Select value={theme} onChange={(e) => setTheme(e.target.value)}>
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="system">System</option>
              </Select>
            </label>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Backend" />
        <CardBody className="space-y-2 text-sm">
          <p>
            API endpoint: <code className="rounded bg-muted px-1.5 py-0.5 text-xs">{API_URL}</code>
          </p>
          <p>
            Status:{" "}
            {health ? (
              <span className="font-semibold text-emerald-500">● online</span>
            ) : error ? (
              <span className="font-semibold text-red-500">● offline</span>
            ) : (
              "checking…"
            )}
          </p>
          {models && (
            <p className="text-xs text-muted-foreground">
              Available models — regressors: {models.regressors.join(", ")} · classifiers: {models.classifiers.join(", ")}
            </p>
          )}
          {error && <ErrorNote message={error} />}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Reports" subtitle="Full PDF: EDA, forecasts, portfolio, risk and recommendations" />
        <CardBody>
          <Button onClick={exportReport} disabled={downloading || !health}>
            {downloading ? "Generating (can take a minute)…" : "Export PDF Report"}
          </Button>
        </CardBody>
      </Card>
    </div>
  );
}
