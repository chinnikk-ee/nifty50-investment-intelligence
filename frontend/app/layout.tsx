import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import { Sidebar } from "@/components/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "NIFTY-50 Investment Intelligence",
  description:
    "AI-powered decision support over historical NIFTY-50 market data: forecasting, portfolios, risk, anomalies and explainable recommendations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          <div className="flex">
            <Sidebar />
            <main className="min-h-screen flex-1 overflow-x-hidden px-6 py-6 lg:px-8">{children}</main>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
