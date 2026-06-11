import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const fmtPct = (x: number | null | undefined, digits = 2) =>
  x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const fmtNum = (x: number | null | undefined, digits = 2) =>
  x == null || Number.isNaN(x) ? "—" : x.toLocaleString("en-IN", { maximumFractionDigits: digits });

export const fmtInr = (x: number | null | undefined) =>
  x == null ? "—" : `₹${x.toLocaleString("en-IN", { maximumFractionDigits: 2 })}`;
