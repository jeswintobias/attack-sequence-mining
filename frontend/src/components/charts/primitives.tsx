import { useCallback, useState, type ReactNode } from "react";
import { cn } from "../../lib/utils";

/**
 * Chart primitives, hand-built rather than pulled from a charting library.
 *
 * Two reasons. The mark specs this project follows are precise — bars capped at
 * 24px with a 4px rounded data-end square at the baseline, 2px lines, markers of
 * at least 8px carrying a 2px surface ring, a 2px surface gap between touching
 * fills, hairline recessive gridlines — and hitting them exactly through a
 * library's theming layer is more work than drawing the SVG. Second, every
 * colour here is a design token, so light and dark stay in step automatically.
 *
 * Series colours are restricted to categorical slots 1 and 2 (blue, orange).
 * Both clear 3:1 on the light surface; slots 3, 4 and 5 do not, and using them
 * would oblige extra relief. Any chart needing more than two series folds the
 * remainder into the table view instead.
 */

export const SERIES = ["var(--ac-series-1)", "var(--ac-series-2)"] as const;

/* ---------------------------------------------------------------- tooltip */

export interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

export function useTooltip() {
  const [tip, setTip] = useState<TooltipState | null>(null);
  const show = useCallback(
    (event: React.MouseEvent, content: ReactNode) => {
      const host = event.currentTarget.closest("[data-chart-host]") as HTMLElement | null;
      const box = host?.getBoundingClientRect();
      setTip({
        x: event.clientX - (box?.left ?? 0),
        y: event.clientY - (box?.top ?? 0),
        content,
      });
    },
    [],
  );
  const hide = useCallback(() => setTip(null), []);
  return { tip, show, hide };
}

export function Tooltip({ tip }: { tip: TooltipState | null }) {
  if (!tip) return null;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 max-w-[16rem] rounded-md border border-hairline bg-raised px-2.5 py-1.5 text-[12px] leading-snug text-ink shadow-lg"
      style={{
        left: Math.max(tip.x + 12, 4),
        top: Math.max(tip.y - 8, 4),
      }}
    >
      {tip.content}
    </div>
  );
}

/* ------------------------------------------------------------------ chrome */

export function ChartHost({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div data-chart-host className={cn("relative", className)}>
      {children}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string }[] }) {
  // A legend is present for two or more series; a single series needs none,
  // because the chart's title already names what is plotted.
  if (items.length < 2) return null;
  return (
    <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5 text-[12px] text-ink-secondary">
          <span
            aria-hidden
            className="size-2.5 rounded-sm"
            style={{ backgroundColor: item.color }}
          />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

/** Compact axis numbers: 1,200 renders as 1.2K so ticks stay narrow. */
export function compact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)}K`;
  if (abs > 0 && abs < 1) return value.toFixed(2);
  return String(Math.round(value));
}

export function toNumber(value: unknown): number {
  const parsed = typeof value === "number" ? value : parseFloat(String(value ?? ""));
  return Number.isFinite(parsed) ? parsed : 0;
}
