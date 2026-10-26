import { ChartHost, Legend, SERIES, Tooltip, compact, useTooltip } from "./primitives";

export interface BarRow {
  label: string;
  values: number[];
}

/**
 * Horizontal bars, one row per category.
 *
 * Horizontal rather than vertical because the categories here are technique
 * identifiers and names — text that reads along the axis without rotation.
 * Bars are capped at 24px with a 4px rounded data-end and a square start at the
 * baseline; a 2px surface gap separates the two series within a row, so they
 * read as distinct without a stroke around either.
 *
 * Values are labelled at the tip only when they fit, and the axis carries the
 * rest. The table view holds every number regardless, so nothing is gated
 * behind a hover.
 */
export function BarChart({
  rows,
  seriesNames,
  valueFormat = compact,
}: {
  rows: BarRow[];
  seriesNames: string[];
  valueFormat?: (value: number) => string;
}) {
  const { tip, show, hide } = useTooltip();
  const seriesCount = Math.min(seriesNames.length, SERIES.length);
  const max = Math.max(...rows.flatMap((row) => row.values.slice(0, seriesCount)), 1);

  // Bars capped at 24px; with two series the pair shares the slot plus a 2px gap.
  const barThickness = seriesCount > 1 ? 9 : 14;
  const rowHeight = seriesCount * barThickness + (seriesCount - 1) * 2 + 14;

  return (
    <ChartHost>
      <div className="space-y-0">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-center gap-3 rounded hover:bg-plane"
            style={{ minHeight: rowHeight }}
          >
            <span
              className="w-[8.5rem] shrink-0 truncate text-right font-mono text-[11px] text-ink-secondary"
              title={row.label}
            >
              {row.label}
            </span>
            <div className="flex min-w-0 flex-1 flex-col gap-0.5 py-1">
              {row.values.slice(0, seriesCount).map((value, index) => {
                const width = max > 0 ? Math.max((value / max) * 100, value > 0 ? 0.6 : 0) : 0;
                return (
                  <div key={index} className="flex items-center gap-2">
                    <div className="h-[9px] min-w-0 flex-1">
                      <div
                        onMouseMove={(event) =>
                          show(
                            event,
                            <>
                              <span className="font-mono">{row.label}</span>
                              <br />
                              {seriesNames[index]}:{" "}
                              <strong>{value.toLocaleString()}</strong>
                            </>,
                          )
                        }
                        onMouseLeave={hide}
                        /* square at the baseline, 4px rounded data-end */
                        className="h-full rounded-r-[4px] transition-[width] duration-300"
                        style={{
                          width: `${width}%`,
                          backgroundColor: SERIES[index],
                          height: barThickness,
                        }}
                      />
                    </div>
                    <span className="w-14 shrink-0 text-right font-mono text-[11px] text-ink-secondary">
                      {valueFormat(value)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <Legend
        items={seriesNames
          .slice(0, seriesCount)
          .map((name, index) => ({ label: name, color: SERIES[index]! }))}
      />
      <Tooltip tip={tip} />
    </ChartHost>
  );
}
