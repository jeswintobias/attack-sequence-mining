import { ChartHost, Tooltip, compact, toNumber, useTooltip } from "./primitives";

/**
 * Matrix heatmap on a single-hue sequential ramp.
 *
 * Sequential encoding means one hue running light to dark — never a rainbow —
 * so magnitude reads monotonically. The lightest step is allowed to recede
 * toward the surface because "near zero" should look like nothing. A 2px gap in
 * the surface colour separates cells, which is what makes neighbouring values
 * read as distinct without a stroke around each one.
 *
 * Every cell carries a hover tooltip, and the table view behind the toggle holds
 * the same numbers, so no value is reachable only by pointing at it.
 */

const STEPS = [
  "var(--ac-seq-100)",
  "var(--ac-seq-250)",
  "var(--ac-seq-400)",
  "var(--ac-seq-550)",
  "var(--ac-seq-700)",
];

export function Heatmap({
  index,
  columns,
  records,
  valueLabel,
}: {
  index: string[];
  columns: string[];
  records: Record<string, string>[];
  valueLabel: string;
}) {
  const { tip, show, hide } = useTooltip();
  const grid = records.map((row) => columns.map((column) => toNumber(row[column])));
  const flat = grid.flat();
  const max = Math.max(...flat, 1);
  const min = Math.min(...flat, 0);

  const stepFor = (value: number) => {
    if (value <= min) return 0;
    const ratio = (value - min) / Math.max(max - min, 1e-9);
    // Compress the low end: these matrices are long-tailed, and a linear split
    // would put almost every cell in the lightest bucket.
    const eased = Math.sqrt(ratio);
    return Math.min(STEPS.length - 1, Math.floor(eased * STEPS.length));
  };

  return (
    <ChartHost>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[2px]">
          <caption className="sr-only">
            {valueLabel} for each pair of techniques; {index.length} rows by{" "}
            {columns.length} columns.
          </caption>
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-surface" />
              {columns.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className="h-20 w-[22px] align-bottom"
                >
                  <span className="block origin-bottom-left translate-x-[14px] -rotate-90 whitespace-nowrap font-mono text-[10px] text-ink-muted">
                    {column}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {index.map((rowLabel, rowIndex) => (
              <tr key={rowLabel}>
                <th
                  scope="row"
                  className="sticky left-0 z-10 bg-surface pr-2 text-right font-mono text-[10px] font-normal text-ink-muted"
                >
                  {rowLabel}
                </th>
                {columns.map((column, columnIndex) => {
                  const value = grid[rowIndex]?.[columnIndex] ?? 0;
                  return (
                    <td
                      key={column}
                      className="size-[22px] rounded-[2px]"
                      style={{
                        backgroundColor:
                          value <= min ? "var(--ac-hairline)" : STEPS[stepFor(value)],
                      }}
                      onMouseMove={(event) =>
                        show(event, (
                          <>
                            <span className="font-mono">
                              {rowLabel} → {column}
                            </span>
                            <br />
                            {valueLabel}: <strong>{value.toLocaleString()}</strong>
                          </>
                        ))
                      }
                      onMouseLeave={hide}
                    >
                      <span className="sr-only">
                        {rowLabel} to {column}: {value}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex items-center gap-2 text-[11px] text-ink-secondary">
        <span>{compact(min)}</span>
        <span className="flex gap-[2px]" aria-hidden>
          {STEPS.map((step) => (
            <span key={step} className="h-2.5 w-7 rounded-[2px]" style={{ backgroundColor: step }} />
          ))}
        </span>
        <span>{compact(max)}</span>
        <span className="ml-1 text-ink-muted">{valueLabel}</span>
      </div>
      <Tooltip tip={tip} />
    </ChartHost>
  );
}
