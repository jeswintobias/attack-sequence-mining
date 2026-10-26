import { useState } from "react";
import { BarChart3, Table2 } from "lucide-react";
import type { EdaChartData } from "../../lib/types";
import { cn } from "../../lib/utils";
import { BarChart } from "./BarChart";
import { DataTable } from "./DataTable";
import { Heatmap } from "./Heatmap";
import { LineChart } from "./LineChart";
import { toNumber } from "./primitives";

/**
 * Which form each chart takes.
 *
 * The form is chosen per chart from the data's job rather than inferred from
 * its shape: magnitude comparisons become bars, a long-tailed distribution
 * across ranks becomes a log line, pairwise matrices become heatmaps, and
 * anything whose job is lookup stays a table. Entries absent from this registry
 * render as tables, which is the right default rather than a failure.
 */
interface FormSpec {
  form: "bar" | "line" | "heatmap" | "table";
  series?: string[];
  valueLabel?: string;
  limit?: number;
  logY?: boolean;
  note?: string;
}

export const FORMS: Record<string, FormSpec> = {
  "01_length_class_share": { form: "bar", series: ["count"], valueLabel: "rows" },
  "02_length_hist_ge4": { form: "bar", series: ["count"], valueLabel: "sequences" },
  "03_top20_techniques": {
    form: "bar",
    series: ["all_rows", "long_seqs"],
    valueLabel: "occurrences",
    note: "Two series: every sighting row against multi-stage sequences only.",
  },
  "04_zipf_rank_freq": {
    form: "line",
    logY: true,
    valueLabel: "frequency",
    note: "A straight line here would mean the frequencies follow a power law.",
  },
  "05_monthly_volume": { form: "line", valueLabel: "sightings" },
  "06_length_by_year": { form: "bar", series: ["mean"], valueLabel: "mean length" },
  "09_cooccurrence_lift": { form: "heatmap", valueLabel: "lift" },
  "10_transition_asymmetry": { form: "heatmap", valueLabel: "asymmetry" },
  "10_transition_count": { form: "heatmap", valueLabel: "transitions" },
  "11_next_label_dist": { form: "bar", limit: 20, valueLabel: "occurrences" },
  "12_subtechnique_depth": { form: "bar", valueLabel: "count" },
};

function numericColumns(chart: EdaChartData): string[] {
  return chart.columns.filter((column) =>
    chart.records.some((record) => {
      const raw = record[column];
      return raw !== undefined && raw !== "" && Number.isFinite(parseFloat(raw));
    }),
  );
}

function ChartBody({ chart, spec }: { chart: EdaChartData; spec: FormSpec }) {
  const numeric = numericColumns(chart);

  if (spec.form === "heatmap" && chart.index) {
    return (
      <Heatmap
        index={chart.index}
        columns={chart.columns}
        records={chart.records}
        valueLabel={spec.valueLabel ?? "value"}
      />
    );
  }

  if (spec.form === "line") {
    const valueColumn = spec.series?.[0] ?? numeric[0];
    if (!valueColumn) return null;
    const labelColumn = chart.index ? null : chart.columns[0];
    const points = chart.records.map((record, position) => ({
      label: chart.index?.[position] ?? String(record[labelColumn!] ?? position),
      x: position,
      y: toNumber(record[valueColumn]),
    }));
    return <LineChart points={points} yLabel={spec.valueLabel ?? valueColumn} logY={spec.logY} />;
  }

  if (spec.form === "bar") {
    const series = (spec.series ?? numeric.slice(0, 2)).filter((name) =>
      chart.columns.includes(name),
    );
    if (!series.length) return null;
    const labelColumn = chart.index ? null : chart.columns[0];
    const rows = chart.records
      .map((record, position) => ({
        label: chart.index?.[position] ?? String(record[labelColumn!] ?? position),
        values: series.map((name) => toNumber(record[name])),
      }))
      .sort((a, b) => (b.values[0] ?? 0) - (a.values[0] ?? 0))
      .slice(0, spec.limit ?? 20);
    return <BarChart rows={rows} seriesNames={series} />;
  }

  return null;
}

export function ChartCard({ chart }: { chart: EdaChartData }) {
  const spec = FORMS[chart.id] ?? { form: "table" as const };
  const canPlot = spec.form !== "table";
  const [view, setView] = useState<"chart" | "table">(canPlot ? "chart" : "table");

  return (
    <section className="rounded-xl border border-hairline bg-surface">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-hairline px-5 py-3.5">
        <div className="min-w-0">
          <h3 className="text-[14px] font-semibold tracking-tight text-ink">{chart.title}</h3>
          {spec.note && (
            <p className="mt-0.5 text-[12px] leading-snug text-ink-secondary">{spec.note}</p>
          )}
        </div>
        {canPlot && (
          <div
            role="radiogroup"
            aria-label={`View ${chart.title} as`}
            className="flex shrink-0 gap-0.5 rounded-lg border border-hairline p-0.5"
          >
            {([["chart", BarChart3, "Chart"], ["table", Table2, "Table"]] as const).map(
              ([value, Icon, label]) => (
                <button
                  key={value}
                  role="radio"
                  aria-checked={view === value}
                  aria-label={label}
                  title={label}
                  onClick={() => setView(value)}
                  className={cn(
                    "rounded-md p-1.5",
                    view === value
                      ? "bg-accent text-accent-ink"
                      : "text-ink-muted hover:bg-plane hover:text-ink",
                  )}
                >
                  <Icon aria-hidden className="size-3.5" />
                </button>
              ),
            )}
          </div>
        )}
      </header>
      <div className="px-5 py-4">
        {view === "chart" ? (
          <ChartBody chart={chart} spec={spec} />
        ) : (
          <DataTable
            columns={chart.columns}
            records={chart.records}
            index={chart.index}
            indexLabel={chart.kind === "matrix" ? "From" : "Key"}
          />
        )}
      </div>
    </section>
  );
}
