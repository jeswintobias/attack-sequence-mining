import { useMemo, useState } from "react";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import type { EdaChart } from "../lib/types";
import { cn } from "../lib/utils";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { ErrorState, LoadingState } from "../components/states/States";
import { Skeleton } from "../components/ui/Skeleton";
import { ChartCard, FORMS } from "../components/charts/ChartRegistry";
import { ModelPerformance } from "../components/ModelPerformance";

type Section = "data" | "model";

function ChartLoader({ id }: { id: string }) {
  const { data, error, loading } = useAsync(() => api.edaChart(id), [id]);
  if (loading) return <Skeleton className="h-64 rounded-xl" />;
  if (error) return <ErrorState error={error} />;
  if (!data) return null;
  return <ChartCard chart={data} />;
}

export function ExploreView() {
  const index = useAsync(() => api.edaIndex(), []);
  const card = useAsync(() => api.meta(), []);
  const [section, setSection] = useState<Section>("data");
  const [selected, setSelected] = useState<string | null>(null);

  // Charts with a chosen visual form lead; lookup tables follow.
  const ordered = useMemo(() => {
    const charts = index.data ?? [];
    const rank = (chart: EdaChart) => (FORMS[chart.id] ? 0 : 1);
    return [...charts].sort((a, b) => rank(a) - rank(b) || a.id.localeCompare(b.id));
  }, [index.data]);

  const visible = selected ? ordered.filter((chart) => chart.id === selected) : ordered;

  if (index.loading || card.loading) return <LoadingState label="Loading exploratory analysis" />;
  if (index.error) return <ErrorState error={index.error} />;

  return (
    <div className="space-y-5">
      <div
        role="radiogroup"
        aria-label="Explore section"
        className="flex gap-1 rounded-lg border border-hairline bg-surface p-0.5"
        style={{ width: "fit-content" }}
      >
        {(
          [
            ["data", "The dataset"],
            ["model", "The evaluation"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            role="radio"
            aria-checked={section === value}
            onClick={() => setSection(value)}
            className={cn(
              "rounded-md px-3 py-1.5 text-[13px] font-medium",
              section === value
                ? "bg-accent text-accent-ink"
                : "text-ink-secondary hover:bg-plane hover:text-ink",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {section === "model" && card.data && <ModelPerformance card={card.data} />}

      {section === "data" && (
        <>
          <Card>
            <CardHeader
              title="Exploratory analysis"
              description={`${ordered.length} views of the sightings data the model was trained on.`}
            />
            <CardBody>
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={() => setSelected(null)}
                  aria-pressed={selected === null}
                  className={cn(
                    "rounded-md border px-2.5 py-1 text-[12px]",
                    selected === null
                      ? "border-accent bg-accent/10 text-accent"
                      : "border-hairline text-ink-secondary hover:bg-plane hover:text-ink",
                  )}
                >
                  All
                </button>
                {ordered.map((chart) => (
                  <button
                    key={chart.id}
                    onClick={() => setSelected(chart.id)}
                    aria-pressed={selected === chart.id}
                    className={cn(
                      "rounded-md border px-2.5 py-1 text-[12px]",
                      selected === chart.id
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-hairline text-ink-secondary hover:bg-plane hover:text-ink",
                    )}
                  >
                    {chart.title}
                  </button>
                ))}
              </div>
            </CardBody>
          </Card>

          <div className="space-y-5">
            {visible.map((chart) => (
              <ChartLoader key={chart.id} id={chart.id} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
