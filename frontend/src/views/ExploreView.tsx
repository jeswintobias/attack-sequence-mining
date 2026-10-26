import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { Pill } from "../components/ui/Badge";
import { ErrorState, LoadingState } from "../components/states/States";

const KIND_LABEL: Record<string, string> = {
  table: "Table",
  indexed_table: "Series",
  matrix: "Heatmap",
};

/**
 * Stage 5 renders these interactively. For now the index proves the EDA
 * endpoints are wired and shows what the Explore tab will contain.
 */
export function ExploreView() {
  const { data, error, loading } = useAsync(() => api.edaIndex(), []);

  if (loading) return <LoadingState label="Loading exploratory analysis" />;
  if (error) return <ErrorState error={error} />;

  return (
    <Card>
      <CardHeader
        title="Exploratory analysis"
        description={`${data?.length ?? 0} charts from the sightings dataset.`}
      />
      <CardBody className="p-0">
        <ul className="divide-y divide-hairline">
          {data?.map((chart) => (
            <li
              key={chart.id}
              className="flex items-center gap-4 px-5 py-3 text-[13px] hover:bg-plane"
            >
              <span className="min-w-0 flex-1 truncate text-ink">{chart.title}</span>
              <Pill className="border border-hairline text-ink-secondary">
                {KIND_LABEL[chart.kind] ?? chart.kind}
              </Pill>
              <span className="w-20 text-right font-mono text-[12px] text-ink-muted">
                {chart.rows} rows
              </span>
            </li>
          ))}
        </ul>
      </CardBody>
    </Card>
  );
}
