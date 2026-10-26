import { CircleCheck, FlaskConical } from "lucide-react";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import { Skeleton } from "./ui/Skeleton";

/**
 * The model card, visible in the product rather than buried in a report.
 *
 * It states the measured accuracy, that the test split was never used, and that
 * top-5 is the number to read. Putting it in the chrome means no screen can show
 * a prediction without its provenance being one glance away.
 */
export function ModelStrip() {
  const { data, loading } = useAsync(() => api.meta(), []);

  if (loading) {
    return (
      <div className="border-b border-hairline bg-surface px-6 py-2">
        <Skeleton className="mx-auto h-4 max-w-6xl" />
      </div>
    );
  }
  if (!data) return null;

  const measured = data.evaluation.measured as Record<string, number>;
  const facts = [
    { label: "top-5", value: measured.seq_top_5?.toFixed(3) },
    { label: "top-1", value: measured.seq_top_1?.toFixed(3) },
    { label: "MRR", value: measured.seq_mrr?.toFixed(4) },
    { label: "classes", value: String(data.label_classes) },
    { label: "sequences", value: data.sequences.toLocaleString() },
  ];

  return (
    <div className="border-b border-hairline bg-surface">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-5 gap-y-1 px-6 py-2 text-[12px]">
        <span className="inline-flex items-center gap-1.5 font-medium text-ink-secondary">
          <FlaskConical aria-hidden className="size-3.5 text-accent" />
          5-fold cross-validated
        </span>
        {facts.map(
          (fact) =>
            fact.value && (
              <span key={fact.label} className="text-ink-secondary">
                {fact.label}{" "}
                <span className="font-mono font-medium text-ink">{fact.value}</span>
              </span>
            ),
        )}
        <span className="ml-auto inline-flex items-center gap-1.5 text-ink-secondary">
          <CircleCheck aria-hidden className="size-3.5 text-good" />
          test split never used
        </span>
      </div>
    </div>
  );
}
