import { ArrowRight, FlaskConical } from "lucide-react";
import type { ReconstructResponse } from "../../lib/types";
import { cn, formatPercent } from "../../lib/utils";

/**
 * Plausible orderings of what has already been seen.
 *
 * Labelled exploratory throughout. Unlike next-step prediction this capability
 * was never evaluated, so it carries no accuracy figure and the interface says
 * so rather than letting the ranking imply one.
 */
export function PathReconstruction({ result }: { result: ReconstructResponse }) {
  const best = result.orderings[0];
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 rounded-lg border border-hairline bg-plane px-3 py-2 text-[12px]">
        <FlaskConical aria-hidden className="mt-0.5 size-3.5 shrink-0 text-ink-muted" />
        <p className="text-ink-secondary">
          Exploratory. Orderings are scored with the transition model and ranked
          against each other only — this capability has no measured accuracy.
        </p>
      </div>

      <ol className="space-y-2">
        {result.orderings.map((ordering, index) => (
          <li
            key={ordering.sequence.join(">")}
            className={cn(
              "rounded-lg border px-3 py-2",
              index === 0 ? "border-accent/40 bg-accent/5" : "border-hairline",
            )}
          >
            <div className="flex flex-wrap items-center gap-1.5">
              {ordering.sequence.map((technique, position) => (
                <span key={position} className="inline-flex items-center gap-1.5">
                  {position > 0 && (
                    <ArrowRight aria-hidden className="size-3 text-ink-muted" />
                  )}
                  <code className="font-mono text-[12px] text-ink">{technique}</code>
                </span>
              ))}
            </div>
            <div className="mt-1.5 flex items-center gap-2">
              <div className="h-1 flex-1 overflow-hidden rounded-full bg-hairline" aria-hidden>
                <div
                  className="h-full rounded-full bg-accent"
                  style={{
                    width: `${Math.max(
                      (ordering.relative_likelihood / (best?.relative_likelihood ?? 1)) * 100,
                      2,
                    )}%`,
                  }}
                />
              </div>
              <span className="font-mono text-[11px] text-ink-secondary">
                {formatPercent(ordering.relative_likelihood)}
              </span>
            </div>
          </li>
        ))}
      </ol>

      <p className="text-[11px] text-ink-muted">
        Scored {result.considered.toLocaleString()} partial orderings.{" "}
        {result.exhaustive
          ? "The search covered every permutation, so this ranking is complete."
          : "The beam pruned some branches, so a better ordering may exist outside it."}
      </p>
    </div>
  );
}
