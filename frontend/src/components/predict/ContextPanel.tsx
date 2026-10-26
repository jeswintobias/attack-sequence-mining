import { Info, TriangleAlert } from "lucide-react";
import type { ContextInfo } from "../../lib/types";
import { formatCount } from "../../lib/utils";
import { ConfidenceMeter } from "../ui/ConfidenceMeter";
import { TechniqueBadge } from "../ui/Badge";

/**
 * How much the prediction actually rests on.
 *
 * This panel exists because of a measured property of the model: temperature
 * scaling brought expected calibration error from 0.0808 to 0.0167 on average,
 * but it does not repair individual sparse contexts. A confident-looking
 * percentage over a context seen twice is the main way this tool could mislead,
 * so the support counts sit beside the forecast rather than behind a link.
 */

const BACKOFF_COPY: Record<ContextInfo["backoff"], string> = {
  second_order: "Using the last two techniques as context.",
  first_order: "Only the last technique has usable support; the two-technique context is too thin.",
  prior: "This context was not seen in training; the ranking falls back to overall technique frequency.",
};

export function ContextPanel({ context }: { context: ContextInfo }) {
  const fellBack = context.source === "first_order_fallback";

  return (
    <div className="space-y-3 text-[12px]">
      <div className="flex items-center justify-between gap-3">
        <span className="text-ink-secondary">Evidence behind this forecast</span>
        <ConfidenceMeter level={context.confidence} />
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
        <div>
          <dt className="text-ink-muted">Two-technique context</dt>
          <dd className="font-mono text-[13px] text-ink">
            {formatCount(context.second_order_support)}{" "}
            <span className="font-sans text-[11px] text-ink-muted">observations</span>
          </dd>
        </div>
        <div>
          <dt className="text-ink-muted">Last-technique context</dt>
          <dd className="font-mono text-[13px] text-ink">
            {formatCount(context.first_order_support)}{" "}
            <span className="font-sans text-[11px] text-ink-muted">observations</span>
          </dd>
        </div>
      </dl>

      <p className="flex gap-2 text-ink-secondary">
        <Info aria-hidden className="mt-0.5 size-3.5 shrink-0 text-ink-muted" />
        {BACKOFF_COPY[context.backoff]}
      </p>

      {fellBack && (
        <div className="flex gap-2 rounded-lg border border-warning/40 px-3 py-2">
          <TriangleAlert aria-hidden className="mt-0.5 size-3.5 shrink-0 text-warning" />
          <p className="text-ink-secondary">
            <strong className="font-medium text-ink">Single technique observed.</strong>{" "}
            Every training sequence has at least two, so the learned model is out of
            its range here. These figures come from the transition model instead and
            match the observed rates exactly. Add a second technique for a model
            forecast.
          </p>
        </div>
      )}

      {context.excluded_already_observed.length > 0 && (
        <div>
          <p className="mb-1 text-ink-muted">
            Excluded, already observed ({context.excluded_already_observed.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {context.excluded_already_observed.slice(0, 8).map((technique) => (
              <TechniqueBadge key={technique} id={technique} className="opacity-60" />
            ))}
            {context.excluded_already_observed.length > 8 && (
              <span className="self-center text-ink-muted">
                +{context.excluded_already_observed.length - 8} more
              </span>
            )}
          </div>
          <p className="mt-1 text-[11px] leading-snug text-ink-muted">
            The training data records each technique once per incident, so a repeat
            cannot be the next label. That is a property of the dataset, not of real
            attacker behaviour.
          </p>
        </div>
      )}

      {context.unknown_techniques.length > 0 && (
        <div className="flex gap-2 rounded-lg border border-warning/40 px-3 py-2">
          <TriangleAlert aria-hidden className="mt-0.5 size-3.5 shrink-0 text-warning" />
          <p className="text-ink-secondary">
            Not in the training vocabulary and therefore ignored:{" "}
            {context.unknown_techniques.join(", ")}.
          </p>
        </div>
      )}
    </div>
  );
}
