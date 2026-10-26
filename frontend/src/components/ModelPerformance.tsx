import { CircleSlash, FlaskConical, Scale } from "lucide-react";
import type { ModelCard } from "../lib/types";
import { Card, CardBody, CardHeader } from "./ui/Card";

/**
 * The evaluation, shown in the product.
 *
 * Including the negative results is deliberate. Four documented approaches —
 * mined pattern features, tactic abstraction, graph embeddings, fixed-weight
 * ensembling — were measured and did not help, and a reader deserves to see
 * what was tried alongside what shipped. A results panel that lists only
 * successes invites the question of what was omitted.
 */
export function ModelPerformance({ card }: { card: ModelCard }) {
  const measured = card.evaluation.measured as Record<string, number>;
  const calibration = card.calibration as Record<string, number | string>;

  const headline = [
    { label: "top-1", value: measured.seq_top_1, note: "one guess" },
    { label: "top-3", value: measured.seq_top_3, note: "three" },
    { label: "top-5", value: measured.seq_top_5, note: "the shortlist" },
    { label: "MRR", value: measured.seq_mrr, note: "mean reciprocal rank" },
  ];

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader
          title="How well it performs"
          description={card.evaluation.protocol}
          aside={
            <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-secondary">
              <FlaskConical aria-hidden className="size-3.5 text-accent" />
              {measured.n_folds ?? 5} folds
            </span>
          }
        />
        <CardBody>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {headline.map((stat) => (
              <div key={stat.label}>
                <dt className="text-[12px] text-ink-secondary">{stat.label}</dt>
                <dd className="font-mono text-[24px] font-semibold leading-tight tracking-tight text-ink">
                  {stat.value?.toFixed(3)}
                </dd>
                <p className="text-[11px] text-ink-muted">{stat.note}</p>
              </div>
            ))}
          </dl>

          <p className="mt-4 border-t border-hairline pt-3 text-[12px] leading-snug text-ink-secondary">
            Top-1 is low because the task is genuinely multi-modal: a repeated
            two-technique context is followed by{" "}
            <strong className="font-medium text-ink">6.02 distinct techniques</strong> on
            average. That ceiling belongs to the data, not the model, which is why the
            shortlist rather than the first row is the result.
          </p>

          <dl className="mt-3 grid gap-x-6 gap-y-2 text-[12px] sm:grid-cols-2">
            <div className="flex justify-between gap-3 border-t border-hairline pt-2">
              <dt className="text-ink-secondary">Macro recall across classes</dt>
              <dd className="font-mono text-ink">{measured.seq_macro_recall?.toFixed(3)}</dd>
            </div>
            <div className="flex justify-between gap-3 border-t border-hairline pt-2">
              <dt className="text-ink-secondary">Spread across folds (MRR sd)</dt>
              <dd className="font-mono text-ink">
                {measured.seq_mrr_sd_across_folds?.toFixed(4)}
              </dd>
            </div>
            <div className="flex justify-between gap-3 border-t border-hairline pt-2">
              <dt className="text-ink-secondary">Training sequences</dt>
              <dd className="font-mono text-ink">{card.sequences.toLocaleString()}</dd>
            </div>
            <div className="flex justify-between gap-3 border-t border-hairline pt-2">
              <dt className="text-ink-secondary">Technique classes</dt>
              <dd className="font-mono text-ink">{card.label_classes}</dd>
            </div>
          </dl>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="Calibration"
          description="Whether a stated percentage means what it says."
          aside={<Scale aria-hidden className="size-4 text-ink-muted" />}
        />
        <CardBody>
          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <p className="text-[12px] text-ink-secondary">Error before correction</p>
              <p className="font-mono text-[18px] font-semibold text-ink">
                {Number(calibration.ece_before).toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-[12px] text-ink-secondary">After temperature scaling</p>
              <p className="font-mono text-[18px] font-semibold text-accent">
                {Number(calibration.ece_after).toFixed(4)}
              </p>
            </div>
            <div>
              <p className="text-[12px] text-ink-secondary">Stated vs observed</p>
              <p className="font-mono text-[18px] font-semibold text-ink">
                {Number(calibration.mean_confidence_after).toFixed(3)} /{" "}
                {Number(calibration.observed_accuracy).toFixed(3)}
              </p>
            </div>
          </div>
          <p className="mt-3 border-t border-hairline pt-3 text-[12px] leading-snug text-ink-secondary">
            The raw model was over-confident. A single temperature, fitted on
            out-of-fold predictions so it never saw the data the final model
            trained on, brings stated confidence into line with observed accuracy.
            It corrects the average and not individual thin contexts, which is why
            the forecast view reports training support beside every number.
          </p>
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="What was tried and did not work"
          description="Measured approaches that failed to improve the model."
          aside={<CircleSlash aria-hidden className="size-4 text-ink-muted" />}
        />
        <CardBody>
          <ul className="space-y-2">
            {card.known_negative_results.map((result, index) => (
              <li key={index} className="flex gap-2.5 text-[13px] leading-snug">
                <span
                  aria-hidden
                  className="mt-1.5 size-1.5 shrink-0 rounded-full bg-ink-muted"
                />
                <span className="text-ink-secondary">{result}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 border-t border-hairline pt-3 text-[12px] leading-snug text-ink-secondary">
            These are reported because an evaluation that lists only what worked
            invites the question of what was left out. Each was measured under the
            same cross-validated protocol as the shipped model.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
