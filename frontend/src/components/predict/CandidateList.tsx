import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Candidate } from "../../lib/types";
import { cn, formatCount, formatPercent } from "../../lib/utils";
import { TacticBadge, TechniqueBadge } from "../ui/Badge";

/**
 * Ranked candidates.
 *
 * The layout follows the calibration finding from the model service layer:
 * temperature scaling fixes calibration on average but not for individual
 * sparse contexts, so the model's number is never shown alone. Each row puts
 * the model estimate directly beside the **observed** transition rate, letting
 * a reader see when the two disagree. Rank is the most prominent element,
 * because rank is what the evaluation actually measured.
 */

function ProbabilityBar({ value, max }: { value: number; max: number }) {
  // Scaled against the top candidate so the ranking stays readable when the
  // whole distribution is flat. Marks are thin with rounded data-ends.
  const width = max > 0 ? Math.max((value / max) * 100, 1.5) : 0;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-hairline" aria-hidden>
      <div
        className="h-full rounded-full bg-accent transition-[width] duration-300"
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function Evidence({ candidate }: { candidate: Candidate }) {
  const order2 = candidate.transitions.find((t) => t.order === 2);
  const order1 = candidate.transitions.find((t) => t.order === 1);
  const rules = candidate.rules.slice(0, 3);

  return (
    <div className="space-y-3 border-t border-hairline bg-plane px-4 py-3 text-[12px]">
      <div>
        <p className="mb-1.5 font-medium text-ink">Observed transitions</p>
        {order1 || order2 ? (
          <ul className="space-y-1 text-ink-secondary">
            {[order2, order1].filter(Boolean).map((evidence) => (
              <li key={evidence!.order} className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono text-ink">
                  {evidence!.context.join(" → ")} → {candidate.technique}
                </span>
                <span>
                  seen <strong className="font-medium text-ink">{formatCount(evidence!.count)}</strong>{" "}
                  of {formatCount(evidence!.context_total)} times
                </span>
                <span className="font-mono text-ink">({formatPercent(evidence!.probability)})</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-ink-secondary">
            This context was never observed in training, so there is no transition
            evidence for this candidate.
          </p>
        )}
      </div>

      {rules.length > 0 && (
        <div>
          <p className="mb-1.5 font-medium text-ink">
            Mined sequential rules that fired
          </p>
          <ul className="space-y-1 text-ink-secondary">
            {rules.map((rule, index) => (
              <li key={index} className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono text-ink">
                  {rule.antecedent.join(" → ")} ⇒ {rule.consequent}
                </span>
                <span>
                  confidence{" "}
                  <strong className="font-medium text-ink">{rule.confidence.toFixed(2)}</strong>,
                  support {formatCount(rule.support)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function CandidateList({ candidates }: { candidates: Candidate[] }) {
  const [expanded, setExpanded] = useState<string | null>(
    candidates[0]?.technique ?? null,
  );
  const max = candidates[0]?.probability ?? 0;
  const coverage = candidates.reduce((sum, c) => sum + c.probability, 0);

  return (
    <div>
      <ul className="divide-y divide-hairline">
        {candidates.map((candidate) => {
          const order1 = candidate.transitions.find((t) => t.order === 1);
          const isOpen = expanded === candidate.technique;
          return (
            <li key={candidate.technique}>
              <div className="flex items-start gap-3 px-4 py-3">
                <span
                  className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-md bg-accent/10 font-mono text-[12px] font-semibold text-accent"
                  aria-label={`Rank ${candidate.rank}`}
                >
                  {candidate.rank}
                </span>

                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <TechniqueBadge id={candidate.technique} />
                    <span className="text-[14px] font-medium text-ink">
                      {candidate.name ?? (
                        <span className="text-ink-muted">Name not in ATT&amp;CK</span>
                      )}
                    </span>
                    {candidate.tactics.map((tactic) => (
                      <TacticBadge key={tactic} tactic={tactic} />
                    ))}
                  </div>

                  <div className="mt-2 flex items-center gap-3">
                    <div className="min-w-0 flex-1">
                      <ProbabilityBar value={candidate.probability} max={max} />
                    </div>
                    {/* Model estimate and observed rate side by side, so a
                        disagreement between them is visible rather than hidden. */}
                    <span className="shrink-0 font-mono text-[12px] text-ink">
                      {formatPercent(candidate.probability)}
                    </span>
                    {order1 && (
                      <span className="hidden shrink-0 text-[12px] text-ink-muted sm:inline">
                        observed {formatPercent(order1.probability)}
                      </span>
                    )}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setExpanded(isOpen ? null : candidate.technique)}
                  aria-expanded={isOpen}
                  aria-label={`${isOpen ? "Hide" : "Show"} evidence for ${candidate.technique}`}
                  className="mt-0.5 shrink-0 rounded p-1 text-ink-muted hover:bg-plane hover:text-ink"
                >
                  <ChevronDown
                    aria-hidden
                    className={cn("size-4 transition-transform", isOpen && "rotate-180")}
                  />
                </button>
              </div>
              {isOpen && <Evidence candidate={candidate} />}
            </li>
          );
        })}
      </ul>

      <p className="border-t border-hairline px-4 py-2.5 text-[12px] text-ink-secondary">
        These {candidates.length} candidates hold{" "}
        <strong className="font-medium text-ink">{formatPercent(coverage)}</strong> of the
        model's probability. Measured top-5 accuracy is 0.756, so the shortlist is the
        result — not its first row.
      </p>
    </div>
  );
}
