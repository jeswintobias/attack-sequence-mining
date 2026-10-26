import type { StageResponse } from "../../lib/types";
import { cn, formatPercent } from "../../lib/utils";

/**
 * Fourteen ATT&CK tactics in kill-chain order.
 *
 * Two channels, deliberately separate: a filled node marks a tactic the prefix
 * has already touched, and a bar above it carries the share of predicted
 * probability mass landing there. Observation and prediction are different
 * claims and must not be conflated into one colour ramp.
 *
 * Labels are abbreviated to fit fourteen columns, with the full tactic name on
 * the element's title and in the accessible name, so nothing depends on the
 * reader decoding an abbreviation.
 */

const SHORT: Record<string, string> = {
  "Reconnaissance": "Recon",
  "Resource Development": "Resource Dev",
  "Initial Access": "Initial Access",
  "Execution": "Execution",
  "Persistence": "Persistence",
  "Privilege Escalation": "Priv Esc",
  "Defense Evasion": "Defense Evasion",
  "Credential Access": "Cred Access",
  "Discovery": "Discovery",
  "Lateral Movement": "Lateral Move",
  "Collection": "Collection",
  "Command and Control": "C2",
  "Exfiltration": "Exfil",
  "Impact": "Impact",
};

export function KillChainStepper({ stage }: { stage: StageResponse }) {
  const peak = Math.max(...stage.tactics.map((t) => t.predicted_weight), 0.0001);

  return (
    <div>
      <div
        className="flex gap-1 overflow-x-auto pb-1"
        role="img"
        aria-label={
          `Kill chain position. Observed: ` +
          `${stage.tactics.filter((t) => t.observed).map((t) => t.tactic).join(", ") || "none"}. ` +
          `Predicted next activity concentrates in ` +
          `${[...stage.tactics].sort((a, b) => b.predicted_weight - a.predicted_weight)[0]?.tactic ?? "no tactic"}.`
        }
      >
        {stage.tactics.map((position) => {
          const height = Math.round((position.predicted_weight / peak) * 100);
          const notable = position.predicted_weight >= 0.08;
          return (
            <div
              key={position.tactic}
              title={`${position.tactic} — ${position.observed ? "observed" : "not observed"}, ${formatPercent(position.predicted_weight)} of predicted mass`}
              className="flex min-w-[58px] flex-1 flex-col items-center gap-1"
            >
              {/* Predicted mass */}
              <div className="flex h-12 w-full items-end justify-center">
                <div
                  className={cn(
                    "w-full rounded-t-[3px] transition-[height] duration-300",
                    notable ? "bg-accent" : "bg-accent/35",
                  )}
                  style={{ height: `${Math.max(height, position.predicted_weight > 0 ? 3 : 0)}%` }}
                />
              </div>
              {/* Direct label, only where it carries information */}
              <span
                className={cn(
                  "h-3.5 font-mono text-[10px] leading-none",
                  notable ? "text-ink" : "text-transparent",
                )}
              >
                {notable ? formatPercent(position.predicted_weight) : "."}
              </span>
              {/* Observation rail */}
              <div className="flex w-full items-center">
                <span
                  className={cn(
                    "h-px flex-1",
                    position.index === 0 ? "bg-transparent" : "bg-hairline",
                  )}
                />
                <span
                  className={cn(
                    "size-2.5 shrink-0 rounded-full border-2",
                    position.observed
                      ? "border-accent bg-accent"
                      : "border-axis bg-surface",
                  )}
                />
                <span
                  className={cn(
                    "h-px flex-1",
                    position.index === 13 ? "bg-transparent" : "bg-hairline",
                  )}
                />
              </div>
              <span
                className={cn(
                  "text-center text-[10px] leading-tight",
                  position.observed ? "font-medium text-ink" : "text-ink-muted",
                )}
              >
                {SHORT[position.tactic] ?? position.tactic}
              </span>
            </div>
          );
        })}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-secondary">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-full border-2 border-accent bg-accent" aria-hidden />
          observed in the prefix
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2 rounded-sm bg-accent" aria-hidden />
          share of predicted mass
        </span>
        {stage.furthest_observed && (
          <span className="ml-auto">
            Furthest reached:{" "}
            <strong className="font-medium text-ink">{stage.furthest_observed}</strong>
          </span>
        )}
      </div>

      {stage.unmapped_techniques.length > 0 && (
        <p className="mt-2 text-[11px] text-ink-muted">
          No tactic on record for {stage.unmapped_techniques.join(", ")}; these do not
          appear on the chain.
        </p>
      )}
    </div>
  );
}
