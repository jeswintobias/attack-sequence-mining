import type { Confidence } from "../../lib/types";
import { cn } from "../../lib/utils";

/**
 * Three ordinal steps of the single sequential blue, never a traffic light.
 *
 * High confidence is not "good" and low confidence is not "danger", so a
 * red/amber/green scale would miscommunicate. The text label is always rendered,
 * so the colour is reinforcement and never the only channel.
 */
const LEVELS: Record<Confidence, { filled: number; swatch: string; label: string }> = {
  low: { filled: 1, swatch: "bg-confidence-low", label: "Low support" },
  medium: { filled: 2, swatch: "bg-confidence-medium", label: "Medium support" },
  high: { filled: 3, swatch: "bg-confidence-high", label: "High support" },
};

export function ConfidenceMeter({
  level,
  className,
}: {
  level: Confidence;
  className?: string;
}) {
  const { filled, swatch, label } = LEVELS[level];
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span className="flex gap-0.5" aria-hidden>
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            /* 2px gaps between segments keep adjacent fills separable. */
            className={cn(
              "h-3 w-1.5 rounded-sm",
              index < filled ? swatch : "bg-hairline",
            )}
          />
        ))}
      </span>
      <span className="text-[12px] font-medium text-ink-secondary">{label}</span>
    </span>
  );
}
