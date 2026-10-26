import type { ReactNode } from "react";
import { cn } from "../../lib/utils";

/**
 * Technique identifiers are the only place monospace is used, so an ID is
 * recognisable as an ID without needing a label.
 */
export function TechniqueBadge({ id, className }: { id: string; className?: string }) {
  return (
    <code
      className={cn(
        "rounded border border-hairline bg-plane px-1.5 py-0.5 font-mono text-[12px] font-medium text-ink",
        className,
      )}
    >
      {id}
    </code>
  );
}

export function TacticBadge({ tactic }: { tactic: string }) {
  return (
    <span className="rounded-full border border-hairline px-2 py-0.5 text-[11px] font-medium text-ink-secondary">
      {tactic}
    </span>
  );
}

export function Pill({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
        className,
      )}
    >
      {children}
    </span>
  );
}
