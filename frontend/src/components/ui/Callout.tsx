import type { ReactNode } from "react";
import { AlertTriangle, CircleAlert, Info } from "lucide-react";
import { cn } from "../../lib/utils";

type Tone = "info" | "warning" | "critical";

/**
 * Status is never carried by colour alone: every tone ships an icon and the
 * caller supplies a text title. The reference palette's warning and serious
 * steps sit below 3:1 on the light surface, which makes the pairing mandatory
 * rather than stylistic.
 */
const TONES: Record<Tone, { icon: typeof Info; ring: string; fg: string }> = {
  info: { icon: Info, ring: "border-hairline", fg: "text-accent" },
  warning: { icon: AlertTriangle, ring: "border-warning/40", fg: "text-warning" },
  critical: { icon: CircleAlert, ring: "border-critical/40", fg: "text-critical" },
};

export function Callout({
  tone = "info",
  title,
  children,
  className,
}: {
  tone?: Tone;
  title: string;
  children?: ReactNode;
  className?: string;
}) {
  const { icon: Icon, ring, fg } = TONES[tone];
  return (
    <div
      className={cn("flex gap-3 rounded-lg border bg-surface px-4 py-3", ring, className)}
      role={tone === "critical" ? "alert" : undefined}
    >
      <Icon aria-hidden className={cn("mt-0.5 size-4 shrink-0", fg)} />
      <div className="min-w-0 text-[13px] leading-snug">
        <p className="font-semibold text-ink">{title}</p>
        {children && <div className="mt-1 text-ink-secondary">{children}</div>}
      </div>
    </div>
  );
}
