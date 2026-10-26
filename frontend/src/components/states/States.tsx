import type { ReactNode } from "react";
import { Loader2, SearchX, ServerCrash } from "lucide-react";
import { ApiError } from "../../lib/api";
import { Callout } from "../ui/Callout";

export function LoadingState({ label = "Loading" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 px-5 py-10 text-[13px] text-ink-secondary"
    >
      <Loader2 aria-hidden className="size-4 animate-spin text-accent" />
      {label}…
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const isApi = error instanceof ApiError;
  const message = error instanceof Error ? error.message : String(error);
  const hint = isApi ? error.hint : undefined;
  return (
    <div className="px-5 py-6">
      <Callout tone="critical" title={isApi && error.status === 0 ? "API unreachable" : "Request failed"}>
        <p>{message}</p>
        {hint && (
          <p className="mt-2 font-mono text-[12px] text-ink-muted">{hint}</p>
        )}
      </Callout>
    </div>
  );
}

export function EmptyState({
  title,
  children,
  icon: Icon = SearchX,
}: {
  title: string;
  children?: ReactNode;
  icon?: typeof SearchX;
}) {
  return (
    <div className="flex flex-col items-center gap-3 px-5 py-14 text-center">
      <Icon aria-hidden className="size-7 text-ink-muted" />
      <div>
        <p className="text-[14px] font-medium text-ink">{title}</p>
        {children && (
          <p className="mx-auto mt-1 max-w-sm text-[13px] leading-snug text-ink-secondary">
            {children}
          </p>
        )}
      </div>
    </div>
  );
}

export { ServerCrash };
