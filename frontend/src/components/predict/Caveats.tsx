import { useState } from "react";
import { ChevronDown, ShieldAlert } from "lucide-react";
import { cn } from "../../lib/utils";

/** The model card's caveats, collapsed but never more than one click away. */
export function Caveats({ caveats }: { caveats: string[] }) {
  const [open, setOpen] = useState(false);
  if (!caveats.length) return null;
  return (
    <div className="rounded-lg border border-hairline">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-[12px] font-medium text-ink"
      >
        <ShieldAlert aria-hidden className="size-3.5 text-ink-muted" />
        How to read this forecast
        <span className="ml-auto text-ink-muted">{caveats.length}</span>
        <ChevronDown
          aria-hidden
          className={cn("size-3.5 text-ink-muted transition-transform", open && "rotate-180")}
        />
      </button>
      {open && (
        <ul className="space-y-2 border-t border-hairline px-3 py-2.5 text-[12px] leading-snug text-ink-secondary">
          {caveats.map((caveat, index) => (
            <li key={index} className="flex gap-2">
              <span aria-hidden className="mt-1.5 size-1 shrink-0 rounded-full bg-ink-muted" />
              {caveat}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
