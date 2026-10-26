import { useRef } from "react";
import { Radar } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { cn } from "../lib/utils";

export type TabId = "predict" | "explore";

export const TABS: { id: TabId; label: string; hint: string }[] = [
  { id: "predict", label: "Forecast", hint: "Rank what the intrusion may do next" },
  { id: "explore", label: "Explore", hint: "Exploratory analysis of the training data" },
];

/**
 * Tabs implement the ARIA tabs pattern by hand rather than pulling in a
 * component library: roving tabindex, arrow-key and Home/End navigation, and
 * aria-controls wired to the panel. Keyboard parity is assessable work, so it
 * is worth owning.
 */
export function Header({
  active,
  onChange,
}: {
  active: TabId;
  onChange: (id: TabId) => void;
}) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});

  function onKeyDown(event: React.KeyboardEvent) {
    const order = TABS.map((tab) => tab.id);
    const current = order.indexOf(active);
    let next = current;
    if (event.key === "ArrowRight") next = (current + 1) % order.length;
    else if (event.key === "ArrowLeft") next = (current - 1 + order.length) % order.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = order.length - 1;
    else return;
    event.preventDefault();
    const id = order[next];
    if (!id) return;
    onChange(id);
    refs.current[id]?.focus();
  }

  return (
    <header className="sticky top-0 z-20 border-b border-hairline bg-plane/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span
            aria-hidden
            className="grid size-8 place-items-center rounded-lg bg-accent text-accent-ink"
          >
            <Radar className="size-[18px]" />
          </span>
          <div className="min-w-0 leading-tight">
            <p className="text-[15px] font-semibold tracking-tight text-ink">AttackCast</p>
            <p className="hidden truncate text-[12px] text-ink-secondary sm:block">
              Where the intrusion goes next, ranked by evidence
            </p>
          </div>
        </div>

        <nav
          role="tablist"
          aria-label="Views"
          onKeyDown={onKeyDown}
          className="ml-auto flex items-center gap-1 rounded-lg border border-hairline bg-surface p-0.5"
        >
          {TABS.map((tab) => {
            const selected = tab.id === active;
            return (
              <button
                key={tab.id}
                ref={(node) => {
                  refs.current[tab.id] = node;
                }}
                role="tab"
                id={`tab-${tab.id}`}
                aria-selected={selected}
                aria-controls={`panel-${tab.id}`}
                tabIndex={selected ? 0 : -1}
                title={tab.hint}
                onClick={() => onChange(tab.id)}
                className={cn(
                  "rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors",
                  selected
                    ? "bg-accent text-accent-ink"
                    : "text-ink-secondary hover:bg-plane hover:text-ink",
                )}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>

        <ThemeToggle />
      </div>
    </header>
  );
}
