import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Plus, Search, X } from "lucide-react";
import type { CatalogueEntry } from "../../lib/types";
import { cn } from "../../lib/utils";
import { TechniqueBadge } from "../ui/Badge";

/**
 * Searchable technique picker.
 *
 * Search matches the **name** as well as the identifier, which is the single
 * largest usability gain in the application: nobody recalls that T1086 is
 * PowerShell. Implements the ARIA combobox pattern with a listbox popup --
 * arrow keys move the active option, Enter commits it, Escape closes -- so the
 * control is usable without a mouse.
 *
 * Order matters here: the model consumes the prefix as a sequence, so the chips
 * render in observation order and can be removed individually rather than being
 * an unordered set.
 */
export function TechniquePicker({
  catalogue,
  observed,
  onChange,
  max = 32,
}: {
  catalogue: CatalogueEntry[];
  observed: string[];
  onChange: (next: string[]) => void;
  max?: number;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const listboxId = useId();

  const byId = useMemo(
    () => new Map(catalogue.map((entry) => [entry.technique, entry])),
    [catalogue],
  );

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const pool = catalogue.filter((entry) => !observed.includes(entry.technique));
    if (!needle) {
      // No query: offer the techniques the training data actually supports.
      return [...pool].sort((a, b) => b.train_frequency - a.train_frequency).slice(0, 12);
    }
    return pool
      .filter(
        (entry) =>
          entry.technique.toLowerCase().includes(needle) ||
          (entry.name ?? "").toLowerCase().includes(needle) ||
          entry.tactics.some((tactic) => tactic.toLowerCase().includes(needle)),
      )
      .slice(0, 20);
  }, [catalogue, observed, query]);

  useEffect(() => setActive(0), [query, open]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!wrapRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  useEffect(() => {
    listRef.current?.querySelector('[data-active="true"]')?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const atLimit = observed.length >= max;

  function add(technique: string) {
    if (atLimit || observed.includes(technique)) return;
    onChange([...observed, technique]);
    setQuery("");
    inputRef.current?.focus();
  }

  function removeAt(index: number) {
    onChange(observed.filter((_, position) => position !== index));
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!open) return setOpen(true);
      const delta = event.key === "ArrowDown" ? 1 : -1;
      setActive((current) => (current + delta + matches.length) % Math.max(matches.length, 1));
      return;
    }
    if (event.key === "Enter") {
      const option = matches[active];
      if (open && option) {
        event.preventDefault();
        add(option.technique);
      }
      return;
    }
    if (event.key === "Escape") {
      setOpen(false);
      return;
    }
    // Backspace on an empty query removes the most recent chip.
    if (event.key === "Backspace" && !query && observed.length) {
      removeAt(observed.length - 1);
    }
  }

  return (
    <div ref={wrapRef} className="relative">
      <div
        className={cn(
          "flex flex-wrap items-center gap-1.5 rounded-lg border border-hairline bg-surface px-2.5 py-2",
          "focus-within:border-accent",
        )}
      >
        {observed.map((technique, index) => {
          const entry = byId.get(technique);
          return (
            <span
              key={`${technique}-${index}`}
              className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-plane py-1 pl-1.5 pr-1 text-[12px]"
            >
              <span className="grid size-4 place-items-center rounded-sm bg-accent/10 font-mono text-[10px] font-semibold text-accent">
                {index + 1}
              </span>
              <code className="font-mono font-medium text-ink">{technique}</code>
              {entry?.name && <span className="text-ink-secondary">{entry.name}</span>}
              <button
                type="button"
                onClick={() => removeAt(index)}
                aria-label={`Remove ${entry?.name ?? technique}`}
                className="ml-0.5 rounded p-0.5 text-ink-muted hover:bg-hairline hover:text-ink"
              >
                <X aria-hidden className="size-3" />
              </button>
            </span>
          );
        })}

        <div className="flex min-w-[12rem] flex-1 items-center gap-1.5">
          <Search aria-hidden className="size-3.5 shrink-0 text-ink-muted" />
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded={open}
            aria-controls={listboxId}
            aria-autocomplete="list"
            aria-activedescendant={open && matches[active] ? `${listboxId}-${active}` : undefined}
            aria-label="Search techniques by name or identifier"
            placeholder={
              atLimit ? `Limit of ${max} reached` : observed.length ? "Add another…" : "Search by name or ID, e.g. PowerShell or T1059"
            }
            disabled={atLimit}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={onKeyDown}
            className="min-w-0 flex-1 bg-transparent py-1 text-[13px] text-ink outline-none placeholder:text-ink-muted"
          />
        </div>
      </div>

      {open && matches.length > 0 && (
        <ul
          ref={listRef}
          id={listboxId}
          role="listbox"
          aria-label="Matching techniques"
          className="absolute z-30 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-hairline bg-raised py-1 shadow-lg"
        >
          {!query && (
            <li className="px-3 py-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-muted">
              Most frequent in training
            </li>
          )}
          {matches.map((entry, index) => (
            <li
              key={entry.technique}
              id={`${listboxId}-${index}`}
              role="option"
              aria-selected={index === active}
              data-active={index === active}
              onMouseEnter={() => setActive(index)}
              onClick={() => add(entry.technique)}
              className={cn(
                "flex cursor-pointer items-center gap-2.5 px-3 py-1.5 text-[13px]",
                index === active ? "bg-accent/10" : "",
              )}
            >
              <TechniqueBadge id={entry.technique} />
              <span className="min-w-0 flex-1 truncate text-ink">
                {entry.name ?? <span className="text-ink-muted">Name not in ATT&amp;CK</span>}
              </span>
              {entry.tactics[0] && (
                <span className="hidden shrink-0 text-[11px] text-ink-muted sm:inline">
                  {entry.tactics[0]}
                </span>
              )}
              <Plus aria-hidden className="size-3.5 shrink-0 text-ink-muted" />
            </li>
          ))}
        </ul>
      )}

      {open && query && matches.length === 0 && (
        <div className="absolute z-30 mt-1 w-full rounded-lg border border-hairline bg-raised px-3 py-3 text-[13px] text-ink-secondary shadow-lg">
          No technique matches “{query}”. The model knows {catalogue.length} techniques.
        </div>
      )}
    </div>
  );
}
