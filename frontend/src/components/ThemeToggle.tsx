import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { cn } from "../lib/utils";

type Choice = "light" | "dark" | "system";
const KEY = "attackcast-theme";

const OPTIONS: { value: Choice; icon: typeof Sun; label: string }[] = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
];

function resolve(choice: Choice): "light" | "dark" {
  if (choice !== "system") return choice;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeToggle() {
  const [choice, setChoice] = useState<Choice>(() => {
    try {
      const stored = localStorage.getItem(KEY);
      return stored === "light" || stored === "dark" ? stored : "system";
    } catch {
      return "system";
    }
  });

  useEffect(() => {
    document.documentElement.dataset.theme = resolve(choice);
    try {
      if (choice === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, choice);
    } catch {
      /* private browsing: the in-memory choice still applies */
    }
    if (choice !== "system") return;
    // Follow the OS while "system" is selected.
    const query = matchMedia("(prefers-color-scheme: dark)");
    const sync = () => (document.documentElement.dataset.theme = resolve("system"));
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, [choice]);

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="flex items-center gap-0.5 rounded-lg border border-hairline bg-surface p-0.5"
    >
      {OPTIONS.map(({ value, icon: Icon, label }) => {
        const active = choice === value;
        return (
          <button
            key={value}
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={label}
            onClick={() => setChoice(value)}
            className={cn(
              "rounded-md p-1.5 transition-colors",
              active
                ? "bg-accent text-accent-ink"
                : "text-ink-muted hover:bg-plane hover:text-ink",
            )}
          >
            <Icon aria-hidden className="size-4" />
          </button>
        );
      })}
    </div>
  );
}
