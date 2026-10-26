/** Join class names, dropping falsy entries. */
export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/** Percent with one decimal below 10%, none above, so small values stay legible. */
export function formatPercent(value: number): string {
  const percent = value * 100;
  return `${percent < 10 ? percent.toFixed(1) : Math.round(percent)}%`;
}

export function formatCount(value: number): string {
  return Math.round(value).toLocaleString();
}
