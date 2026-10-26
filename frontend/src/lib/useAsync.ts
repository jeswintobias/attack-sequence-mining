import { useCallback, useEffect, useState } from "react";

export interface AsyncState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

/**
 * Minimal async hook. Deliberately not a data-fetching library: the app makes a
 * handful of calls and a dependency would outweigh the benefit.
 */
export function useAsync<T>(factory: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  // The factory identity is intentionally not a dependency; callers control
  // re-runs through `deps`, which keeps inline closures from looping.
  const run = useCallback(factory, deps);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    run()
      .then((value) => active && setData(value))
      .catch((cause) => active && setError(cause))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [run, nonce]);

  return { data, error, loading, reload: () => setNonce((n) => n + 1) };
}
