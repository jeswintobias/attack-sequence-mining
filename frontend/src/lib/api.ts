/**
 * Typed client for the AttackCast API.
 *
 * Errors are normalised into ApiError so every view can render one failure
 * shape. FastAPI returns validation problems as 422 with a `detail` payload
 * that is either a string or a list of field errors; both are flattened into a
 * readable message rather than shown as raw JSON.
 */

import type {
  CatalogueEntry,
  EdaChart,
  EdaChartData,
  Health,
  ModelCard,
  PredictResponse,
  ReconstructResponse,
  StageResponse,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface FieldError {
  loc?: (string | number)[];
  msg?: string;
}

function flattenDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return (detail as FieldError[])
      .map((item) => {
        const field = (item.loc ?? []).filter((part) => part !== "body").join(".");
        return field ? `${field}: ${item.msg ?? "invalid"}` : (item.msg ?? "invalid");
      })
      .join("; ");
  }
  return null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      "Cannot reach the AttackCast API.",
      0,
      "Start it with: python -m uvicorn src.serve.app:app --reload",
    );
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      message = flattenDetail(body.detail) ?? message;
    } catch {
      /* keep the status line */
    }
    throw new ApiError(
      message,
      response.status,
      response.status === 503
        ? "Model artifacts are missing. Run: python src/serve/artifacts.py"
        : undefined,
    );
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),
  meta: () => request<ModelCard>("/meta"),
  catalogue: () => request<CatalogueEntry[]>("/catalogue"),

  predict: (observed: string[], topK = 5) =>
    post<PredictResponse>("/predict", { observed, top_k: topK }),

  stage: (observed: string[]) => post<StageResponse>("/stage", { observed }),

  reconstruct: (observed: string[], topN = 5, beamWidth = 50) =>
    post<ReconstructResponse>("/reconstruct", {
      observed,
      top_n: topN,
      beam_width: beamWidth,
    }),

  edaIndex: () => request<EdaChart[]>("/eda"),
  edaChart: (id: string) => request<EdaChartData>(`/eda/${encodeURIComponent(id)}`),
  edaImageUrl: (id: string) => `${BASE}/eda/${encodeURIComponent(id)}/image`,
};
