/**
 * Mirrors the Pydantic contract in src/serve/contract.py.
 *
 * Kept hand-written rather than generated so the comments explaining *why* a
 * field exists travel with it. If the Python contract changes, change this too;
 * the API tests guard the shapes on the server side.
 */

export type Backoff = "second_order" | "first_order" | "prior";
export type PredictionSource = "learned_model" | "first_order_fallback";
export type Confidence = "high" | "medium" | "low";

export interface RuleEvidence {
  antecedent: string[];
  consequent: string;
  confidence: number;
  support: number;
}

export interface TransitionEvidence {
  context: string[];
  order: 1 | 2;
  count: number;
  context_total: number;
  probability: number;
}

export interface Candidate {
  rank: number;
  technique: string;
  /** null when ATT&CK has no name on record; render the identifier instead. */
  name: string | null;
  probability: number;
  tactics: string[];
  transitions: TransitionEvidence[];
  rules: RuleEvidence[];
}

export interface ContextInfo {
  observed_count: number;
  /** Declares the out-of-distribution fallback for single-technique input. */
  source: PredictionSource;
  /** Derived from training support, NOT from the probability value. */
  confidence: Confidence;
  backoff: Backoff;
  second_order_support: number;
  first_order_support: number;
  excluded_already_observed: string[];
  unknown_techniques: string[];
}

export interface ModelInfo {
  name: string;
  trained_on: string;
  sequences: number;
  examples: number;
  label_classes: number;
  metrics: Record<string, number>;
}

export interface PredictResponse {
  candidates: Candidate[];
  context: ContextInfo;
  model: ModelInfo;
  caveats: string[];
}

export interface TacticPosition {
  tactic: string;
  index: number;
  observed: boolean;
  predicted_weight: number;
}

export interface StageResponse {
  tactics: TacticPosition[];
  furthest_observed: string | null;
  unmapped_techniques: string[];
  caveats: string[];
}

export interface OrderingCandidate {
  sequence: string[];
  log_likelihood: number;
  relative_likelihood: number;
}

export interface ReconstructResponse {
  orderings: OrderingCandidate[];
  considered: number;
  /** True when the beam covered every permutation, so the ranking is complete. */
  exhaustive: boolean;
  caveats: string[];
}

export interface CatalogueEntry {
  technique: string;
  name: string | null;
  tactics: string[];
  predictable: boolean;
  train_frequency: number;
}

export type ChartKind = "table" | "indexed_table" | "matrix";

export interface EdaChart {
  id: string;
  title: string;
  kind: ChartKind;
  rows: number;
  has_image: boolean;
}

export interface EdaChartData extends Omit<EdaChart, "rows" | "has_image"> {
  columns: string[];
  records: Record<string, string>[];
  /** Row labels; present for indexed_table and matrix only. */
  index: string[] | null;
}

export interface Health {
  status: string;
  application: string;
  model_loaded: boolean;
  label_classes: number;
  calibration_temperature: number;
}

/** The model card. Loosely typed because it is a record, read as a whole. */
export interface ModelCard {
  application: string;
  tagline: string;
  model: string;
  built_at: string;
  trained_on: string;
  sequences: number;
  examples: number;
  label_classes: number;
  feature_count: number;
  evaluation: {
    protocol: string;
    expected_seq_mrr: number;
    measured: Record<string, number | number[] | boolean>;
  };
  calibration: Record<string, number | string>;
  known_negative_results: string[];
  caveats: string[];
  test_data_loaded: boolean;
}
