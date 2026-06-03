// shapes for the read-only report endpoints. these pass stored ml-core artifacts through
// untouched, so the keys stay snake_case here (unlike the camelCase action endpoints).

export interface LengthStats {
  unit: string;
  mean: number;
  median: number;
  minimum: number;
  maximum: number;
  stdev: number;
}

export interface VocabularyStats {
  size: number;
  total_tokens: number;
  type_token_ratio: number;
}

export interface ClassDistribution {
  counts: Record<string, number>;
  proportions: Record<string, number>;
  n_classes: number;
  imbalance_ratio: number;
}

export interface DatasetProfile {
  total_samples: number;
  n_columns: number;
  char_length: LengthStats;
  word_length: LengthStats;
  vocabulary: VocabularyStats;
  class_distribution: ClassDistribution | null;
}

export interface ValidationIssue {
  severity: "error" | "warning";
  code: string;
  message: string;
  count: number;
}

export interface ValidationReport {
  is_valid: boolean;
  total_samples: number;
  issues: ValidationIssue[];
}

export interface DatasetAnalysis {
  profile: DatasetProfile;
  validation: ValidationReport;
}

export interface MetricSummary {
  name: string;
  mean: number;
  std: number;
  values: number[];
}

export interface BenchmarkModel {
  metadata: {
    embedding: string;
    classifier: string;
    n_folds: number;
    dataset_version: string | null;
    notes: string;
  };
  metrics: Record<string, MetricSummary>;
  confusion: {
    labels: string[];
    matrix: number[][];
  };
  fit_seconds: number;
  predict_seconds: number;
}

export interface BenchmarkSummary {
  dataset: string | null;
  generated_at: string;
  n_models: number;
  ranking_metric: string;
  notes: string;
  best: Record<string, { label: string; value: number }>;
  by_classifier: Record<string, number>;
  by_embedding: Record<string, number>;
  models: BenchmarkModel[];
}
