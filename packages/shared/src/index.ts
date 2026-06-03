// api contract types shared between the web app and the backend.
// these mirror the api responses so the frontend stays in sync with /api/v1.

export interface HealthResponse {
  status: string;
  service: string;
}

// --- dataset intelligence (phase 2) ---
// these mirror the /api/v1/datasets responses, which ml-core produces.

export interface LengthStats {
  unit: "char" | "word";
  mean: number;
  median: number;
  minimum: number;
  maximum: number;
  stdev: number;
}

export interface VocabularyStats {
  size: number;
  totalTokens: number;
  typeTokenRatio: number;
}

export interface ClassDistribution {
  counts: Record<string, number>;
  proportions: Record<string, number>;
  nClasses: number;
  imbalanceRatio: number;
}

export interface DatasetStatistics {
  charLength: LengthStats;
  wordLength: LengthStats;
  vocabulary: VocabularyStats;
}

export interface DatasetProfile {
  totalSamples: number;
  nColumns: number;
  statistics: DatasetStatistics;
  classDistribution: ClassDistribution | null;
}

export type IssueSeverity = "error" | "warning";

export interface ValidationIssue {
  severity: IssueSeverity;
  code: string;
  message: string;
  count: number;
}

export interface ValidationReport {
  isValid: boolean;
  totalSamples: number;
  issues: ValidationIssue[];
}

// --- experiments + benchmarking (phase 3) ---
// these mirror the /api/v1/experiments responses, produced by ml-core.

export interface MetricSummary {
  name: string;
  mean: number;
  std: number;
  values: number[];
}

export interface ConfusionMatrixResult {
  labels: string[];
  matrix: number[][]; // rows = true class, columns = predicted class
  falsePositives: Record<string, number>;
  falseNegatives: Record<string, number>;
  support: Record<string, number>;
}

export interface BenchmarkMetadata {
  embedding: string;
  classifier: string;
  nFolds: number;
  datasetVersion: string | null;
  preprocessing: Record<string, unknown>;
  notes: string;
  runId: string;
  createdAt: string;
}

export interface BenchmarkResult {
  metadata: BenchmarkMetadata;
  metrics: Record<string, MetricSummary>;
  foldMetrics: Record<string, number>[];
  confusion: ConfusionMatrixResult;
  fitSeconds: number;
  predictSeconds: number;
}

// the experiment api returns benchmark results.
export type ExperimentResult = BenchmarkResult;

// the compare endpoint is a loose rollup; nested objects keep ml-core's snake_case keys.
export interface ExperimentSummary {
  best: Record<string, Record<string, unknown>>;
  fastestFit: Record<string, unknown> | null;
  fastestPredict: Record<string, unknown> | null;
  ranking: Record<string, unknown>[];
}

// --- predictions + explainability (phase 4) ---
// camelCase contracts mirroring POST /api/v1/predict and GET /api/v1/predict/explanation.

export interface PredictionResponse {
  label: string;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  model: string;
  metadata: Record<string, unknown>;
}

export interface FeatureWeight {
  token: string;
  weight: number;
}

export interface FeatureImportance {
  model: string;
  positiveClass: string;
  negativeClass: string;
  topPositive: FeatureWeight[];
  topNegative: FeatureWeight[];
}
