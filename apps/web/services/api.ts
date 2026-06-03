// typed api calls grouped by domain. all go through apiFetch so base url and error
// handling stay in one place (see api-client.ts).

import type { FeatureImportance, PredictionResponse } from "@itai/shared";

import { apiFetch } from "./api-client";
import type { BenchmarkSummary, DatasetAnalysis } from "@/types/reports";

export interface ModelInfo {
  embedding: string;
  classifier: string;
  preprocessing: Record<string, boolean>;
  labelNames: Record<string, string>;
  trainedAt: string | null;
  nTrainSamples: number;
}

export function getDatasetAnalysis(): Promise<DatasetAnalysis> {
  return apiFetch<DatasetAnalysis>("/datasets/analysis");
}

export function listBenchmarks(): Promise<string[]> {
  return apiFetch<string[]>("/benchmarks");
}

export function getBenchmark(task: string): Promise<BenchmarkSummary> {
  return apiFetch<BenchmarkSummary>(`/benchmarks/${task}`);
}

export function getModelInfo(): Promise<ModelInfo> {
  return apiFetch<ModelInfo>("/predict/model");
}

export function predict(text: string): Promise<PredictionResponse> {
  return apiFetch<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export function getExplanation(topN = 15): Promise<FeatureImportance> {
  return apiFetch<FeatureImportance>(`/predict/explanation?topN=${topN}`);
}
