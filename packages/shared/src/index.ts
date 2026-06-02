// api contract types shared between the web app and the backend.
// these mirror the api responses so the frontend stays in sync with /api/v1.

export type Label = "human" | "ai";

export interface ClassProbability {
  label: Label;
  probability: number;
}

export interface Prediction {
  label: Label;
  confidence: number;
  probabilities: ClassProbability[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  sampleCount: number;
  classDistribution: Record<Label, number>;
  createdAt: string;
}

export interface HealthResponse {
  status: string;
  service: string;
}
