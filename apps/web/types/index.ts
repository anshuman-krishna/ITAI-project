// re-export shared api contract types so app code imports from one place.
// ui-only types that never cross the api boundary can also live here.
export type {
  Label,
  Prediction,
  ClassProbability,
  DatasetSummary,
  HealthResponse,
} from "@itai/shared";
