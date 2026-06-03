"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { FeatureImportance, PredictionResponse } from "@itai/shared";

import { Card, CardTitle } from "@/components/ui/card";
import { ApiError } from "@/services/api-client";
import { getExplanation, predict } from "@/services/api";

const SAMPLE =
  "The implications of widespread automation are multifaceted and warrant careful consideration. " +
  "Furthermore, a comprehensive framework is essential to optimize outcomes across diverse stakeholders.";

export default function PredictPage() {
  const [text, setText] = useState("");
  const mutation = useMutation({ mutationFn: predict });
  const explanation = useQuery({
    queryKey: ["explanation"],
    queryFn: () => getExplanation(12),
    retry: false,
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Prediction demo</h1>
      <p className="max-w-2xl text-sm text-neutral-500">
        Paste a passage to classify it as human- or AI-written. The model returns a label, a
        confidence score, and the full probability distribution.
      </p>

      <Card>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={7}
          placeholder="Paste text to classify…"
          className="w-full resize-y rounded-lg border border-neutral-200 bg-transparent p-3 text-sm outline-none focus:border-neutral-400 dark:border-neutral-800"
        />
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={() => mutation.mutate(text)}
            disabled={!text.trim() || mutation.isPending}
            className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-white dark:text-neutral-900"
          >
            {mutation.isPending ? "Classifying…" : "Classify"}
          </button>
          <button
            onClick={() => setText(SAMPLE)}
            className="text-sm text-neutral-500 hover:text-neutral-800 dark:hover:text-neutral-200"
          >
            Use sample
          </button>
        </div>
      </Card>

      {mutation.isError ? (
        <Card>
          <p className="text-sm text-red-600">
            {mutation.error instanceof ApiError && mutation.error.status === 503
              ? "No trained model is loaded. Run scripts/train_model.py, then restart the API."
              : "Prediction failed. Is the backend running?"}
          </p>
        </Card>
      ) : null}

      {mutation.data ? <Result result={mutation.data} /> : null}

      {explanation.data ? <Explanation importance={explanation.data} /> : null}
    </div>
  );
}

function Result({ result }: { result: PredictionResponse }) {
  const probs = result.probabilities ?? {};
  return (
    <Card>
      <CardTitle>Result</CardTitle>
      <div className="flex items-baseline gap-3">
        <span className="text-2xl font-semibold capitalize">{result.label}</span>
        {result.confidence !== null ? (
          <span className="text-neutral-500">{(result.confidence * 100).toFixed(1)}% confidence</span>
        ) : (
          <span className="text-sm text-neutral-400">(model does not expose probabilities)</span>
        )}
      </div>
      <div className="mt-4 space-y-2">
        {Object.entries(probs).map(([label, p]) => (
          <div key={label} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="capitalize">{label}</span>
              <span className="font-mono text-neutral-500">{(p * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 rounded bg-neutral-100 dark:bg-neutral-800">
              <div
                className="h-2 rounded bg-neutral-800 dark:bg-neutral-200"
                style={{ width: `${p * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs text-neutral-400">model: {result.model}</p>
    </Card>
  );
}

function Explanation({ importance }: { importance: FeatureImportance }) {
  return (
    <Card>
      <CardTitle>What drives the model</CardTitle>
      <p className="mb-3 text-sm text-neutral-500">
        Top tokens by linear-model weight. These are global model coefficients, not per-text
        attributions.
      </p>
      <div className="grid gap-6 sm:grid-cols-2">
        <TokenColumn title={`Pushes toward ${importance.positiveClass}`} weights={importance.topPositive} />
        <TokenColumn title={`Pushes toward ${importance.negativeClass}`} weights={importance.topNegative} />
      </div>
    </Card>
  );
}

function TokenColumn({ title, weights }: { title: string; weights: FeatureImportance["topPositive"] }) {
  return (
    <div>
      <p className="mb-2 text-sm font-medium capitalize">{title}</p>
      <ul className="space-y-1 text-sm">
        {weights.map((w) => (
          <li key={w.token} className="flex justify-between">
            <span className="font-mono">{w.token}</span>
            <span className="font-mono text-neutral-400">{w.weight.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
