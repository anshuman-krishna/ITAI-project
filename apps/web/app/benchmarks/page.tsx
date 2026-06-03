"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { Async } from "@/components/ui/async";
import { Card, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { getBenchmark, listBenchmarks } from "@/services/api";
import type { BenchmarkModel, BenchmarkSummary } from "@/types/reports";

const METRICS: { key: string; label: string }[] = [
  { key: "accuracy", label: "accuracy" },
  { key: "f1_weighted", label: "F1 (weighted)" },
  { key: "f1_macro", label: "F1 (macro)" },
  { key: "precision_macro", label: "precision (macro)" },
  { key: "recall_macro", label: "recall (macro)" },
];

export default function BenchmarksPage() {
  const tasks = useQuery({ queryKey: ["benchmarks"], queryFn: listBenchmarks, retry: false });
  const [selected, setSelected] = useState<string | null>(null);
  const task = selected ?? tasks.data?.[0] ?? null;

  const summary = useQuery({
    queryKey: ["benchmark", task],
    queryFn: () => getBenchmark(task as string),
    enabled: !!task,
    retry: false,
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Benchmarks</h1>

      {tasks.data && tasks.data.length > 0 ? (
        <div className="flex gap-2">
          {tasks.data.map((t) => (
            <button
              key={t}
              onClick={() => setSelected(t)}
              className={cn(
                "rounded-md border px-3 py-1.5 text-sm transition-colors",
                t === task
                  ? "border-neutral-900 bg-neutral-900 text-white dark:border-white dark:bg-white dark:text-neutral-900"
                  : "border-neutral-200 text-neutral-600 hover:bg-neutral-100 dark:border-neutral-800 dark:hover:bg-neutral-800",
              )}
            >
              {t}
            </button>
          ))}
        </div>
      ) : null}

      {task ? (
        <Async query={summary}>{(data) => <BenchmarkView summary={data} />}</Async>
      ) : (
        <Async query={tasks}>{() => <p className="text-sm text-neutral-500">No benchmarks yet.</p>}</Async>
      )}
    </div>
  );
}

function BenchmarkView({ summary }: { summary: BenchmarkSummary }) {
  return (
    <div className="space-y-6">
      {summary.notes ? (
        <Card className="border-amber-300 bg-amber-50 dark:border-amber-900/60 dark:bg-amber-950/30">
          <CardTitle>Note</CardTitle>
          <p className="text-sm text-neutral-700 dark:text-neutral-300">{summary.notes}</p>
        </Card>
      ) : null}

      <Card>
        <CardTitle>
          Results — {summary.dataset} · {summary.n_models} models · ranked by {summary.ranking_metric}
        </CardTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neutral-200 text-left text-neutral-500 dark:border-neutral-800">
                <th className="py-2 pr-4 font-medium">embedding</th>
                <th className="py-2 pr-4 font-medium">classifier</th>
                {METRICS.map((m) => (
                  <th key={m.key} className="py-2 pr-4 font-medium">
                    {m.label}
                  </th>
                ))}
                <th className="py-2 pr-4 font-medium">fit ms</th>
              </tr>
            </thead>
            <tbody>
              {summary.models.map((model, i) => (
                <Row key={i} model={model} />
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function Row({ model }: { model: BenchmarkModel }) {
  return (
    <tr className="border-b border-neutral-100 last:border-0 dark:border-neutral-900">
      <td className="py-2 pr-4 font-mono">{model.metadata.embedding}</td>
      <td className="py-2 pr-4 font-mono">{model.metadata.classifier}</td>
      {METRICS.map((m) => (
        <td key={m.key} className="py-2 pr-4 font-mono">
          {model.metrics[m.key] ? model.metrics[m.key].mean.toFixed(3) : "—"}
        </td>
      ))}
      <td className="py-2 pr-4 font-mono text-neutral-500">{(model.fit_seconds * 1000).toFixed(1)}</td>
    </tr>
  );
}
