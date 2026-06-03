"use client";

import { useQuery } from "@tanstack/react-query";

import { Async } from "@/components/ui/async";
import { Card, CardTitle, Stat } from "@/components/ui/card";
import { getDatasetAnalysis } from "@/services/api";
import type { ClassDistribution, LengthStats, ValidationIssue } from "@/types/reports";

export default function DatasetPage() {
  const query = useQuery({ queryKey: ["analysis"], queryFn: getDatasetAnalysis, retry: false });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold tracking-tight">Dataset analysis</h1>
      <Async query={query}>
        {(data) => {
          const p = data.profile;
          return (
            <div className="space-y-6">
              <div className="grid gap-4 sm:grid-cols-3">
                <Stat label="Samples" value={p.total_samples.toLocaleString()} />
                <Stat label="Unique tokens" value={p.vocabulary.size.toLocaleString()} />
                <Stat
                  label="Type/token ratio"
                  value={p.vocabulary.type_token_ratio.toFixed(4)}
                  hint="lexical richness"
                />
              </div>

              {p.class_distribution ? <ClassBalance dist={p.class_distribution} /> : null}

              <div className="grid gap-4 sm:grid-cols-2">
                <LengthCard title="Characters per text" stats={p.char_length} />
                <LengthCard title="Words per text" stats={p.word_length} />
              </div>

              <Validation issues={data.validation.issues} isValid={data.validation.is_valid} />
            </div>
          );
        }}
      </Async>
    </div>
  );
}

function ClassBalance({ dist }: { dist: ClassDistribution }) {
  const entries = Object.entries(dist.counts).sort((a, b) => b[1] - a[1]);
  return (
    <Card>
      <CardTitle>Class distribution</CardTitle>
      <div className="space-y-2">
        {entries.map(([label, count]) => (
          <div key={label} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="font-mono">{label}</span>
              <span className="text-neutral-500">
                {count.toLocaleString()} ({(dist.proportions[label] * 100).toFixed(2)}%)
              </span>
            </div>
            <div className="h-2 rounded bg-neutral-100 dark:bg-neutral-800">
              <div
                className="h-2 rounded bg-neutral-800 dark:bg-neutral-200"
                style={{ width: `${dist.proportions[label] * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      {dist.imbalance_ratio > 3 ? (
        <p className="mt-3 text-sm text-amber-600 dark:text-amber-500">
          Imbalance ratio {dist.imbalance_ratio.toFixed(0)}:1 — heavily skewed. See the detection
          benchmark for why this makes AI-vs-human classification on this corpus degenerate.
        </p>
      ) : null}
    </Card>
  );
}

function LengthCard({ title, stats }: { title: string; stats: LengthStats }) {
  const rows: [string, number][] = [
    ["mean", stats.mean],
    ["median", stats.median],
    ["min", stats.minimum],
    ["max", stats.maximum],
    ["stdev", stats.stdev],
  ];
  return (
    <Card>
      <CardTitle>{title}</CardTitle>
      <dl className="grid grid-cols-2 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between pr-4">
            <dt className="text-neutral-500">{k}</dt>
            <dd className="font-mono">{Math.round(v).toLocaleString()}</dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}

function Validation({ issues, isValid }: { issues: ValidationIssue[]; isValid: boolean }) {
  return (
    <Card>
      <CardTitle>Validation</CardTitle>
      <p className="mb-2 text-sm">
        Status:{" "}
        <span className={isValid ? "text-emerald-600" : "text-red-600"}>
          {isValid ? "valid" : "invalid"}
        </span>
      </p>
      {issues.length === 0 ? (
        <p className="text-sm text-neutral-500">No issues found.</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {issues.map((issue) => (
            <li key={issue.code}>
              <span
                className={
                  issue.severity === "error"
                    ? "font-medium text-red-600"
                    : "font-medium text-amber-600"
                }
              >
                {issue.severity}
              </span>{" "}
              <span className="font-mono text-neutral-500">{issue.code}</span> — {issue.message} (
              {issue.count})
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
