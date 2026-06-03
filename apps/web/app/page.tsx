"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { Card, Stat } from "@/components/ui/card";
import { getDatasetAnalysis, listBenchmarks, getModelInfo } from "@/services/api";

export default function Dashboard() {
  const analysis = useQuery({ queryKey: ["analysis"], queryFn: getDatasetAnalysis, retry: false });
  const benchmarks = useQuery({ queryKey: ["benchmarks"], queryFn: listBenchmarks, retry: false });
  const model = useQuery({ queryKey: ["model"], queryFn: getModelInfo, retry: false });

  const samples = analysis.data?.profile.total_samples;
  const imbalance = analysis.data?.profile.class_distribution?.imbalance_ratio;

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">AI-generated text detection</h1>
        <p className="max-w-2xl text-neutral-500">
          A research-grade platform that ingests and profiles a text dataset, benchmarks classical
          embeddings and classifiers under identical conditions, and serves explainable predictions.
          This interface consumes the live API.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="Dataset samples"
          value={samples ?? "—"}
          hint={imbalance ? `class imbalance ${imbalance.toFixed(0)}:1` : undefined}
        />
        <Stat label="Benchmark tasks" value={benchmarks.data?.length ?? "—"} hint="detection + validation" />
        <Stat
          label="Loaded model"
          value={model.data ? model.data.classifier : "—"}
          hint={model.data ? model.data.embedding : "train a model to enable prediction"}
        />
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        <NavCard href="/dataset" title="Dataset analysis" body="Profile, class balance, length and vocabulary statistics, and validation findings." />
        <NavCard href="/benchmarks" title="Benchmarks" body="Embedding × classifier results with per-model metrics and confusion matrices." />
        <NavCard href="/predict" title="Prediction demo" body="Classify a passage and inspect the tokens driving the decision." />
      </section>
    </div>
  );
}

function NavCard({ href, title, body }: { href: string; title: string; body: string }) {
  return (
    <Link href={href}>
      <Card className="h-full transition-colors hover:border-neutral-400 dark:hover:border-neutral-600">
        <p className="font-medium">{title}</p>
        <p className="mt-1 text-sm text-neutral-500">{body}</p>
      </Card>
    </Link>
  );
}
