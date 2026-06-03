import { cn } from "@/lib/utils";

export function Card({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-xl border border-neutral-200 bg-white p-5 dark:border-neutral-800 dark:bg-neutral-950",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="mb-3 text-sm font-medium text-neutral-500">{children}</h2>;
}

export function Stat({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <Card>
      <p className="text-sm text-neutral-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight">{value}</p>
      {hint ? <p className="mt-1 text-xs text-neutral-400">{hint}</p> : null}
    </Card>
  );
}
