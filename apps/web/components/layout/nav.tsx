"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/dataset", label: "Dataset" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/predict", label: "Prediction" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <nav className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-4">
        <Link href="/" className="font-semibold tracking-tight">
          ITAI
        </Link>
        <div className="flex gap-1 text-sm">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "rounded-md px-3 py-1.5 transition-colors",
                  active
                    ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                    : "text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
