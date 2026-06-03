import type { Metadata } from "next";

import { Nav } from "@/components/layout/nav";
import { QueryProvider } from "@/providers/query-provider";
import "./globals.css";

export const metadata: Metadata = {
  title: "ITAI — AI Text Detection",
  description: "A platform for detecting AI-generated text: dataset analysis, benchmarking, and prediction.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <QueryProvider>
          <Nav />
          <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
        </QueryProvider>
      </body>
    </html>
  );
}
