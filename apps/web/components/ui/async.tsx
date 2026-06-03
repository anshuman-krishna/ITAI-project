import type { UseQueryResult } from "@tanstack/react-query";

import { ApiError } from "@/services/api-client";
import { Card } from "./card";

// shared loading/error rendering for react-query results, so pages stay focused on layout.
export function Async<T>({
  query,
  children,
}: {
  query: UseQueryResult<T>;
  children: (data: T) => React.ReactNode;
}) {
  if (query.isLoading) {
    return <p className="text-sm text-neutral-500">Loading…</p>;
  }
  if (query.isError) {
    const err = query.error;
    const hint =
      err instanceof ApiError && err.status === 503
        ? "The model or report has not been generated yet. Run the project scripts (see README)."
        : err instanceof ApiError && err.status === 404
          ? "This data has not been generated yet. Run the project scripts (see README)."
          : "Could not reach the API. Is the backend running on the configured URL?";
    return (
      <Card>
        <p className="text-sm font-medium">Unavailable</p>
        <p className="mt-1 text-sm text-neutral-500">{hint}</p>
      </Card>
    );
  }
  return <>{children(query.data as T)}</>;
}
