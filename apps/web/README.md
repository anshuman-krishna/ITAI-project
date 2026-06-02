# web

Next.js frontend (App Router, TypeScript, Tailwind v4). Data fetching goes through
React Query hooks that wrap the typed client in `services/api-client.ts`.

```
app/          routes, layouts, and pages
components/    ui, grouped by responsibility (see components/README.md)
hooks/         react query hooks over services
services/      api client and external calls
lib/           framework-agnostic helpers (e.g. cn)
providers/     react context providers (query client, ...)
types/         types, re-exporting the @itai/shared api contracts
```

## Run locally

```bash
cp .env.local.example .env.local
pnpm install            # from the repo root
pnpm --filter @itai/web dev
```

App runs at http://localhost:3000 and talks to the API at `NEXT_PUBLIC_API_URL`.
