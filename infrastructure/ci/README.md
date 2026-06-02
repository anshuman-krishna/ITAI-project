# ci

The active pipeline lives in `.github/workflows/` (GitHub requires that path). It runs
linting, type checks, and build validation on every push.

This folder is for CI concerns that don't belong in a workflow file — reusable scripts,
deployment gates, or release tooling — as they're added.
