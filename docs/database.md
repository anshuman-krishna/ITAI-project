# Database: current state and migration path

## Decision

The MVP does **not** use a database. Benchmark results, the dataset profile, the trained
model, and the report artifacts are all file-backed:

- benchmark runs → a JSON registry (`ml_core.benchmarking.BenchmarkRegistry` over
  `JsonBenchmarkStore`)
- dataset analysis → JSON + figures under `reports/dataset/`
- trained model → a directory of artifacts under `models/detector/`

This is a deliberate choice, not an omission. The registry is small, write-once, human
readable, diffable, and trivially portable, and it already sits behind an interface
(`BenchmarkRegistry`) that callers depend on instead of the storage. For a single-machine
research MVP, a database would add operational weight without buying anything the project
needs today.

The FastAPI app keeps a thin, unused database scaffold (`app/db`, `app/repositories`,
`app/models`) and `docker-compose` provisions Postgres, so the wiring is ready when growth
justifies it — but no tables are defined yet.

## When to introduce a database

Add Postgres when one of these becomes real:

- results must be **queried** (filter by dataset, embedding, date) rather than listed
- multiple users or processes write concurrently
- experiments, datasets, and predictions need to be **related** and audited
- the API must serve history that does not fit a single JSON file

## Migration path (low-risk, incremental)

1. **Define models + Alembic.** Add SQLAlchemy models for `experiments`, `datasets`,
   `predictions` (the entities already named in the project notes) and an Alembic baseline
   migration. The async session and declarative base already exist in `app/db`.
2. **Swap the store, keep the surface.** Implement a `SqlBenchmarkStore` with the same
   methods as `JsonBenchmarkStore` (`load` / `write` / `append`). Because the runner, the
   API, and the reporting layer all depend on `BenchmarkRegistry`, only the store
   construction changes — no caller is touched.
3. **Move artifacts to object storage.** Large files (datasets, model artifacts) move to a
   local volume now and S3/MinIO later; the database tracks them by metadata and location,
   never stores the bytes.
4. **Backfill.** A one-off script reads the existing JSON registry and writes it into the
   new store, so no history is lost.

The point of the registry abstraction is exactly this: the storage backend can change
without rippling through the platform.
