# Report assets

Curated tables, figures, and summaries for the final report, assembled by
`scripts/build_report_assets.py` from the generated `reports/` directory.

## Contents

- `dataset/` — dataset profile, class distribution, length and token-frequency figures
- `benchmarks/` — detection (primary) and prompt_id (system-validation) summaries, comparison
  charts, and confusion matrices
- `explainability/` — top weighted features of the trained linear detector

## Note on the detection benchmark

The detection benchmark is intentionally reported as a negative result: the dataset carries
only 3 AI-generated essays out of 1,378, so every classifier collapses to the majority class.
See `benchmarks/detection_summary.md` for the full discussion. The prompt_id benchmark is a
balanced, real-text task used only to validate that the benchmarking pipeline produces
meaningful, differentiated results.
