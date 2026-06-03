"""assemble docs/report-assets/ from the generated reports.

reports/ is gitignored (regenerated artifacts), so this curates a committed set of tables,
figures, and summaries to make final report writing easy. run the analysis, benchmark, and
training scripts first; then run this.

usage:
    python scripts/build_report_assets.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = ROOT / "docs" / "report-assets"

# (source relative to reports/, destination relative to docs/report-assets/)
ASSETS_MANIFEST = [
    ("dataset/summary.md", "dataset/summary.md"),
    ("dataset/dataset_profile.json", "dataset/dataset_profile.json"),
    ("dataset/class_distribution.png", "dataset/class_distribution.png"),
    ("dataset/word_length_distribution.png", "dataset/word_length_distribution.png"),
    ("dataset/char_length_distribution.png", "dataset/char_length_distribution.png"),
    ("dataset/top_tokens.png", "dataset/top_tokens.png"),
    ("benchmarks/detection/benchmark_summary.md", "benchmarks/detection_summary.md"),
    ("benchmarks/detection/benchmark_summary.csv", "benchmarks/detection_summary.csv"),
    ("benchmarks/detection/classifier_comparison.png", "benchmarks/detection_classifier_comparison.png"),
    ("benchmarks/detection/confusion_tfidf_logistic_regression.png", "benchmarks/detection_confusion.png"),
    ("benchmarks/prompt_id/benchmark_summary.md", "benchmarks/prompt_id_summary.md"),
    ("benchmarks/prompt_id/benchmark_summary.csv", "benchmarks/prompt_id_summary.csv"),
    ("benchmarks/prompt_id/classifier_comparison.png", "benchmarks/prompt_id_classifier_comparison.png"),
    ("benchmarks/prompt_id/confusion_tfidf_logistic_regression.png", "benchmarks/prompt_id_confusion.png"),
    ("explainability/feature_importance.md", "explainability/feature_importance.md"),
    ("explainability/feature_importance.png", "explainability/feature_importance.png"),
]

INDEX = """# Report assets

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
"""


def main() -> None:
    copied, missing = 0, []
    for src_rel, dst_rel in ASSETS_MANIFEST:
        src = REPORTS / src_rel
        dst = ASSETS / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "README.md").write_text(INDEX)

    print(f"report assets: copied {copied} files to {ASSETS}")
    if missing:
        print(f"missing {len(missing)} (run the analysis/benchmark/train scripts first):")
        for m in missing:
            print(f"  - reports/{m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
