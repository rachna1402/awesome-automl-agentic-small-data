# Benchmarking AutoML and Automated Model Search on Small Datasets

Research coursework (RSI2026510), including a literature review and a reproducible implementation study.

## Contents

- `01_Review_Paper/`: literature review PDF, LaTeX source, bibliography, and earlier Lab 1 material.
- `02_Implementation_Paper/`: controlled comparison of Logistic Regression, Random Forest, tuned SVM, random search, and Optuna TPE.
- `02_Implementation_Paper/notebooks/01_reproduce_colab.ipynb`: reproduction notebook.
- `02_Implementation_Paper/src/`: benchmark and verification scripts.
- `02_Implementation_Paper/results/`: saved measured results, predictions, logs, figures, and verification report.

## Reproduce

Read `02_Implementation_Paper/README.md` first. Install dependencies from its `requirements.txt`, then execute the supplied notebook or benchmark script. Results may vary across environments; do not treat the recorded numbers as universal optimum.

## Scope and limitations

The implementation is a **controlled comparative study**, not a reproduction of the published AutoML-Agent system or FLAML. Digits is a supplementary flattened-image-feature task; the tabular-only analyses are reported separately. Claims are limited to the actual recorded experiments.

## Privacy and provenance

Institutional plagiarism/AI-writing reports and the instructor's copyrighted assignment handout are intentionally excluded from this public-repository package. Do not upload private reports or credentials.
