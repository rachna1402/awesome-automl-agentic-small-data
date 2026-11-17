# RSI2026510 — Corrected small-data ML experiment

## What was actually run

A **controlled comparative study** (the course's approved Pathway B) extending the review *Benchmarking AutoML and Agentic Pipelines Against Expert-Tuned Models on Small Datasets*. It evaluates fixed Logistic Regression, fixed Random Forest, nested-CV-tuned RBF-SVM, and two Optuna sampler strategies. **It is not a full FLAML/auto-sklearn/AutoML-Agent reproduction or an LLM-agent experiment**, and the tuned SVM is a protocol-defined reference, not independently assessed human expert performance.

The study also records a four-trial **startup prefix of the TPE study**, as a search-budget sensitivity condition. TPE's first four trials are random initialization; the remaining trials use TPE. The random and TPE samplers use the same family/parameter search space, the same inner CV fold indices, and the same 12 evaluation-trial budget. Because configurations incur different fit times, a matched number of trials is **not** equal wall-clock compute.

Datasets: Breast Cancer, Wine, Iris (**primary tabular-only subset**) and Digits600 (**supplementary flattened-image feature-vector task; not treated as conventional tabular data**). All come from scikit-learn. Digits600 is a fixed, stratified 600-row subset, not the entire 1,797-row digits dataset.

Evaluation: seeds **42, 123**; 3 stratified outer folds per seed; 3 shared stratified inner folds for all selected/tuned methods. Each scored observation is held out from the fitting and search path. Classifier-based preprocessing is fit inside sklearn `Pipeline`. Primary measure: balanced accuracy. Secondary: macro-F1 and measured wall-clock fit+search time. All outputs originate from the executed `src/benchmark.py` script.

## Reproduce locally

Create a virtual environment with Python 3.13 if possible, and install `requirements.txt`. Original versions are in `results/environment.json` and `requirements_exact.txt`.

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/benchmark.py --seeds 42 123 --trials 12
python src/verify_outputs.py
python src/plots.py
```

On Windows PowerShell, set environment variables with `$env:OPENBLAS_NUM_THREADS='1'` etc., or omit the thread controls while noting that time and floating-point behavior may vary. Use the provided Colab notebook after uploading this ZIP. Colab execution has **not been independently validated here**.

**Warning:** The benchmark overwrites the result CSV files and environment file. Preserve a copy of the original results before rerunning, particularly if executing on another library or hardware version. Wall-clock times can change across machines; small score differences across versions may occur.

## Files

- `src/benchmark.py`: actual train/select/test pipeline, logs and CSV export
- `src/verify_outputs.py`: recomputes each metric from held-out predictions, checks fold alignment, full coverage, trial counts
- `src/plots.py`: regenerates PDF/PNG figures using only CSV evidence
- `results/fold_metrics.csv`: per outer fold, seed, dataset, method measured metrics
- `results/predictions.csv`: saved held-out labels and predictions
- `results/search_trial_logs.csv`: real inner-CV scores and sampler configurations
- `results/per_dataset_summary.csv`, `tabular_only_summary.csv`, `overall_summary.csv`: summaries
- `results/dataset_profiles.csv`, `environment.json`, `verification_report.json`: inputs/provenance and verification evidence
- `results/logs/final.log`: stdout from actual complete run
- `paper/manuscript.tex`, `.pdf`, `references.bib`: measured-results research paper
- `research_decisions.md`, `baseline_validation.md`, `ai_assistance_log.md`: transparent project records
- `notebooks/01_reproduce_colab.ipynb`: ZIP upload, rerun and checks

## Research-integrity requirements still outside this ZIP

The instructor requires institution-accepted plagiarism/similarity and AI-writing reports **for both the prior review and this implementation paper**. None is invented here. The student must obtain all four authentic reports, understand the code, review the manuscript, and conduct the required five-minute presentation. A GitHub publication must be done through the student's own account and comply with the instructor's submission process.

## Scope limits

This is a small course-scale benchmark, not proof of global optimality or publication-level novelty. Two seeds, four datasets (only three inherently tabular), modest 12-trial automated budgets, correlated repeated CV results and unequal total computational effort prevent strong statistical or broad superiority claims. No p-value is reported from treating folds as independent samples. Do not tune the methods based on their held-out scores.
