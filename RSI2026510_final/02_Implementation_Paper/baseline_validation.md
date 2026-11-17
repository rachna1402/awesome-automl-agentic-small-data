# Baseline and experimental-output validation

## Actually completed
- Source was executed in local Linux Python 3.13.5 using the package versions in `results/environment.json`.
- Four datasets loaded: Breast Cancer, Wine, Iris and stratified 600-row Digits subset; descriptive profiles are in `results/dataset_profiles.csv`.
- Two outer seeds (42 and 123), three stratified outer folds per seed, and three shared stratified inner folds were applied to all tuned/automated methods.
- Baseline pipeline: standardize inside sklearn Pipeline for logistic regression and RBF SVM; Random Forest operates without scaling. All fitting and selection use outer-training rows only.
- Tuned SVM: 4 values of C x 3 gamma values = 12 candidates, evaluated in the inner three-fold CV.
- Optuna random and Optuna TPE: same two model families and search bounds, 12 trials per fold. TPE startup is 4 random trials; after trial 4 model-based TPE suggestions may be used.
- Independent score audit (`src/verify_outputs.py`) **PASSED**. Recalculated every balanced-accuracy, macro-F1 and ordinary-accuracy value from `results/predictions.csv`. All methods share each held-out fold and original label, with each record covered exactly once per method and seed. Search trial logs contain exactly the expected 12 trials for each optimizer on each outer fold. Final numerical counts are documented in `results/verification_report.json`.
- Figures regenerate from CSV via `src/plots.py` and the LaTeX paper is generated from those summaries.

## Technical corrections compared with the earlier project
- Earlier search and tuned SVM did not necessarily use the same inner validation split. The new script passes the same precomputed splits to both.
- Earlier implementation used one seed and eight search trials. The corrected study uses two seeds and a complete 12-trial search, alongside an early four-trial sensitivity point.
- The initial three-task study mixed two tabular datasets with a digit-image-derived feature-vector task. The corrected study adds Iris and reports primary (three tabular datasets) separately from all four tasks.
- A tuned SVM remains a protocol-defined model comparator and is not rebranded as a human-participant expert baseline. Optuna remains an Optuna search, not FLAML or agentic ML.

## Unverified and limitations
- Independent Colab rerun on the student's account or a second machine has not been performed.
- No equal wall-clock budget: same 12 candidate-trial budget is established for the two automated search strategies, but model families differ in fit cost.
- Repeated CV folds and seeds are not independent for inferential testing; no significance test or claim of optimal performance is made.
- No published full AutoML library or LLM agent was evaluated; this is the instructor-approved controlled-comparison study pathway.
