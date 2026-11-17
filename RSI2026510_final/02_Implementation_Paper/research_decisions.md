# Research decision and evidence ledger

- **Review:** *Benchmarking AutoML and Agentic Pipelines Against Expert-Tuned Models on Small Datasets* (November 2026 course review).
- **VERIFIED FROM REVIEW:** The review's Sections 5–8 discuss validation leakage, small-data instability, unequal search budgets, agent nondeterminism, and limited cost comparisons. Its proposed benchmark is not a performed experiment.
- **VERIFIED FROM PUBLICATIONS:** Varma & Simon (BMC Bioinformatics, 2006) discuss model-selection bias and nested CV; Akiba et al. (KDD 2019) document Optuna; Gijsbers et al. (JMLR 2024) benchmark AutoML with controlled protocols. See `paper/references.bib`.
- **INFERRED, NOT NOVELTY CLAIM:** A small-budget comparison between optimized and fixed ML pipelines may reveal dataset-dependent trade-offs.
- **Decision:** Choose assignment pathway B (controlled comparative study), not a fabricated new algorithm.
- **Question:** Under common evaluation splits, does 12-trial Optuna-TPE exceed an RBF SVM tuned by a 12-point grid, and how does it compare to matched-trial random sampling?
- **Secondary question:** Does extending an early search prefix of four trials to 12 alter held-out generalization and measured cost?
- **Datasets:** Breast Cancer, Wine, Iris (primary tabular tasks); Digits600 (supplementary numeric vectors derived from images, clearly marked).
- **Method restrictions:** Optuna searches across SVM and Random Forest; Logistic Regression and fixed RF are additional reference baselines; do not label Optuna as FLAML or label an algorithmically tuned SVM as a professional human expert.
- **Controls:** Outer stratified CV, shared inner CV splits, fold-fitted preprocessing, 2 seeds, fixed trial counts, predeclared balanced accuracy and macro-F1.
- **Resource feasibility:** Local Linux CPU, no usable network installation of FLAML; record exact environment and measured run log.
- **UNKNOWN / NOT TESTED:** Full agentic ML, TabPFN, large OpenML collections, externally validated human expertise, larger HPO budgets and independent Colab reproduction.
- **Critical interpretation rule:** A higher observed score is not proof of statistical superiority or guaranteed near-optimality.
- **Actual execution:** Run `src/benchmark.py`, then `src/verify_outputs.py` and `src/plots.py` to regenerate evidence.
