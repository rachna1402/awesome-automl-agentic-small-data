# End-sem evaluation readiness and authentic evidence

**Course evaluation:** 18 November 2026, 9:00–11:00 AM (from instructor notice).

## Bring or upload
- Earlier review paper PDF (the November 2026 literature-grounded review).
- `paper/manuscript.pdf`: new experimental paper with citations and measured results.
- Source code, Colab notebook, CSV outputs, plots and `results/logs/final.log` from this package.
- `results/verification_report.json`: arithmetic/data-split audit that returns PASS.
- `research_decisions.md`, `baseline_validation.md`, `ai_assistance_log.md`: evidence/provenance.
- GitHub repository after the student pushes these files using their own account (not yet pushed by the assistant).
- Four official institution-accepted reports: (1) review similarity/plagiarism, (2) review AI-writing, (3) implementation similarity/plagiarism, (4) implementation AI-writing. **None is generated or asserted by this project.**

## Five-minute demonstration outline (no invented execution)
- **0:00--0:45** Review theme: unreliable model ranking and cost effects on small datasets; state narrower research question.
- **0:45--1:30** Show the dataset profile and exact repeated nested-CV split protocol, including the train-only preprocessing pipelines.
- **1:30--2:30** Open `src/benchmark.py`; explain Logistic Regression, Random Forest, 12-grid SVM, random 12-trial Optuna and TPE 12-trial study with four-trial prefix.
- **2:30--3:25** Open actual `results/tabular_only_summary.csv` and `results/overall_summary.csv`; explain why their leading methods can differ because Digits is supplementary.
- **3:25--4:10** Show actual figure `results/figures/budget_effect.pdf`; explain why more trials do not guarantee test improvement and folds are dependent.
- **4:10--5:00** Run `python src/verify_outputs.py` to show the PASS result, highlight provenance and explain limitations. For a full rerun use the separate Colab notebook.

## Likely viva questions
- What is balanced accuracy and why use it? Average recall over classes; avoids raw-accuracy dominance when class counts vary.
- What is nested CV? Inner folds tune and outer folds assess on unseen data.
- Why are inner fold indices identical across optimizers? Avoid changing validation partitions as a comparison confound.
- What is TPE4? Best configuration after four random startup trials in the TPE study, not four iterations of model-based TPE.
- Are results near-optimal? Unknown; a 12-trial bounded search cannot prove near-optimality.
- Is there an implemented LLM agent? No; the study selected the approved controlled comparison pathway. This must not be presented as a complete AutoML/agentic comparison.
- Why not report p-values? Two seeds/three data tasks are too few independent data conditions and CV folds overlap.
- How did AI help? See the authentic assistance log; the student must understand and independently verify the code before claiming their own execution.
