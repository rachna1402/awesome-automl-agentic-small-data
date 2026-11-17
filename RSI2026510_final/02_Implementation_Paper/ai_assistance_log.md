# AI-assisted coding provenance and required student verification

## Recorded AI work in this conversation (authentic to this artifact)
1. Compared the two course PDFs and the prior review to choose a tractable controlled-comparison experiment.
2. Inspected earlier generated experiment files, including source, metrics, predictions and manuscript.
3. Recomputed earlier metrics from saved outer-fold predictions; validated their arithmetic.
4. Generated a new repeated nested-CV benchmark, added matched sampler spaces, fixed inner-fold randomization, and implemented CSV output recording.
5. Executed the benchmark in the provided compute environment; retained the actual run log and result files.
6. Wrote a separate independent output-validation script, ran it, and generated charts from saved results.
7. Generated an evidence-grounded implementation manuscript and reproduction notebook.

## AI limitations and concrete corrections observed
- The earlier baseline used one random seed and different inner-fold randomization for tuned-SVM vs automated search. New code uses shared inner fold indices.
- The earlier automated method used eight trials, too few to claim near-optimal search. New code records an explicitly bounded 12-trial study and a search-budget ablation, with no optimality claim.
- The Digits task consists of image-derived numeric feature vectors, so it is not presented as an ordinary tabular dataset; the new benchmark adds Iris and explicitly separates primary tabular analysis.
- FLAML and a fully functional LLM agent are NOT executed here. The automated optimizer is Optuna; paper claims are adjusted accordingly.

## Actions the student must perform and record personally
- Read and understand each source module and paper section.
- Rerun the notebook or script on their own Colab/laptop, keep authentic console output, note discrepancies.
- Verify dataset provenance and the published bibliography; explain why folds do not leak test labels.
- Record their own actual debugging, decisions, corrections and resulting command/logs. Do not sign a declaration claiming independent execution that did not occur.
- Obtain institution-approved similarity and AI-writing reports for both papers.
