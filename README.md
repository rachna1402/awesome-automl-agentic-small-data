# Awesome AutoML and Agentic ML for Small-Data Tabular Learning

A curated research repository on **Benchmarking AutoML and Agentic Pipelines Against Expert-Tuned Models on Small Datasets**.

**Student:** Rachna Gupta  
**Roll Number:** RSI2026510  
**Assigned Topic:** Benchmarking AutoML and Agentic Pipelines Against Expert-Tuned Models on Small Datasets  
**Topic ID:** 27

## Contents
- [Overview](#overview)
- [AI-Assisted Research Paper](#ai-assisted-research-paper)
- [Citation Integrity Audit](#citation-integrity-audit)
- [Survey and Review Papers](#survey-and-review-papers)
- [Foundational Papers](#foundational-papers)
- [Recent Research](#recent-research)
- [Small-Data and Benchmark Research](#small-data-and-benchmark-research)
- [Datasets](#datasets)
- [Tools and Libraries](#tools-and-libraries)
- [GitHub Implementations](#github-implementations)
- [Tutorials and Learning Resources](#tutorials-and-learning-resources)
- [License](#license)

## Overview
Small-data tabular machine learning is challenging because limited observations can make model selection, validation, and hyperparameter optimization statistically unstable. AutoML reduces manual work by automating parts of preprocessing, model selection, and hyperparameter optimization. More recent agentic systems use large language models to plan, execute, evaluate, and revise machine-learning workflows. Tabular foundation models such as TabPFN represent another approach aimed at small-data prediction.

This repository organizes research around how expert-tuned models, conventional AutoML, agentic ML, and small-data-specialized models can be compared fairly. Important dimensions include predictive performance, validation design, computational budget, human effort, reproducibility, search-space restrictions, information access, and cost.

## AI-Assisted Research Paper
**Benchmarking AutoML and Agentic Pipelines Against Expert-Tuned Models on Small Datasets**  
*A Literature-Grounded Benchmarking Framework for Small-Data Tabular Machine Learning*

[View the AI-assisted research paper](paper/AI_Assisted_Research_Paper.pdf)

## Citation Integrity Audit
References were checked for title, authors, year, venue, existence, and a matching scholarly/publisher link.

[View the Citation Integrity Audit](citation-audit/Citation_Integrity_Audit.pdf)

## Survey and Review Papers
- **Automated Machine Learning: Methods, Systems, Challenges** — Hutter, Kotthoff & Vanschoren, 2019. [Springer](https://doi.org/10.1007/978-3-030-05318-5)
- **A Brief Introduction to AutoML** — Hutter, 2019. [arXiv](https://arxiv.org/abs/1908.00709)
- **Automated Machine Learning: State-of-the-Art and Open Challenges** — Feurer & Hutter, 2019. [arXiv](https://arxiv.org/abs/1904.12054)
- **AutoML: A Survey of the State-of-the-Art** — He, Zhao & Chu, 2021. [DOI](https://doi.org/10.1016/j.knosys.2021.106622)

## Foundational Papers
- **Efficient and Robust Automated Machine Learning** — Feurer et al., 2015. [NeurIPS](https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html) — Auto-sklearn, meta-learning, ensembles.
- **Sequential Model-Based Optimization for General Algorithm Configuration** — Hutter et al., 2011. [Springer DOI](https://doi.org/10.1007/978-3-642-25566-3_40) — SMBO foundation.
- **Auto-WEKA: Automatic Model Selection and Hyperparameter Optimization in WEKA** — Thornton et al., 2013. [Springer](https://doi.org/10.1007/s10994-013-5360-0) — Algorithm selection + HPO.
- **Random Search for Hyper-Parameter Optimization** — Bergstra & Bengio, 2012. [JMLR](https://jmlr.org/papers/v13/bergstra12a.html) — HPO baseline.
- **Practical Bayesian Optimization of Machine Learning Algorithms** — Snoek, Larochelle & Adams, 2012. [NeurIPS](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms) — Bayesian HPO.
- **TPOT: A Tree-based Pipeline Optimization Tool for Automating Machine Learning** — Olson & Moore, 2016. [PMLR](https://proceedings.mlr.press/v64/olson_tpot_2016.html) — Genetic-programming pipeline search.
- **Bayesian Optimization for Automated Model Selection** — Malkomes, Schaff & Garnett, 2016. [PMLR](https://proceedings.mlr.press/v64/malkomes_bayesian_2016.html) — Automated model selection.

## Recent Research
- **Hyperband** — Li et al., 2018. [JMLR](https://jmlr.org/papers/v18/16-558.html) — Adaptive resource allocation and early stopping.
- **BOHB** — Falkner, Klein & Hutter, 2018. [PMLR](https://proceedings.mlr.press/v80/falkner18a.html) — Bayesian + bandit HPO.
- **AutoGluon-Tabular** — Erickson et al., 2020. [arXiv](https://arxiv.org/abs/2003.06505) — Tabular AutoML and ensembling.
- **FLAML** — Wang et al., 2021. [arXiv](https://arxiv.org/abs/1911.04768) — Fast/lightweight AutoML.
- **AMLB: an AutoML Benchmark** — Gijsbers et al., 2024. [JMLR](https://www.jmlr.org/papers/v25/22-0493.html) — Controlled AutoML benchmarking.
- **TabPFN** — Hollmann et al., 2023. [OpenReview](https://openreview.net/forum?id=eu9fVjVasr4) — Small-data tabular foundation model.
- **AutoML-Agent** — Trirat, Jeong & Hwang, 2025. [PMLR](https://proceedings.mlr.press/v267/trirat25a.html) — Multi-agent LLM AutoML.

## Small-Data and Benchmark Research
- **Squeezing Lemons with Hammers** — Knauer & Rodner, 2024. [arXiv](https://arxiv.org/abs/2405.07662) — 44 data-scarce tabular classification datasets.
- **PMLBmini** — Knauer, Grimm & Rodner, 2024. [arXiv](https://arxiv.org/abs/2409.01635) — 44 binary classification datasets with at most 500 samples.
- **AutoML Benchmark with shorter time constraints and early stopping** — Campero Jurado, Gijsbers & Vanschoren, 2025. [arXiv](https://arxiv.org/abs/2504.01222) — Budget-sensitive benchmarking.
- **Benchmarking AutoML for regression tasks on small tabular data in materials design** — 2022. [Scientific Reports DOI](https://doi.org/10.1038/s41598-022-23327-1) — Small-data AutoML evaluation.
- **Evaluation of Representation Models for Text Classification with AutoML Tools** — Brändle et al., 2021. [arXiv](https://arxiv.org/abs/2106.12798) — AutoML evaluation.

## Datasets
### 1. UCI Adult
Mixed-type tabular classification dataset.  
https://archive.ics.uci.edu/dataset/2/adult

### 2. Breast Cancer Wisconsin (Diagnostic)
Compact binary classification dataset.  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### 3. Wine Quality
Tabular prediction dataset for quality prediction.  
https://archive.ics.uci.edu/dataset/186/wine+quality

## Tools and Libraries
1. **auto-sklearn** — automated model selection, preprocessing, HPO and ensembles.  
   https://automl.github.io/auto-sklearn/master/
2. **AutoGluon** — automated tabular modeling and ensembling.  
   https://auto.gluon.ai/
3. **FLAML** — fast, lightweight AutoML and tuning.  
   https://microsoft.github.io/FLAML/
4. **TPOT** — genetic-programming pipeline optimization.  
   https://epistasislab.github.io/tpot/
5. **TabPFN** — small-tabular-data foundation model implementation.  
   https://github.com/PriorLabs/TabPFN

## GitHub Implementations
- https://github.com/automl/auto-sklearn
- https://github.com/autogluon/autogluon
- https://github.com/microsoft/FLAML
- https://github.com/EpistasisLab/tpot
- https://github.com/PriorLabs/TabPFN

## Tutorials and Learning Resources
1. AutoGluon Tabular Quick Start — https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html
2. FLAML Task-Oriented AutoML — https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/
3. auto-sklearn Documentation — https://automl.github.io/auto-sklearn/master/
4. TPOT Documentation — https://epistasislab.github.io/tpot/
5. TabPFN Repository and Examples — https://github.com/PriorLabs/TabPFN

## License
Original repository documentation: MIT License. Third-party resources remain under their own licenses. This repository links to papers and resources rather than redistributing copyrighted PDFs.
