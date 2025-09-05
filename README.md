# arc-leakage-tscv
Runnable artefacts for the ARC paper: feature-lineage template and structure-aware cross-validation (rolling-origin time series + GroupKFold) with per-fold preprocessing. Includes locked-holdout protocol, seed manifest, and R/Python examples. Permissive license; Zenodo-citable (DOI).

<!-- Badges: add after first release -->
<!-- ![License](https://img.shields.io/badge/license-Apache--2.0-blue) [![DOI](https://zenodo.org/badge/1043130443.svg)](https://doi.org/10.5281/zenodo.16931847) -->

## Layout
```arc-leakage-tscv/
├─ README.md
├─ CITATION.cff
├─ LICENSE-APACHE
├─ LICENSE-CC-BY
├─ templates/
│  └─ feature_lineage.csv
├─ tscv/
│  ├─ rolling_origin.R
│  └─ blocked_cv.py
└─ .github/
   └─ workflows/ci.yml
```
## Prerequisites
**R:** rsample, dplyr, yardstick, glmnet, tibble  
**Python:** scikit-learn (≥1.1), pandas, numpy

```r
install.packages(c("rsample","dplyr","yardstick","glmnet","tibble"))
```
```python
pip install scikit-learn pandas numpy
Rscript tscv/rolling_origin.R
python tscv/blocked_cv.py
```
```r install
Rscript tscv/rolling_origin.R
```
``` python install
python tscv/blocked_cv.py
```
## Case Study: Responsible AutoML Deployment with ARC

This repository contains a case study applying the Automation Risk Checklist (ARC) framework to the UCI Diabetes 130-US hospitals dataset. (http://dx.doi.org/10.1155/2014/781670)

The dataset includes over 100,000 hospital encounters, with demographic, diagnostic, treatment, and outcome information. The prediction task is 30-day hospital readmission.

Our study demonstrates how naive use of AutoML pipelines can lead to misleadingly perfect performance when post-discharge features are inadvertently included (e.g., discharge_disposition_id). By systematically applying ARC, we:

* Detect and exclude leakage variables through ablation checks.

* Ensure patient-level independence with GroupKFold validation.

* Assess calibration of predicted probabilities and improve them with temperature scaling.

* Evaluate fairness across gender and racial subgroups using equalised odds.

* Summarise findings in a structured ARC checklist and composite audit report.

Baseline discrimination under ARC-clean conditions is modest (AUROC ~0.64), consistent with prior work, underscoring the importance of governance checks in avoiding spurious overestimation of AutoML performance.
