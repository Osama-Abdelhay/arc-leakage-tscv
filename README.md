# arc-leakage-tscv
Runnable artefacts for the ARC paper: feature-lineage template and structure-aware cross-validation (rolling-origin time series + GroupKFold) with per-fold preprocessing. Includes locked-holdout protocol, seed manifest, and R/Python examples. Permissive license; Zenodo-citable (DOI).

<!-- Badges: add after first release -->
<!-- ![License](https://img.shields.io/badge/license-Apache--2.0-blue) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD) -->

## Layout

## Prerequisites
**R:** rsample, dplyr, yardstick, glmnet, tibble  
**Python:** scikit-learn (≥1.1), pandas, numpy

```r
install.packages(c("rsample","dplyr","yardstick","glmnet","tibble"))
pip install scikit-learn pandas numpy
Rscript tscv/rolling_origin.R
python tscv/blocked_cv.py
