# === Baseline Models (Vanilla) ===============================================
# 0) Imports & setup
import pandas as pd
import numpy as np
from pathlib import Path
import sklearn

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression

# Optional: XGBoost / LightGBM (skip gracefully if not installed)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# 1) Load & define target
base = Path("Path/")
df = pd.read_csv(base / "diabetic_data.csv")
df = df.replace('?', np.nan)

# Binary target: readmission <30 days
df['y'] = (df['readmitted'] == '<30').astype(int)

# 2) Features (baseline: include all non-ID, non-target columns)
id_cols = ['encounter_id', 'patient_nbr']
drop_cols = id_cols + ['y']  # keep 'readmitted' as a regular feature (baseline = naive)
Xcols = [c for c in df.columns if c not in drop_cols]
X = df[Xcols].copy()
y = df['y'].values

# 3) Preprocessing
#    - numeric: median impute + standardize
#    - categorical: most_frequent impute + one-hot encode
num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))  # keeps CSR-friendly
])
# Detect correct param
ohe_kwargs = {"handle_unknown": "ignore"}
if sklearn.__version__ >= "1.2":
    ohe_kwargs["sparse_output"] = True
else:
    ohe_kwargs["sparse"] = True

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(**ohe_kwargs))
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], remainder="drop", sparse_threshold=1.0)

# 4) Helper: evaluate a pipeline via stratified CV (vanilla, no tuning)
def eval_baseline(pipe, X, y, name, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred  = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "AUROC": roc_auc_score(y, proba),
        "AP": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "Accuracy": accuracy_score(y, pred)
    }
    return metrics

results = []

# 5) Logistic Regression (vanilla)
logreg = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))
])
results.append(eval_baseline(logreg, X, y, "LogisticRegression"))

# 6) XGBoost (vanilla, if available)
if HAS_XGB:
    xgb = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=100,  # vanilla
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
    results.append(eval_baseline(xgb, X, y, "XGBoost"))
else:
    results.append({"model": "XGBoost", "AUROC": None, "AP": None, "Brier": None, "Accuracy": None})

# 7) LightGBM (vanilla, if available)
if HAS_LGBM:
    lgbm = Pipeline([
        ("pre", pre),
        ("clf", LGBMClassifier(
            random_state=42,  # vanilla defaults
            n_jobs=-1
        ))
    ])
    results.append(eval_baseline(lgbm, X, y, "LightGBM"))
else:
    results.append({"model": "LightGBM", "AUROC": None, "AP": None, "Brier": None, "Accuracy": None})

# 8) Report
out = pd.DataFrame(results).set_index("model")
print(out)

# === Improved Baseline v2 Models ============================================

import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.base import clone

# 1) Numeric correlations with target
num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
corrs = {}
for col in num_cols:
    corrs[col] = np.corrcoef(df[col].fillna(0), y)[0,1] if df[col].notna().sum()>0 else 0
corrs = pd.Series(corrs).sort_values(key=lambda x: -abs(x))

# Plot correlation heatmap (numeric only)
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# Drop highly correlated (>0.95)
drop_high_corr = set()
corr_matrix = df[num_cols].corr().abs()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i,j] > 0.95:
            colname = corr_matrix.columns[i]
            drop_high_corr.add(colname)

num_keep = [c for c in num_cols if c not in drop_high_corr]

# 2) Univariate selection for categorical
cat_cols = [c for c in X.columns if c not in num_cols]
cat_data = X[cat_cols].fillna("MISSING")

# Encode categoricals temporarily with label counts for χ²
cat_data_enc = pd.get_dummies(cat_data, dummy_na=True)
selector = SelectKBest(chi2, k=min(20, cat_data_enc.shape[1]))
selector.fit(cat_data_enc, y)
cat_keep = cat_data_enc.columns[selector.get_support(indices=True)].tolist()

# Final feature set for v2
Xv2 = pd.concat([df[num_keep], cat_data_enc[cat_keep]], axis=1)

# 3) Preprocessing for v2
pre_v2 = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_keep),
    ("cat", "passthrough", cat_keep)  # already one-hot encoded
])

def eval_model(pipe, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:,1]
    pred = (proba >= 0.5).astype(int)
    return pd.Series({
        "AUROC": roc_auc_score(y, proba),
        "AP": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "Accuracy": accuracy_score(y, pred)
    }, name=name)

results_v2 = []

# Logistic Regression v2
logit_v2 = Pipeline([("pre", pre_v2),
                     ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))])
results_v2.append(eval_model(logit_v2, Xv2, y, "LogisticRegression_v2"))

# XGBoost v2
if HAS_XGB:
    xgb_v2 = Pipeline([("pre", pre_v2),
                       ("clf", XGBClassifier(n_estimators=100, use_label_encoder=False,
                                             eval_metric="logloss", random_state=42, n_jobs=-1))])
    results_v2.append(eval_model(xgb_v2, Xv2, y, "XGBoost_v2"))

# LightGBM v2
if HAS_LGBM:
    lgbm_v2 = Pipeline([("pre", pre_v2),
                        ("clf", LGBMClassifier(random_state=42, n_jobs=-1))])
    results_v2.append(eval_model(lgbm_v2, Xv2, y, "LightGBM_v2"))

out_v2 = pd.DataFrame(results_v2)
print(out_v2)
# === Task 3 — ARC-clean Vanilla Models (no tuning) ============================
import pandas as pd, numpy as np, sklearn
from pathlib import Path

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression

# Optional: XGBoost / LightGBM (use if installed)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# --- Load & target -----------------------------------------------------------
base = Path("Path/")
df = pd.read_csv(base / "diabetic_data.csv").replace("?", np.nan)

# Target: readmitted within 30 days
df["y"] = (df["readmitted"] == "<30").astype(int)

# --- ARC: define post-index (leakage) features to exclude --------------------
# INDEX_TIME = at admission (early in encounter)
LEAKAGE_COLS = [
    "time_in_hospital",          # length of stay -> known at/after discharge
    "discharge_disposition_id",  # outcome at discharge (post-index)
]
ID_COLS = ["encounter_id", "patient_nbr"]
TARGET_COLS = ["y", "readmitted"]  # ensure target & raw label excluded from X

# Construct clean feature set
drop_cols = set(LEAKAGE_COLS + ID_COLS + TARGET_COLS)
Xcols = [c for c in df.columns if c not in drop_cols]
X = df[Xcols].copy()
y = df["y"].values
groups = df["patient_nbr"].astype(str).values

# --- Preprocessor (numeric/categorical) with OHE version compatibility -------
ohe_kwargs = {"handle_unknown": "ignore"}
ohe_kwargs["sparse_output" if sklearn.__version__ >= "1.2" else "sparse"] = True

num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
cat_cols = [c for c in X.columns if c not in num_cols]

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler(with_mean=False)),  # CSR-friendly
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(**ohe_kwargs)),
])

pre = ColumnTransformer(
    [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
    remainder="drop",
    sparse_threshold=1.0,
)

# --- Helper: grouped CV evaluation ------------------------------------------
def eval_grouped_cv(pipe, X, y, groups, name, n_splits=5, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(
        pipe, X, y, cv=gkf, method="predict_proba", groups=groups
    )[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pd.Series(
        {
            "AUROC": roc_auc_score(y, proba),
            "AP": average_precision_score(y, proba),
            "Brier": brier_score_loss(y, proba),
            "Accuracy": accuracy_score(y, pred),
        },
        name=name,
    )

results = []

# --- 1) Logistic Regression (vanilla) ----------------------------------------
logit = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)),
])
results.append(eval_grouped_cv(logit, X, y, groups, "LogisticRegression_ARCclean"))

# --- 2) XGBoost (vanilla) ----------------------------------------------------
if HAS_XGB:
    xgb = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=100,  # vanilla
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    results.append(eval_grouped_cv(xgb, X, y, groups, "XGBoost_ARCclean"))
else:
    results.append(pd.Series(
        {"AUROC": np.nan, "AP": np.nan, "Brier": np.nan, "Accuracy": np.nan},
        name="XGBoost_ARCclean",
    ))

# --- 3) LightGBM (vanilla) ---------------------------------------------------
if HAS_LGBM:
    lgbm = Pipeline([
        ("pre", pre),
        ("clf", LGBMClassifier(random_state=42, n_jobs=-1)),
    ])
    results.append(eval_grouped_cv(lgbm, X, y, groups, "LightGBM_ARCclean"))
else:
    results.append(pd.Series(
        {"AUROC": np.nan, "AP": np.nan, "Brier": np.nan, "Accuracy": np.nan},
        name="LightGBM_ARCclean",
    ))

# --- Report ------------------------------------------------------------------
out_arc_clean = pd.DataFrame(results)
print("Excluded (leakage) columns actually found & dropped:",
      [c for c in LEAKAGE_COLS if c in df.columns])
print("n_features used:", X.shape[1])
print(out_arc_clean)
# === Leakage Ablation: quantify impact of "bad" features =====================
import pandas as pd, numpy as np, sklearn
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Optional: tree baselines too (set flags below)
USE_XGB = True
USE_LGBM = True
try:
    from xgboost import XGBClassifier
except Exception:
    USE_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    USE_LGBM = False

# --- 0) Load and basic ARC-clean --------------------------------------------
base = Path("Path/")
df = pd.read_csv(base / "diabetic_data.csv").replace("?", np.nan)
df["y"] = (df["readmitted"] == "<30").astype(int)

LEAKAGE_POOL = [
    "time_in_hospital",          # length of stay (post-index)
    "discharge_disposition_id",  # discharge outcome (post-index)
    # add more suspects if desired:
    # "readmitted", "number_diagnoses", "diag_1","diag_2","diag_3"
]
ID_COLS = ["encounter_id", "patient_nbr"]
TARGET_COLS = ["y", "readmitted"]

# Base clean feature set (ARC-clean)
drop_base = set(ID_COLS + TARGET_COLS + LEAKAGE_POOL)  # clean excludes suspects
X_base_cols = [c for c in df.columns if c not in drop_base]
y = df["y"].values
groups = df["patient_nbr"].astype(str).values

# --- 1) Utilities -------------------------------------------------------------
ohe_kwargs = {"handle_unknown": "ignore"}
ohe_kwargs["sparse_output" if sklearn.__version__ >= "1.2" else "sparse"] = True

def make_preprocessor(X):
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False))
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(**ohe_kwargs))
            ]), cat_cols),
        ],
        remainder="drop", sparse_threshold=1.0
    )
    return pre

def eval_grouped_cv(pipe, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(pipe, X, y, cv=gkf, method="predict_proba", groups=groups)[:,1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y, proba),
        "AP": average_precision_score(y, proba),
        "Brier": brier_score_loss(y, proba),
        "Accuracy": accuracy_score(y, pred)
    }

def build_models(pre):
    models = {
        "Logistic": Pipeline([("pre", pre),
                              ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))])
    }
    if USE_XGB:
        models["XGBoost"] = Pipeline([("pre", pre),
                                      ("clf", XGBClassifier(n_estimators=100, use_label_encoder=False,
                                                            eval_metric="logloss", random_state=42, n_jobs=-1))])
    if USE_LGBM:
        models["LightGBM"] = Pipeline([("pre", pre),
                                       ("clf", LGBMClassifier(random_state=42, n_jobs=-1))])
    return models

def run_setting(name, cols):
    Xs = df[cols].copy()
    pre = make_preprocessor(Xs)
    models = build_models(pre)
    rows = []
    for mname, pipe in models.items():
        met = eval_grouped_cv(pipe, Xs, y, groups, n_splits=5)
        rows.append({"Setting": name, "Model": mname, **met, "n_features": len(cols)})
    return rows

# --- 2) Define ablation settings ---------------------------------------------
settings = []

# ARC-clean (baseline clean)
settings.append(("ARCclean", X_base_cols))

# Add each leakage variable back individually
for leak in LEAKAGE_POOL:
    if leak in df.columns:
        cols = X_base_cols + [leak]
        settings.append((f"ARCclean+{leak}", cols))

# Add BOTH leakage variables together
both = [c for c in LEAKAGE_POOL if c in df.columns]
if both:
    settings.append((f"ARCclean+{'&'.join(both)}", X_base_cols + both))

# --- 3) Run all settings ------------------------------------------------------
all_rows = []
for name, cols in settings:
    all_rows.extend(run_setting(name, cols))

ablation = pd.DataFrame(all_rows).sort_values(["Setting", "Model"]).reset_index(drop=True)
print(ablation)

# --- 4) Optional: AUROC bar plot ---------------------------------------------
plt.figure(figsize=(9, 4))
for m in ablation["Model"].unique():
    subset = ablation[ablation["Model"]==m]
    plt.plot(subset["Setting"], subset["AUROC"], marker="o", label=m)
plt.xticks(rotation=30, ha="right")
plt.ylabel("AUROC")
plt.title("Leakage Ablation: AUROC by setting (GroupKFold, no tuning)")
plt.legend()
plt.tight_layout()
plt.show()
# === ARC Calibration Plots (Before/After Scaling) ===========================
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression as CalibLogit

def calibration_summary(y_true, proba, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy='quantile')
    # logistic regression on logit(p) to estimate slope/intercept
    eps = 1e-6
    p = np.clip(proba, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1, 1)
    lr = CalibLogit(solver="lbfgs").fit(X_logit, y_true)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return prob_true, prob_pred, slope, intercept, lr

def plot_calibration(y_true, proba, model_name):
    prob_true, prob_pred, slope, intercept, _ = calibration_summary(y_true, proba)
    plt.plot(prob_pred, prob_true, marker="o", label=f"{model_name} (slope={slope:.2f})")
    return slope, intercept

def apply_temperature_scaling(y_val, proba_val, proba_test):
    # Fit scaling on validation set
    eps = 1e-6
    p = np.clip(proba_val, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1, 1)
    calib = CalibLogit(solver="lbfgs").fit(X_logit, y_val)
    # Apply to test predictions
    ptest = np.clip(proba_test, eps, 1-eps)
    X_logit_test = np.log(ptest/(1-ptest)).reshape(-1, 1)
    return calib.predict_proba(X_logit_test)[:,1]

# --- Example: run calibration for Logistic ARCclean -------------------------
# Use your OOF predictions from GroupKFold ARC-clean (say LogisticRegression_ARCclean)
from sklearn.model_selection import GroupKFold

def get_oof(pipe, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(pipe, X, y, cv=gkf, method="predict_proba", groups=groups)[:,1]
    return proba

# Rebuild ARC-clean Logistic
logit_arc = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))
])
proba_logit = get_oof(logit_arc, df[X_base_cols], y, groups)

# Calibration curves before/after
plt.figure(figsize=(6,5))
plt.plot([0,1],[0,1],"--", color="gray", label="Perfectly calibrated")

# Before scaling
slope, intercept = plot_calibration(y, proba_logit, "LogReg ARCclean (raw)")

# After scaling: split 80/20 dev/val for calibration fit
from sklearn.model_selection import train_test_split
idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
proba_scaled = apply_temperature_scaling(y[idx_val], proba_logit[idx_val], proba_logit)

slope2, intercept2 = plot_calibration(y, proba_scaled, "LogReg ARCclean (temp-scaled)")

plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Reliability curves (ARCclean Logistic)")
plt.legend()
plt.show()

print(f"Raw calibration slope={slope:.2f}, intercept={intercept:.2f}")
print(f"Scaled calibration slope={slope2:.2f}, intercept={intercept2:.2f}")


# === ARC-clean: Calibration for Logistic / XGBoost / LightGBM ===============
import numpy as np, pandas as pd, sklearn, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression as SkLogit
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline

# Optional tree models
HAS_XGB, HAS_LGBM = True, True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGBM = False

# --- helpers -----------------------------------------------------------------
def get_oof(pipe, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(pipe, X, y, cv=gkf, method="predict_proba", groups=groups)[:,1]
    return proba

def temp_scale_fit(y_val, p_val):
    """Platt/temperature scaling: fit a logistic reg on logit(p)."""
    eps = 1e-6
    p = np.clip(p_val, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1,1)
    lr = SkLogit(solver="lbfgs", max_iter=1000).fit(X_logit, y_val)
    return lr

def temp_scale_apply(lr, p):
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1,1)
    return lr.predict_proba(X_logit)[:,1]

def cal_summary(y_true, p, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    eps = 1e-6
    pp = np.clip(p, eps, 1-eps)
    X_logit = np.log(pp/(1-pp)).reshape(-1,1)
    lr = SkLogit(solver="lbfgs", max_iter=1000).fit(X_logit, y_true)
    slope = float(lr.coef_.ravel()[0]); intercept = float(lr.intercept_.ravel()[0])
    return prob_true, prob_pred, slope, intercept

def discr_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        AUROC = roc_auc_score(y, p),
        AP    = average_precision_score(y, p),
        Brier = brier_score_loss(y, p)
    )

# --- models (ARC-clean, no tuning) ------------------------------------------
def make_logreg(pre): 
    return Pipeline([("pre", pre), ("clf", SkLogit(solver="liblinear", max_iter=1000, random_state=42))])

def make_xgb(pre):
    return Pipeline([("pre", pre), ("clf", XGBClassifier(
        n_estimators=100, eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1
    ))])

def make_lgbm(pre):
    return Pipeline([("pre", pre), ("clf", LGBMClassifier(random_state=42, n_jobs=-1))])

constructors = [("Logistic", make_logreg)]
if HAS_XGB:  constructors.append(("XGBoost", make_xgb))
if HAS_LGBM: constructors.append(("LightGBM", make_lgbm))

# --- run & plot --------------------------------------------------------------
save_dir = Path("/mnt/data"); save_dir.mkdir(parents=True, exist_ok=True)
rows = []

for name, ctor in constructors:
    model = ctor(pre)
    # OOF probabilities (GroupKFold)
    p_raw = get_oof(model, df[X_base_cols], y, groups, n_splits=5)

    # Split a small validation slice just to fit the temperature scaler
    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    scaler = temp_scale_fit(y[idx_val], p_raw[idx_val])
    p_scaled = temp_scale_apply(scaler, p_raw)

    # Summaries
    prob_true_raw, prob_pred_raw, slope_raw, int_raw = cal_summary(y, p_raw)
    prob_true_s,   prob_pred_s,   slope_s,   int_s   = cal_summary(y, p_scaled)

    # Discrimination (should be unchanged or near-identical after scaling)
    d_raw = discr_metrics(y, p_raw)
    d_s   = discr_metrics(y, p_scaled)

    # Plot reliability curves
    plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1],"--", color="gray", label="Perfectly calibrated")
    plt.plot(prob_pred_raw, prob_true_raw, marker="o", label=f"{name} raw (slope={slope_raw:.2f})")
    plt.plot(prob_pred_s,   prob_true_s,   marker="o", label=f"{name} scaled (slope={slope_s:.2f})")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Reliability curves — {name} (ARC-clean)")
    plt.legend()
    fname = save_dir / f"Calibration_{name.replace(' ','_')}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=160); plt.show()
    print(f"Saved: {fname}")

    rows.append(dict(
        Model=name,
        AUROC_raw=d_raw["AUROC"],  AP_raw=d_raw["AP"],  Brier_raw=d_raw["Brier"],
        AUROC_scaled=d_s["AUROC"], AP_scaled=d_s["AP"], Brier_scaled=d_s["Brier"],
        Slope_raw=slope_raw, Intercept_raw=int_raw,
        Slope_scaled=slope_s, Intercept_scaled=int_s
    ))

calib_summary = pd.DataFrame(rows)
print(calib_summary)
# === Calibration panel: Logistic + XGBoost + LightGBM (ARC-clean, no tuning) ===
import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression as SkLogit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---- Load + ARC-clean feature set
base = Path("Path/")
df = pd.read_csv(base / "diabetic_data.csv").replace("?", np.nan)
df["y"] = (df["readmitted"]=="<30").astype(int)
y = df["y"].values
groups = df["patient_nbr"].astype(str).values
LEAKAGE = ["time_in_hospital","discharge_disposition_id"]
DROP = set(LEAKAGE + ["encounter_id","patient_nbr","y","readmitted"])
Xcols = [c for c in df.columns if c not in DROP]

# Preprocessor (OHE version-compat)
ohe_kwargs = {"handle_unknown":"ignore"}
ohe_kwargs["sparse_output" if sklearn.__version__ >= "1.2" else "sparse"] = True
num_cols = [c for c in Xcols if np.issubdtype(df[c].dtype, np.number)]
cat_cols = [c for c in Xcols if c not in num_cols]
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler(with_mean=False))])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(**ohe_kwargs))])
pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)],
                        remainder="drop", sparse_threshold=1.0)

# Optional estimators
HAS_XGB, HAS_LGBM = True, True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LGBM = False

# ---- helpers
def oof_proba(pipe, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return cross_val_predict(pipe, X, y, cv=gkf, method="predict_proba", groups=groups)[:,1]

def fit_temp_scaler(y_val, p_val):
    eps = 1e-6
    p = np.clip(p_val, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1,1)
    return SkLogit(solver="lbfgs", max_iter=1000).fit(X_logit, y_val)

def apply_temp_scaler(model, p):
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1,1)
    return model.predict_proba(X_logit)[:,1]

def cal_curve_and_slope(y_true, p, n_bins=10):
    pt, pp = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
    # slope/intercept via logistic on logit(p)
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    X_logit = np.log(p/(1-p)).reshape(-1,1)
    lr = SkLogit(solver="lbfgs", max_iter=1000).fit(X_logit, y_true)
    return pt, pp, float(lr.coef_.ravel()[0]), float(lr.intercept_.ravel()[0])

from sklearn.linear_model import LogisticRegression
def mk_logit(pre):
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            solver="liblinear", 
            max_iter=1000, 
            random_state=42
        ))
    ])
def mk_xgb(pre):
    return Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        ))
    ])

def mk_lgbm(pre):
    return Pipeline([
        ("pre", pre),
        ("clf", LGBMClassifier(
            random_state=42,
            n_jobs=-1
        ))
    ])
constructors = [("Logistic", mk_logit)]
if HAS_XGB:  constructors.append(("XGBoost", mk_xgb))
if HAS_LGBM: constructors.append(("LightGBM", mk_lgbm))

# (Optional) subsample for faster plotting runs; set to None to use all
SAMPLE_N = None
X_plot = df[Xcols].copy()
y_plot = y
groups_plot = groups
if SAMPLE_N is not None and SAMPLE_N < len(df):
    idx = np.random.RandomState(42).choice(len(df), SAMPLE_N, replace=False)
    X_plot = X_plot.iloc[idx].reset_index(drop=True)
    y_plot = y_plot[idx]
    groups_plot = groups_plot[idx]

# Temp-scaling fit split
idx_all = np.arange(len(y_plot))
from sklearn.model_selection import train_test_split
_, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42, stratify=y_plot)

# ---- build panel
n = len(constructors)
fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
axes = axes[0]

panel_rows = []
for ax, (name, ctor) in zip(axes, constructors):
    pipe = ctor(pre)
    p_raw = oof_proba(pipe, X_plot, y_plot, groups_plot, n_splits=5)
    scaler = fit_temp_scaler(y_plot[idx_val], p_raw[idx_val])
    p_scaled = apply_temp_scaler(scaler, p_raw)

    pt_raw, pp_raw, slope_raw, int_raw = cal_curve_and_slope(y_plot, p_raw)
    pt_s,   pp_s,   slope_s,   int_s   = cal_curve_and_slope(y_plot, p_scaled)

    ax.plot([0,1],[0,1],"--", label="Perfectly calibrated")
    ax.plot(pp_raw, pt_raw, marker="o", label=f"{name} raw (slope={slope_raw:.2f})")
    ax.plot(pp_s,   pt_s,   marker="o", label=f"{name} scaled (slope={slope_s:.2f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability — {name} (ARC-clean)")
    ax.legend()

    panel_rows.append({"Model":name, "Slope_raw":slope_raw, "Intercept_raw":int_raw,
                       "Slope_scaled":slope_s, "Intercept_scaled":int_s})

plt.tight_layout()
save_path = Path("/mnt/data/Calibration_Panel.png")
plt.savefig(save_path, dpi=160)
plt.show()

calib_panel_summary = pd.DataFrame(panel_rows)
print(calib_panel_summary)
print(f"Saved to {save_path}")

# --- Clean up duplicated rows, round, and export
calib_panel_summary = calib_panel_summary.drop_duplicates(subset=["Model"]).reset_index(drop=True)

# Round for reporting
cols_to_round = ["Slope_raw", "Intercept_raw", "Slope_scaled", "Intercept_scaled"]
calib_panel_summary[cols_to_round] = calib_panel_summary[cols_to_round].round(2)

print(calib_panel_summary)

# Save CSV alongside the panel figure
import pandas as pd
csv_path = "Path//Calibration_Panel_Summary.csv"
calib_panel_summary.to_csv(csv_path, index=False)
print(f"Saved calibration table: {csv_path}")
print("Panel figure: /mnt/data/Calibration_Panel.png")
# === ARC-clean Fairness Panel ================================================
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

# Assume df, X_base_cols, y, groups, pre already defined
# Use Logistic (ARC-clean) for demo — you can swap in XGB/LGBM

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logit_arc = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(solver="liblinear", max_iter=1000, random_state=42))
])

# Get OOF predictions (GroupKFold)
gkf = GroupKFold(n_splits=5)
proba = cross_val_predict(logit_arc, df[X_base_cols], y, cv=gkf,
                          groups=groups, method="predict_proba")[:,1]
pred = (proba >= 0.5).astype(int)

df_fair = df.copy()
df_fair["y_true"] = y
df_fair["y_pred"] = pred
df_fair["proba"] = proba

# --- helper to compute fairness metrics per group ----------------------------
def subgroup_metrics(df, group_col, ref=None):
    res = []
    for g, sub in df.groupby(group_col):
        tn, fp, fn, tp = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0,1]).ravel()
        tpr = tp/(tp+fn) if (tp+fn)>0 else np.nan
        fpr = fp/(fp+tn) if (fp+tn)>0 else np.nan
        sel = (sub["y_pred"]==1).mean()
        res.append(dict(group=g, n=len(sub), TPR=tpr, FPR=fpr, SelRate=sel))
    out = pd.DataFrame(res).sort_values("n", ascending=False).reset_index(drop=True)
    # Reference = majority group unless specified
    if ref is None:
        ref = out.iloc[0]["group"]
    ref_row = out[out["group"]==ref].iloc[0]
    out["ΔTPR"] = out["TPR"] - ref_row["TPR"]
    out["ΔFPR"] = out["FPR"] - ref_row["FPR"]
    out["SelRateRatio_vsRef"] = out["SelRate"] / ref_row["SelRate"] if ref_row["SelRate"]>0 else np.nan
    out["Reference"] = out["group"].eq(ref)
    return out

fair_gender = subgroup_metrics(df_fair, "gender")
fair_race   = subgroup_metrics(df_fair, "race")

print("=== Fairness by Gender ===")
print(fair_gender)
print("\n=== Fairness by Race ===")
print(fair_race)

# --- Optional: bar plots -----------------------------------------------------
import matplotlib.pyplot as plt

def plot_fairness(df, title, metric="TPR"):
    ref_val = df.loc[df["Reference"], metric].values[0]
    plt.figure(figsize=(6,4))
    plt.bar(df["group"], df[metric], color=["tab:blue" if r else "tab:orange" for r in df["Reference"]])
    plt.axhline(ref_val, color="gray", linestyle="--")
    plt.title(f"{title}: {metric} by group")
    plt.ylabel(metric)
    plt.show()

plot_fairness(fair_gender, "Gender", "TPR")
plot_fairness(fair_gender, "Gender", "FPR")
plot_fairness(fair_race, "Race", "TPR")
plot_fairness(fair_race, "Race", "FPR")
#--- Fimal Checklist
import matplotlib.pyplot as plt
import pandas as pd

# --- Load final checklist
checklist = pd.read_csv("Path//ARC_Checklist__Final.csv")

# --- Build composite figure
fig = plt.figure(figsize=(16,12))

# Panel A: Checklist Table
ax1 = plt.subplot2grid((3,2), (0,0), colspan=2)
ax1.axis("off")
tbl = ax1.table(cellText=checklist.values,
                colLabels=checklist.columns,
                cellLoc="left", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.5)
ax1.set_title("Panel A – ARC Checklist", fontsize=14, fontweight="bold")

# Panel B: Calibration (load saved PNG)
ax2 = plt.subplot2grid((3,2), (1,0))
img_cal = plt.imread("Path//Calibration_Panel.png")
ax2.imshow(img_cal)
ax2.axis("off")
ax2.set_title("Panel B – Calibration (raw vs scaled)", fontsize=12)

# Panel C: Fairness (TPR/FPR by Gender & Race) – load from earlier saved barplots if available
# For simplicity, replot directly if you have fair_gender, fair_race dataframes
def plot_metric(df, metric, title, ax):
    ref_val = df.loc[df["Reference"], metric].values[0]
    colors = ["tab:blue" if r else "tab:orange" for r in df["Reference"]]
    ax.bar(df["group"], df[metric], color=colors)
    ax.axhline(ref_val, color="gray", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(metric)

ax3 = plt.subplot2grid((3,2), (1,1))
plot_metric(fair_gender, "TPR", "Gender TPR", ax3)

ax4 = plt.subplot2grid((3,2), (2,1))
plot_metric(fair_race, "TPR", "Race TPR", ax4)

ax5 = plt.subplot2grid((3,2), (2,0))
plot_metric(fair_gender, "FPR", "Gender FPR", ax5)

# Adjust layout
plt.tight_layout()
save_path = "Path//ARC_Audit_Report.png"
plt.savefig(save_path, dpi=200)
plt.show()

print(f"Saved combined ARC audit report: {save_path}")
