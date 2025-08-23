# requires: scikit-learn>=1.1, pandas, numpy
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
n = 120
df = pd.DataFrame({
    "t": np.arange(n),
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
})
df["y"] = (rng.random(n) < 1/(1+np.exp(-(0.6*df.x1-0.4*df.x2)))).astype(int)
X, y = df[["x1","x2"]].values, df["y"].values

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
])

# TimeSeriesSplit example (no leakage across time)
tscv = TimeSeriesSplit(n_splits=5, test_size=12)
aucs = []
for tr, te in tscv.split(X):
    pipe.fit(X[tr], y[tr])
    p = pipe.predict_proba(X[te])[:,1]
    aucs.append(roc_auc_score(y[te], p))
print("TS AUC mean:", np.mean(aucs))

# GroupKFold example (e.g., user_id) – groups must not spill over folds
groups = np.repeat(np.arange(24), 5)[:n]
gkf = GroupKFold(n_splits=5)
aucs = []
for tr, te in gkf.split(X, y, groups=groups):
    pipe.fit(X[tr], y[tr])
    p = pipe.predict_proba(X[te])[:,1]
    aucs.append(roc_auc_score(y[te], p))
print("GroupKFold AUC mean:", np.mean(aucs))
