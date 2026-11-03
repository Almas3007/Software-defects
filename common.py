"""Shared data loading, cleaning, selectors, models and metrics for the revised JM1 study."""
import os, re, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["jm1", "kc1", "pc1", "kc2"]
FEATURES = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t", "lOCode", "lOComment",
            "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"]
SEED = 42


def read_arff(name):
    """Plain ARFF reader; '?' becomes NaN and the last column is the label."""
    cols, rows, in_data = [], [], False
    with open(os.path.join(HERE, "data", f"{name}.arff"), errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if in_data:
                rows.append([v.strip().strip("'\"") for v in s.split(",")])
            elif s.lower().startswith("@attribute"):
                cols.append(re.split(r"\s+", s, maxsplit=2)[1].strip("'\""))
            elif s.lower().startswith("@data"):
                in_data = True
    df = pd.DataFrame(rows, columns=cols)
    label = cols[-1]
    y = df[label].str.lower().map({"true": 1, "yes": 1, "1": 1, "false": 0, "no": 0, "0": 0}).astype(int)
    X = df[cols[:-1]].apply(pd.to_numeric, errors="coerce")
    X.columns = [FEATURES[[f.lower() for f in FEATURES].index(c.lower())] if c.lower() in [f.lower() for f in FEATURES]
                 else {"locodeandcomment": "locCodeAndComment", "loccodeandcomment": "locCodeAndComment"}.get(c.lower(), c)
                 for c in X.columns]
    X = X[FEATURES]
    X["__y"] = y.values
    return X


def clean(df):
    """Remove exact duplicates, then every copy of a metric vector that carries both labels."""
    info = {"raw": len(df), "raw_def": int(df["__y"].sum()), "missing_rows": int(df[FEATURES].isna().any(axis=1).sum())}
    d = df.drop_duplicates()
    info["dup_removed"] = len(df) - len(d)
    key = pd.util.hash_pandas_object(d[FEATURES].fillna(-1.0), index=False)
    conflict = d.groupby(key.values)["__y"].transform("nunique") > 1
    d = d[~conflict.values]
    info["conflict_removed"] = int(conflict.sum())
    info["clean"] = len(d)
    info["clean_def"] = int(d["__y"].sum())
    return d.reset_index(drop=True), info


def load(name, cleaned=True):
    df = read_arff(name)
    if cleaned:
        df, _ = clean(df)
    return df[FEATURES].values.astype(float), df["__y"].values.astype(int)


def prep(Xtr, *others):
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = sc.fit_transform(imp.fit_transform(Xtr))
    return [A] + [sc.transform(imp.transform(o)) for o in others]


# ---------- feature selection (always fitted on training data only) ----------
def rf(seed, n=100):
    return RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=1)


def sel_rfecv(Xtr, ytr, seed):
    Z = prep(Xtr)[0]
    m = RFECV(rf(seed), step=1, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
              scoring="average_precision", n_jobs=1).fit(Z, ytr)
    return m.support_, m.cv_results_["mean_test_score"].tolist()


def sel_mi(Xtr, ytr, seed, k):
    Z = prep(Xtr)[0]
    s = mutual_info_classif(Z, ytr, random_state=seed)
    return top_mask(s, k)


def sel_perm(Xtr, ytr, seed, k):
    a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=seed)
    Za, Zb = prep(a, b)
    m = rf(seed).fit(Za, ya)
    s = permutation_importance(m, Zb, yb, scoring="average_precision", n_repeats=5, random_state=seed).importances_mean
    return top_mask(s, k)


def sel_l1(Xtr, ytr, seed):
    Z = prep(Xtr)[0]
    m = LogisticRegression(penalty="l1", C=0.01, solver="liblinear", class_weight="balanced", random_state=seed).fit(Z, ytr)
    mask = np.abs(m.coef_[0]) > 1e-8
    return mask if mask.any() else top_mask(np.abs(m.coef_[0]), 1)


def sel_corr(Xtr, ytr, thr=0.95):
    d = pd.DataFrame(prep(Xtr)[0])
    rel = d.corrwith(pd.Series(ytr), method="spearman").abs().fillna(0).values
    c = d.corr(method="spearman").abs().fillna(0).values
    keep = []
    for j in np.argsort(-rel):
        if all(c[j, i] < thr for i in keep):
            keep.append(j)
    m = np.zeros(d.shape[1], bool)
    m[keep] = True
    return m


def top_mask(s, k):
    m = np.zeros(len(s), bool)
    m[np.argsort(-np.asarray(s))[:k]] = True
    return m


def nogueira(masks):
    """Stability index of Nogueira et al. (2018) for a binary selection matrix."""
    Z = np.asarray(masks, float)
    M, d = Z.shape
    p = Z.mean(0)
    var = M / (M - 1) * p * (1 - p)
    k = Z.sum(1).mean()
    denom = k / d * (1 - k / d)
    return float(1 - var.mean() / denom) if denom > 0 else 1.0


# ---------- classifiers and tuning grids ----------
CLFS = ["LR", "SVM", "NB", "RF", "GB", "XGB"]
GRID = {
    "LR": [{"C": c} for c in (0.01, 0.1, 1, 10)],
    "SVM": [{"C": c, "gamma": g} for c in (1, 10) for g in ("scale", 0.01)],
    "NB": [{"var_smoothing": v} for v in (1e-9, 1e-6, 1e-3)],
    "RF": [{"max_depth": d, "min_samples_leaf": l} for d in (None, 10) for l in (1, 5)],
    "GB": [{"learning_rate": r, "max_depth": d} for r in (0.05, 0.1) for d in (2, 3)],
    "XGB": [{"learning_rate": r, "max_depth": d} for r in (0.05, 0.1) for d in (3, 6)],
}


def make(name, params, imb, y, seed):
    w = imb == "weight"
    if name == "LR":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if w else None, random_state=seed, **params)
    if name == "SVM":
        return SVC(kernel="rbf", class_weight="balanced" if w else None, random_state=seed, **params)
    if name == "NB":
        return GaussianNB(priors=[.5, .5] if w else None, **params)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced" if w else None, random_state=seed, n_jobs=1, **params)
    if name == "GB":
        return GradientBoostingClassifier(n_estimators=200, random_state=seed, **params)
    if name == "XGB":
        spw = (y == 0).sum() / max((y == 1).sum(), 1) if w else 1.0
        return XGBClassifier(n_estimators=300, scale_pos_weight=spw, eval_metric="logloss", n_jobs=1, random_state=seed, **params)
    raise ValueError(name)


def fit_score(name, params, imb, Xtr, ytr, tests, seed):
    """Fit scaler, optional SMOTE and model on training rows only, then score each test matrix."""
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Z = sc.fit_transform(imp.fit_transform(Xtr))
    y = ytr
    if imb == "smote":
        k = min(5, int(y.sum()) - 1)
        Z, y = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Z, y)
    m = make(name, params, imb, y, seed)
    if name == "GB" and imb == "weight":
        m.fit(Z, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        m.fit(Z, y)
    out = []
    for T in tests:
        Q = sc.transform(imp.transform(T))
        out.append(m.decision_function(Q) if hasattr(m, "decision_function") and name != "GB" else m.predict_proba(Q)[:, 1])
    return out, m


def default_thr(name):
    return 0.0 if name in ("LR", "SVM") else 0.5


def best_f1_thr(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f))])


def min_cost_thr(y, s, cfn=5.0):
    o = np.argsort(-s)
    ys = y[o]
    fp, tp = np.cumsum(1 - ys), np.cumsum(ys)
    cost = cfn * (y.sum() - tp) + fp
    return float(s[o][int(np.argmin(cost))])


def tune(name, imb, Xtr, ytr, seed):
    """Inner 3-fold search on PR-AUC; thresholds come from the out-of-fold scores of the winner."""
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr))
    best = (-1, None, None)
    for params in GRID[name]:
        oof = np.zeros(len(ytr))
        for a, b in inner:
            oof[b] = fit_score(name, params, imb, Xtr[a], ytr[a], [Xtr[b]], seed)[0][0]
        ap = average_precision_score(ytr, oof)
        if ap > best[0]:
            best = (ap, params, oof)
    return best[1], best_f1_thr(ytr, best[2]), min_cost_thr(ytr, best[2])


def effort_recall(y, s, loc, frac=0.2):
    """Share of defective modules found when inspecting the top-ranked modules up to 20% of total LOC."""
    o = np.argsort(-s)
    cum = np.cumsum(np.nan_to_num(loc[o]))
    n = np.searchsorted(cum, frac * cum[-1], side="right")
    return float(y[o][:n].sum() / max(y.sum(), 1))


def metrics(y, s, thr, loc=None):
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    d = dict(acc=accuracy_score(y, p), bacc=balanced_accuracy_score(y, p),
             prec=precision_score(y, p, zero_division=0), rec=recall_score(y, p, zero_division=0),
             f1=f1_score(y, p, zero_division=0), mcc=matthews_corrcoef(y, p),
             tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), cost5=float(5 * fn + fp) / len(y))
    if len(np.unique(s)) > 1:
        d.update(roc=roc_auc_score(y, s), pr=average_precision_score(y, s))
    else:
        d.update(roc=0.5, pr=float(y.mean()))
    if loc is not None:
        d["effort20"] = effort_recall(y, s, loc)
    return d
