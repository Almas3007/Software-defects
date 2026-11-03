"""Repeat the authors' first RFECV setting (200 trees, 5-fold, F1, raw JM1) under different seeds."""
import json, os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
import common as c

def one(seed):
    X, y = c.load("jm1", cleaned=False)
    a, _, ya, _ = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    m = RFECV(c.rf(seed, 200), step=1, cv=5, scoring="f1").fit(c.prep(a)[0], ya)
    return dict(seed=seed, n=int(m.n_features_), mask=m.support_.astype(int).tolist(), curve=m.cv_results_["mean_test_score"].tolist())

r = Parallel(n_jobs=7)(delayed(one)(s) for s in [42, 1, 2, 3, 4, 5, 6])
json.dump(r, open(os.path.join(c.HERE, "results", "rfecv_repro.json"), "w"))
print([x["n"] for x in r], "stability", c.nogueira([x["mask"] for x in r]))
for x in r: print(x["seed"], x["n"], round(max(x["curve"]), 4), round(x["curve"][12], 4), round(x["curve"][20], 4))
