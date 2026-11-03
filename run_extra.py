"""Holdout table, leakage and duplicate audit, cross-project runs, timing and memory. Run after run_cv.py."""
import json, os, pickle, sys, time, tracemalloc
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import common as c

RES = os.path.join(c.HERE, "results")
LOC = c.FEATURES.index("loc")


def holdout_one(m, X, y, tr, te, M):
    params, tf1, tcost = c.tune(m, "weight", X[tr][:, M], y[tr], c.SEED)
    (s,), _ = c.fit_score(m, params, "weight", X[tr][:, M], y[tr], [X[te][:, M]], c.SEED)
    rows = []
    rng = np.random.default_rng(c.SEED)
    for mode, thr in (("default", c.default_thr(m)), ("f1", tf1)):
        d = c.metrics(y[te], s, thr, X[te][:, LOC])
        boots = {"f1": [], "pr": [], "mcc": []}
        for _ in range(1000):
            i = rng.integers(0, len(te), len(te))
            if y[te][i].sum() == 0:
                continue
            b = c.metrics(y[te][i], s[i], thr)
            for k in boots:
                boots[k].append(b[k])
        for k, v in boots.items():
            d[k + "_lo"], d[k + "_hi"] = np.percentile(v, [2.5, 97.5])
        rows.append({"clf": m, "thr": mode, "params": str(params), "threshold": thr, **d})
    return rows, s.tolist()


def holdout(jobs):
    X, y = c.load("jm1")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=c.SEED)
    M, curve = c.sel_rfecv(X[tr], y[tr], c.SEED)
    out = Parallel(n_jobs=jobs)(delayed(holdout_one)(m, X, y, tr, te, M) for m in c.CLFS)
    json.dump(dict(rows=[r for o, _ in out for r in o], scores={m: o[1] for m, o in zip(c.CLFS, out)}, y=y[te].tolist(),
                   mask=M.astype(int).tolist(), curve=curve, n_train=len(tr), n_test=len(te)),
              open(os.path.join(RES, "holdout.json"), "w"), default=float)


def audit_fold(kind, X, y, tr, te, M_global, idx):
    """Compare global selection on raw data, per-fold selection on raw data and per-fold selection on cleaned data."""
    seed = c.SEED + idx
    M = M_global if kind == "raw_global" else c.sel_rfecv(X[tr], y[tr], seed)[0]
    rows = []
    for m, p in (("LR", {"C": 1}), ("RF", {"max_depth": None, "min_samples_leaf": 1}), ("XGB", {"learning_rate": 0.1, "max_depth": 6})):
        (s,), _ = c.fit_score(m, p, "weight", X[tr][:, M], y[tr], [X[te][:, M]], seed)
        rows.append(dict(kind=kind, clf=m, idx=idx, **c.metrics(y[te], s, c.default_thr(m))))
    return rows


def audit(jobs):
    tasks = []
    for kind, cleaned in (("raw_global", False), ("raw_nested", False), ("clean_nested", True)):
        X, y = c.load("jm1", cleaned=cleaned)
        Mg = c.sel_rfecv(X, y, c.SEED)[0] if kind == "raw_global" else None
        rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=c.SEED)
        tasks += [(kind, X, y, tr, te, Mg, i) for i, (tr, te) in enumerate(rs.split(X, y))]
    rows = Parallel(n_jobs=jobs)(delayed(audit_fold)(*t) for t in tasks)
    json.dump([r for g in rows for r in g], open(os.path.join(RES, "audit.json"), "w"), default=float)


def cross_one(src, m):
    Xs, ys = c.load(src)
    M, _ = c.sel_rfecv(Xs, ys, c.SEED)
    params, tf1, _ = c.tune(m, "weight", Xs[:, M], ys, c.SEED)
    tests = {t: c.load(t) for t in c.DATASETS if t != src}
    sc, _ = c.fit_score(m, params, "weight", Xs[:, M], ys, [v[0][:, M] for v in tests.values()], c.SEED)
    return [dict(src=src, tgt=t, clf=m, nfeat=int(M.sum()), **c.metrics(v[1], s, tf1, v[0][:, LOC])) for (t, v), s in zip(tests.items(), sc)]


def cross(jobs):
    out = Parallel(n_jobs=jobs)(delayed(cross_one)(s, m) for s in c.DATASETS for m in ("LR", "RF", "XGB"))
    json.dump([r for g in out for r in g], open(os.path.join(RES, "cross.json"), "w"), default=float)


def timing():
    """Single-process wall time and peak Python-tracked memory on JM1, 80/20 split, five runs."""
    X, y = c.load("jm1")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=c.SEED)
    t0 = time.perf_counter(); M, _ = c.sel_rfecv(X[tr], y[tr], c.SEED); sel_secs = time.perf_counter() - t0
    fixed = {"LR": {"C": 1}, "SVM": {"C": 1, "gamma": "scale"}, "NB": {"var_smoothing": 1e-9},
             "RF": {"max_depth": None, "min_samples_leaf": 1}, "GB": {"learning_rate": 0.1, "max_depth": 3},
             "XGB": {"learning_rate": 0.1, "max_depth": 6}}
    rows = []
    for m in c.CLFS:
        for name, mask in (("All", np.ones(X.shape[1], bool)), ("RFECV", M)):
            fits, preds, peaks, sizes = [], [], [], []
            for r in range(5):
                t = time.perf_counter()
                _, model = c.fit_score(m, fixed[m], "weight", X[tr][:, mask], y[tr], [], c.SEED)
                fits.append(time.perf_counter() - t)
                Q = c.prep(X[tr][:, mask], X[te][:, mask])[1]
                t = time.perf_counter()
                model.predict(Q)
                preds.append(time.perf_counter() - t)
                sizes.append(len(pickle.dumps(model)))
            tracemalloc.start()
            c.fit_score(m, fixed[m], "weight", X[tr][:, mask], y[tr], [], c.SEED)
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            rows.append(dict(clf=m, sel=name, nfeat=int(mask.sum()), fit_ms=1e3 * np.median(fits),
                             pred_ms=1e3 * np.median(preds), peak_mb=peaks[0] / 2**20, model_kb=np.median(sizes) / 1024))
    json.dump(dict(selection_secs=sel_secs, rows=rows), open(os.path.join(RES, "timing.json"), "w"), default=float)


if __name__ == "__main__":
    jobs = 7
    for step, fn in (("audit", lambda: audit(jobs)), ("holdout", lambda: holdout(jobs)), ("cross", lambda: cross(jobs)), ("timing", timing)):
        t = time.time(); fn(); print(step, "done", round(time.time() - t), flush=True)
