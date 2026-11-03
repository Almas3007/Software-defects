"""Nested, repeated stratified CV. One JSON file per outer fold so an interrupted run can resume."""
import json, os, sys, time
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
import common as c

OUT = os.path.join(c.HERE, "results", "folds")
os.makedirs(OUT, exist_ok=True)
REPEATS, FOLDS = 3, 5
LOC = c.FEATURES.index("loc")


def configs():
    cfg = [("All", m, "weight") for m in c.CLFS] + [("RFECV", m, "weight") for m in c.CLFS]
    cfg += [("RFECV", m, i) for m in c.CLFS for i in ("none", "smote")]
    cfg += [(s, m, "weight") for s in ("MI", "Perm", "L1", "Corr") for m in ("LR", "RF", "XGB")]
    return cfg


def run_fold(ds, idx, tr, te):
    path = os.path.join(OUT, f"{ds}_{idx:02d}.json")
    if os.path.exists(path):
        return
    X, y = c.load(ds)
    seed = c.SEED + idx
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    masks, times = {}, {}
    t = time.time(); masks["RFECV"], curve = c.sel_rfecv(Xtr, ytr, seed); times["RFECV"] = time.time() - t
    k = int(masks["RFECV"].sum())
    t = time.time(); masks["MI"] = c.sel_mi(Xtr, ytr, seed, k); times["MI"] = time.time() - t
    t = time.time(); masks["Perm"] = c.sel_perm(Xtr, ytr, seed, k); times["Perm"] = time.time() - t
    t = time.time(); masks["L1"] = c.sel_l1(Xtr, ytr, seed); times["L1"] = time.time() - t
    t = time.time(); masks["Corr"] = c.sel_corr(Xtr, ytr); times["Corr"] = time.time() - t
    masks["All"] = np.ones(X.shape[1], bool)
    rows = []
    for sel, m, imb in configs():
        M = masks[sel]
        params, thr_f1, thr_cost = c.tune(m, imb, Xtr[:, M], ytr, seed)
        t = time.time()
        (s,), _ = c.fit_score(m, params, imb, Xtr[:, M], ytr, [Xte[:, M]], seed)
        fit_pred = time.time() - t
        base = dict(ds=ds, idx=idx, sel=sel, clf=m, imb=imb, params=str(params), secs=fit_pred, nfeat=int(M.sum()))
        for mode, thr in (("default", c.default_thr(m)), ("f1", thr_f1), ("cost", thr_cost)):
            rows.append({**base, "thr": mode, **c.metrics(yte, s, thr, Xte[:, LOC])})
    for name, s in (("Majority", np.zeros(len(yte))), ("LOC-rank", np.nan_to_num(Xte[:, LOC]))):
        thr = 0.5 if name == "Majority" else c.best_f1_thr(ytr, np.nan_to_num(Xtr[:, LOC]))
        rows.append(dict(ds=ds, idx=idx, sel="-", clf=name, imb="-", thr="f1", params="", secs=0, nfeat=0 if name == "Majority" else 1, **c.metrics(yte, s, thr, Xte[:, LOC])))
    with open(path + ".tmp", "w") as fh:
        json.dump(dict(rows=rows, masks={k_: v.astype(int).tolist() for k_, v in masks.items()}, curve=curve, sel_secs=times), fh)
    os.replace(path + ".tmp", path)
    print(ds, idx, "done", flush=True)


if __name__ == "__main__":
    tasks = []
    for ds in reversed(c.DATASETS):  # small datasets first, so failures surface early
        X, y = c.load(ds)
        rs = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=c.SEED)
        tasks += [(ds, i, tr, te) for i, (tr, te) in enumerate(rs.split(X, y))]
    Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 7, batch_size=1)(delayed(run_fold)(*t) for t in tasks)
