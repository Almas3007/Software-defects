"""Aggregate fold results into CSV summaries and print the numbers used in the paper."""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats
import common as c
RES = os.path.join(c.HERE, "results")
pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 40)
rows, masks, curves, sel = [], {}, {}, []
for f in sorted(glob.glob(os.path.join(RES, "folds", "*.json"))):
    j = json.load(open(f)); ds = os.path.basename(f).split("_")[0]
    rows += j["rows"]
    for k, v in j["masks"].items(): masks.setdefault((ds, k), []).append(v)
    curves.setdefault(ds, []).append(j["curve"])
    sel.append(dict(ds=ds, **j["sel_secs"]))
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "fold_metrics.csv"), index=False)
M = ["acc", "bacc", "prec", "rec", "f1", "mcc", "roc", "pr", "effort20", "cost5"]
def tab(q): return q.groupby(["ds", "sel", "clf", "imb", "thr"])[M].agg(["mean", "std"])

print("=== JM1 RFECV weight: f1 thr + default"); 
for th in ("default", "f1", "cost"):
    q = R[(R.ds == "jm1") & (R.sel == "RFECV") & (R.imb == "weight") & (R.thr == th)]
    print(th); print((q.groupby("clf")[M].mean() * 100).round(2)); print((q.groupby("clf")[["f1","mcc","pr"]].std() * 100).round(2))
print("=== baselines")
print((R[R.clf.isin(["Majority", "LOC-rank"])].groupby(["ds", "clf"])[M].mean() * 100).round(2))
print("=== all datasets, RFECV weight f1"); q = R[(R.sel == "RFECV") & (R.imb == "weight") & (R.thr == "f1")]
for mt in ("f1", "mcc", "pr", "roc", "rec", "prec", "effort20"):
    print(mt); print((q.pivot_table(index="clf", columns="ds", values=mt) * 100).round(2)[c.DATASETS])

print("=== All vs RFECV (weight, f1): mean diff, paired tests, per dataset")
out = []
for ds in c.DATASETS:
    for m in c.CLFS:
        a = R[(R.ds == ds) & (R.sel == "All") & (R.clf == m) & (R.imb == "weight") & (R.thr == "f1")].sort_values("idx")
        b = R[(R.ds == ds) & (R.sel == "RFECV") & (R.clf == m) & (R.imb == "weight") & (R.thr == "f1")].sort_values("idx")
        for mt in ("pr", "mcc", "f1"):
            d = b[mt].values - a[mt].values
            J = len(d); rho = 1 / 5  # test/train ratio for 5-fold, Nadeau-Bengio correction
            se = np.sqrt((1 / J + rho / (1 - rho)) * d.var(ddof=1)) if d.std() > 0 else np.nan
            t = d.mean() / se if se and se == se else 0
            pc = 2 * stats.t.sf(abs(t), J - 1)
            out.append(dict(ds=ds, clf=m, metric=mt, all=a[mt].mean(), rfecv=b[mt].mean(), diff=d.mean(), p_corrected=pc))
AV = pd.DataFrame(out); AV.to_csv(os.path.join(RES, "all_vs_rfecv.csv"), index=False)
print(AV[AV.metric == "pr"].round(4)); print(AV[AV.metric == "mcc"].round(4))
print("n significant p<.05:", AV.groupby("metric").p_corrected.apply(lambda s: (s < .05).sum()), "of", AV.groupby("metric").size().iloc[0])
print("nfeat RFECV per ds:"); print(R[(R.sel.isin(["RFECV", "MI", "Perm", "L1", "Corr"]))&(R.thr=="f1")].groupby(["ds", "sel"]).nfeat.agg(["mean", "min", "max"]).round(2))

print("=== selector ablation (LR,RF,XGB; weight; f1 thr)")
q = R[R.clf.isin(["LR", "RF", "XGB"]) & (R.imb == "weight") & (R.thr == "f1") & R.sel.isin(["All", "RFECV", "MI", "Perm", "L1", "Corr"])]
for mt in ("pr", "mcc", "f1"):
    print(mt); print((q.pivot_table(index=["ds", "clf"], columns="sel", values=mt) * 100).round(2)[["All", "RFECV", "MI", "Perm", "L1", "Corr"]])
print("=== stability")
st = pd.DataFrame([dict(ds=d, sel=s, stab=c.nogueira(v), meank=np.mean(np.sum(v, 1))) for (d, s), v in masks.items() if s != "All"])
print(st.pivot(index="ds", columns="sel", values="stab").round(3)); st.to_csv(os.path.join(RES, "stability.csv"), index=False)
freq = pd.DataFrame({d: np.mean(masks[(d, "RFECV")], 0) for d in c.DATASETS}, index=c.FEATURES); print(freq.round(2)); freq.to_csv(os.path.join(RES, "rfecv_freq.csv"))
print("JM1 RFECV sizes", np.sum(masks[("jm1", "RFECV")], 1).tolist())
print("sel secs"); print(pd.DataFrame(sel).groupby("ds").mean().round(2))

print("=== imbalance JM1+all, RFECV")
q = R[(R.sel == "RFECV")]
for mt in ("f1", "mcc", "pr", "rec", "prec"):
    print(mt); print((q[q.ds == "jm1"].pivot_table(index="clf", columns=["imb", "thr"], values=mt) * 100).round(2))
print("mean across ds/clf"); print((q.pivot_table(index=["imb", "thr"], columns="ds", values=["f1", "mcc", "pr"]) * 100).round(2))

print("=== Friedman + Holm Wilcoxon vs best, RFECV/weight/f1 on MCC and PR, per dataset")
fr = []
for ds in c.DATASETS:
    for mt in ("mcc", "pr"):
        p = R[(R.ds == ds) & (R.sel == "RFECV") & (R.imb == "weight") & (R.thr == "f1")].pivot(index="idx", columns="clf", values=mt)[c.CLFS]
        chi, pf = stats.friedmanchisquare(*[p[k] for k in c.CLFS])
        ranks = p.rank(axis=1, ascending=False).mean().round(2)
        best = p.mean().idxmax()
        ps = {k: stats.wilcoxon(p[best], p[k]).pvalue for k in c.CLFS if k != best}
        items = sorted(ps.items(), key=lambda x: x[1]); adj = {}; run = 0
        for i, (k, v) in enumerate(items):
            run = max(run, min(1, v * (len(items) - i))); adj[k] = round(run, 4)
        print(ds, mt, "chi2", round(chi, 2), "p", f"{pf:.2e}", "best", best, "ranks", ranks.to_dict(), "holm", adj)
        fr.append(dict(ds=ds, metric=mt, chi2=chi, p=pf, best=best, ranks=ranks.to_dict(), holm=adj))
json.dump(fr, open(os.path.join(RES, "friedman.json"), "w"), default=float)

print("=== audit"); A = pd.DataFrame(json.load(open(os.path.join(RES, "audit.json"))))
print((A.groupby(["clf", "kind"])[["acc", "prec", "rec", "f1", "mcc", "roc", "pr"]].mean() * 100).round(2))
print("=== holdout"); H = json.load(open(os.path.join(RES, "holdout.json")))
print(pd.DataFrame(H["rows"]).drop(columns=["params", "cost5"]).round(4).to_string()); print("mask", [f for f, m in zip(c.FEATURES, H["mask"]) if m], H["n_train"], H["n_test"])
print(pd.DataFrame(H["rows"])[["clf", "thr", "params", "threshold"]].to_string())
print("=== cross"); C = pd.DataFrame(json.load(open(os.path.join(RES, "cross.json"))))
print((C.pivot_table(index=["src", "tgt"], columns="clf", values="mcc") * 100).round(1)); print((C.pivot_table(index=["src", "tgt"], columns="clf", values="pr") * 100).round(1)); print((C.pivot_table(index=["src", "tgt"], columns="clf", values="roc") * 100).round(1))
print("=== timing"); T = json.load(open(os.path.join(RES, "timing.json"))); print(T["selection_secs"]); print(pd.DataFrame(T["rows"]).round(2).to_string())
print("params chosen jm1 RFECV weight"); print(R[(R.ds=="jm1")&(R.sel=="RFECV")&(R.imb=="weight")&(R.thr=="f1")].groupby("clf").params.agg(lambda s: s.value_counts().index[0]))
