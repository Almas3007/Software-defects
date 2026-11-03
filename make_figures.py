"""Figures and a few extra paired tests quoted in the paper."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.metrics import precision_recall_curve
import common as c
RES = os.path.join(c.HERE, "results"); FIG = os.path.join(c.HERE, "..", "figures")
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
R = pd.read_csv(os.path.join(RES, "fold_metrics.csv"))

def nb(a, b):
    d = a - b; J = len(d); se = np.sqrt((1 / J + .25) * d.var(ddof=1)); return d.mean(), 2 * stats.t.sf(abs(d.mean() / se), J - 1)
def get(ds, sel, clf, imb="weight", thr="f1", m="mcc"):
    return R[(R.ds == ds) & (R.sel == sel) & (R.clf == clf) & (R.imb == imb) & (R.thr == thr)].sort_values("idx")[m].values
print("Corr vs All, per ds/clf: (metric: diff,p)")
for ds in c.DATASETS:
    for k in ("LR", "RF", "XGB"):
        print(ds, k, {m: tuple(np.round(nb(get(ds, "Corr", k, m=m), get(ds, "All", k, m=m)), 4)) for m in ("mcc", "pr")})
print("RF vs LOC-rank")
for ds in c.DATASETS:
    print(ds, {m: tuple(np.round(nb(get(ds, "RFECV", "RF", m=m), get(ds, "-", "LOC-rank", imb="-", m=m)), 4)) for m in ("mcc", "pr", "f1", "roc")})
print("F1 threshold vs default on unweighted RF jm1", nb(get("jm1", "RFECV", "RF", "none", "f1", "f1"), get("jm1", "RFECV", "RF", "none", "default", "f1")))
print("mean corr size stability; RF fold confusion jm1:", R[(R.ds == "jm1") & (R.sel == "RFECV") & (R.clf == "RF") & (R.imb == "weight") & (R.thr == "f1")][["tn", "fp", "fn", "tp"]].sum().div(3).round(0).to_dict())

js = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "folds", "jm1_*.json")))]
C = 100 * np.array([j["curve"] for j in js]); sizes = [sum(j["masks"]["RFECV"]) for j in js]
print("curve mean at 1,3,5,10,13,17,21:", {k: round(C.mean(0)[k - 1], 2) for k in (1, 3, 5, 10, 13, 17, 21)}, "chosen", min(sizes), max(sizes))
fig, ax = plt.subplots(figsize=(3.5, 2.05)); x = np.arange(1, 22)
ax.axvspan(min(sizes) - .5, max(sizes) + .5, color="#f4d9b8", alpha=.65, lw=0)
ax.fill_between(x, C.mean(0) - C.std(0), C.mean(0) + C.std(0), color="#1f4e79", alpha=.2, lw=0)
ax.plot(x, C.mean(0), color="#1f4e79", lw=1.4, marker="o", ms=3)
ax.text((min(sizes) + max(sizes)) / 2, C.mean(0).min() + .6, f"Sizes chosen\n({min(sizes)}–{max(sizes)})", ha="center", va="bottom", fontsize=8, color="#7a4510")
ax.set_xlabel("Number of retained metrics"); ax.set_ylabel("Inner-CV PR-AUC (%)"); ax.set_xticks([1, 3, 5, 9, 13, 17, 21]); ax.set_xlim(.4, 21.6); ax.grid(axis="y", alpha=.3)
fig.savefig(os.path.join(FIG, "rfecv_curve.pdf")); plt.close(fig)

F = pd.read_csv(os.path.join(RES, "rfecv_freq.csv"), index_col=0)[c.DATASETS]; F = F.loc[F.mean(1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(3.5, 3.75)); ax.imshow(F.values, cmap="Blues", vmin=0, vmax=1.25, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([d.upper() for d in c.DATASETS]); ax.xaxis.tick_top(); ax.set_yticks(range(len(F))); ax.set_yticklabels(F.index, fontsize=8)
ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(F)), minor=True); ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="both", length=0)
for i in range(F.shape[0]):
    for j in range(4):
        v = F.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > .7 else "black")
for sp in ax.spines.values(): sp.set_visible(False)
fig.savefig(os.path.join(FIG, "selection_freq.pdf")); plt.close(fig)

H = json.load(open(os.path.join(RES, "holdout.json"))); y = np.array(H["y"])
fig, ax = plt.subplots(figsize=(3.4, 1.8)); cols = dict(LR="#1f4e79", SVM="#c55a11", NB="#7f7f7f", RF="#2e7d32", GB="#8e44ad", XGB="#b8860b")
pr = {r["clf"]: r["pr"] for r in H["rows"]}
for k in c.CLFS:
    p, r, _ = precision_recall_curve(y, H["scores"][k]); ax.plot(r, p, lw=1, color=cols[k], label=f"{k} ({pr[k]:.3f})")
ax.axhline(y.mean(), color="k", ls="--", lw=.8, label=f"Prevalence ({y.mean():.3f})"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6.3, ncol=2); ax.grid(alpha=.25)
fig.savefig(os.path.join(FIG, "pr_curves.pdf")); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 4.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, t, shade=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#dbe8f4" if shade else "#f4f4f4", ec="#1f4e79" if shade else "#666666", lw=.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.15, zorder=3)
def arrow(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=.8, color="#333333", shrinkA=0, shrinkB=0), zorder=4)
box(14, 90.5, 72, 9, "NASA MDP tables: JM1, KC1, PC1, KC2")
box(14, 77, 72, 9, "Cleaning: remove duplicates and\ncontradictory records")
arrow((50, 90.5), (50, 86))
ax.add_patch(FancyBboxPatch((1, 15), 98, 57.5, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec="#444444", lw=.8, ls="--"))
arrow((50, 77), (50, 72.5))
ax.text(50, 69, "Outer loop: stratified 5-fold, repeated 3 times", ha="center", va="center", fontsize=8, style="italic")
ax.text(32, 64.6, "Training fold", ha="center", va="center", fontsize=8, weight="bold"); ax.text(82, 33, "Test fold", ha="center", va="center", fontsize=8, weight="bold")
ys = [(51.5, "Select metrics: RFECV or\ncomparator"), (35.5, "Inner 3-fold: tune on PR-AUC,\nimbalance option, threshold"), (19, "Refit imputer, scaler and\nmodel on the training fold")]
for y, t in ys: box(4, y, 56, 11, t, shade=True)
arrow((32, 51.5), (32, 46.5)); arrow((32, 35.5), (32, 30))
box(68, 19, 28, 11, "Score once,\nuntouched")
arrow((60, 24.5), (68, 24.5))
box(14, 1, 72, 9, "MCC, PR-AUC, F1, E20, cost, stability,\npaired tests")
arrow((82, 19), (82, 12.5)); ax.plot([82, 50], [12.5, 12.5], color="#333333", lw=.8, zorder=4); arrow((50, 12.5), (50, 10))
fig.savefig(os.path.join(FIG, "framework.pdf")); plt.close(fig)
