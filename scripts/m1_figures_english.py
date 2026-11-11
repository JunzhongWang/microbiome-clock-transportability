#!/usr/bin/env python3
"""English versions of M1 main figures (submission-ready, Microbiome/BMC).

Reads the same result CSVs as the Chinese versions and regenerates:
  en_fig1_age_stratified.png   (Fig. 1 in the manuscript)
  en_fig2_bias_and_gate.png    (Fig. 3)
  en_fig3_rage_distribution.png(Fig. 4)
  en_fig4_deviation.png        (Fig. 5)
The multi-clock composite (fig2_multiclock) and figS1 are already in English.
"""
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/metaphlan")
OUT = os.path.join(ROOT, "results/m1_metaphlan")
SEED = 42
N_PC = 30
MIN_PREVALENCE = 0.10
PSEUDO = 1e-6

# ================= Fig 1 : age-stratified performance =================
df = pd.read_csv(os.path.join(OUT, "age_stratified_clock.csv"))
tr = pd.read_csv(os.path.join(OUT, "clock_transportability.csv"))

TAG_EN = {
    "全体 0-107": "All ages 0–107",
    "成人 18+": "Adults 18+",
    "青年 18-40": "Young adults 18–40",
    "中年 40-70": "Middle-aged 40–70",
    "老年 60+": "Older adults 60+",
    "儿童 <18": "Children <18",
}
cv = df[(df["mode"] == "A_loose") & df.gbdt_r2.notna()].copy()
order = list(TAG_EN.keys())
cv["tag"] = pd.Categorical(cv["tag"], categories=order, ordered=True)
cv = cv.sort_values("tag")

fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=300)
y = np.arange(len(cv))[::-1]
ax.barh(y + 0.2, cv.ridge_r2, height=0.36, color="#378ADD", label="Ridge")
ax.barh(y - 0.2, cv.gbdt_r2, height=0.36, color="#1D9E75", label="GBDT")
ax.set_yticks(y)
ax.set_yticklabels(
    [f"{TAG_EN[t]}\n n={n:,} · age SD={s:.1f} y" for t, n, s in zip(cv.tag, cv.n, cv.age_sd)],
    fontsize=9)
ax.axvline(0, color="#888780", lw=0.8)
ax.axvline(0.444, color="#A32D2D", ls="--", lw=1.2)
ax.text(0.452, len(cv) - 1.45, "Published clock\n(R² = 0.444)", fontsize=8, color="#A32D2D", va="top")
for i, r2 in enumerate(cv.ridge_r2):
    ax.text(max(r2, 0) + 0.006, y[i] + 0.2, f"{r2:.3f}", va="center", fontsize=8, color="#378ADD")
ax.set_xlabel("Cross-validated $R^2$ (chronological age predicted from gut microbiome)", fontsize=11)
ax.set_xlim(-0.02, 0.62)
ax.set_title("Accuracy of gut-microbiome 'ageing clocks' is driven by early life\n"
             "Within the adult band relevant to chronic liver disease (40–70 y), R² is only 0.10–0.15",
             fontsize=12, pad=12)
ax.legend(fontsize=9, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#ececec", lw=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "en_fig1_age_stratified.png"), bbox_inches="tight")
plt.close()
print("en_fig1 ok")

# ================= Fig 3 (old fig2) : bias vs effect size =================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300, gridspec_kw={"width_ratios": [1.15, 1]})

ax = axes[0]
b = tr.bias.dropna()
ax.hist(b, bins=24, color="#85B7EB", edgecolor="white")
ax.axvline(0, color="#444", lw=1)
for v, c in [(3, "#1D9E75"), (7, "#A32D2D")]:
    ax.axvline(v, color=c, ls="--", lw=1.4)
    ax.axvline(-v, color=c, ls="--", lw=1.4)
ax.text(7.4, ax.get_ylim()[1] * 0.92, "±7 y\n(upper bound of\npublished mBAG effects)",
        fontsize=7.5, color="#A32D2D")
ax.text(-12.5, ax.get_ylim()[1] * 0.92, "±3 y\n(lower bound)", fontsize=7.5, color="#1D9E75")
ax.set_xlabel("Projection bias (predicted − chronological age, years) onto a single study", fontsize=10)
ax.set_ylabel("No. of studies", fontsize=10)
ax.set_title(f"Projection bias across 54 healthy studies\nSD = {b.std():.1f} y — 2.3× the published effect size",
             fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
scan = []
for th in np.arange(0.0, 0.61, 0.05):
    n = (tr.r_age >= th).sum()
    scan.append((th, n, n / len(tr) * 100))
scan = pd.DataFrame(scan, columns=["thr", "n_pass", "pct"])
ax.plot(scan.thr, scan.pct, "o-", color="#534AB7", lw=2, ms=5)
for _, r in scan.iterrows():
    if r.thr in (0.2, 0.3, 0.4):
        ax.annotate(f"r≥{r.thr:.1f}\n{r.pct:.0f}%",
                    (r.thr, r.pct), textcoords="offset points", xytext=(6, -14),
                    fontsize=8, color="#534AB7")
ax.axvline(0.3, color="#888780", ls=":", lw=1)
ax.set_xlabel("Calibration-gate threshold ρ (within-study $r$ of pred_age ~ age)", fontsize=10)
ax.set_ylabel("Studies passing gate (%)", fontsize=10)
ax.set_title("Threshold scan for the pre-projection calibration gate\nRecommended ρ ≥ 0.3 (only 30% of studies pass)",
             fontsize=11)
ax.set_ylim(-4, 106)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#f0f0f0", lw=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "en_fig2_bias_and_gate.png"), bbox_inches="tight")
plt.close()
print("en_fig2 ok")

# ================= Fig 4 (old fig3) : r_age distribution =================
fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=300)
r = tr.r_age.dropna()
ax.hist(r, bins=20, color="#B5D4F4", edgecolor="white")
ax.axvline(0.3, color="#A32D2D", ls="--", lw=1.4, label="Calibration gate ρ = 0.3")
ax.axvline(0.036, color="#A32D2D", lw=2.2,
           label="QinN_2014 (only liver-disease cohort with ages), r = 0.036")
ax.set_xlabel("Within-study correlation $r$ (pred_age ~ age), 54 healthy studies", fontsize=10)
ax.set_ylabel("No. of studies", fontsize=10)
ax.set_title("Most studies fail to track age after clock projection", fontsize=12)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "en_fig3_rage_distribution.png"), bbox_inches="tight")
plt.close()
print("en_fig3 ok")

# ================= Fig 5 (old fig4) : healthy-manifold deviation =================
# Recompute D with the identical pipeline as m1_deviation_metric.py
meta = pd.read_csv(os.path.join(DATA, "sample_metadata.tsv.gz"), sep="\t",
                   index_col=0, compression="infer")
meta.index = meta.index.astype(str)
frames = []
for p in sorted(glob.glob(os.path.join(DATA, "per_study", "*.tsv.gz"))):
    d0 = pd.read_csv(p, sep="\t", index_col=0)
    d0.index = d0.index.astype(str)
    d0.columns = [c.strip() for c in d0.columns]
    frames.append(d0.apply(pd.to_numeric, errors="coerce").fillna(0.0))
X = pd.concat(frames, axis=0, sort=False)
common = X.index.intersection(meta.index)
X, meta = X.loc[common], meta.loc[common]
X = X.fillna(0.0)
X = X.loc[:, X.sum(axis=0) > 0]

meta.columns = [c.lower() for c in meta.columns]
lab = meta["study_condition"].astype(str).str.lower()
dis = meta["disease"].astype(str).str.lower()
age = pd.to_numeric(meta["age"], errors="coerce")

healthy_mask = dis.eq("healthy") & age.notna()
Xh_raw = X.loc[healthy_mask]
prev = (Xh_raw > 0).mean(axis=0)
Xh_raw = Xh_raw.loc[:, prev >= MIN_PREVALENCE]
feat = Xh_raw.columns
print(f"[healthy training set] {Xh_raw.shape[0]} x {Xh_raw.shape[1]}")

scaler = StandardScaler()
Xh_log = np.log10(Xh_raw.values.astype(float) + PSEUDO)
Xh_s = scaler.fit_transform(Xh_log)
pca = PCA(n_components=N_PC, random_state=SEED)
PC = pca.fit_transform(Xh_s)
sd_pc = PC.std(axis=0)


def deviation(Xsub_raw):
    Xl = np.log10(Xsub_raw[feat].values.astype(float) + PSEUDO)
    Xs = scaler.transform(Xl)
    pc = pca.transform(Xs)
    return np.sqrt(((pc / sd_pc) ** 2).sum(axis=1))


D_train = deviation(Xh_raw)

res = pd.read_csv(os.path.join(OUT, "deviation_scores.csv"))
per_sample = pd.read_csv(os.path.join(OUT, "deviation_per_sample.csv"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

ax = axes[0]
ax.hist(D_train, bins=40, color="#D3D1C7", alpha=0.9,
        label=f"Healthy training set (n={len(D_train):,})")
for _, rr in res.iterrows():
    sub = per_sample[per_sample.study == rr.study]
    c = sub[sub.is_case]["D"]
    t = sub[~sub.is_case]["D"]
    ax.hist(t, bins=30, histtype="step", lw=1.6, color="#1D9E75", alpha=0.85,
            label=f"{rr.study} controls (n={len(t)})")
    ax.hist(c, bins=30, histtype="step", lw=1.6, color="#A32D2D", alpha=0.85, ls="--",
            label=f"{rr.study} cases (n={len(c)})")
ax.set_xlabel("Distance from the healthy-manifold centre, D", fontsize=10)
ax.set_ylabel("No. of samples", fontsize=10)
ax.set_title("Deviation of the gut microbiome from the healthy centre", fontsize=11)
ax.legend(fontsize=7, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
plot_df = res.sort_values("cohens_d")
colors = ["#A32D2D" if p < 0.05 else "#B4B2A9" for p in plot_df.p_mannwhitney]
ax.barh(np.arange(len(plot_df)), plot_df.cohens_d, color=colors)
ax.set_yticks(np.arange(len(plot_df)))
ax.set_yticklabels([f"{s}\n{n_case}/{n_ctrl}" for s, n_case, n_ctrl
                    in zip(plot_df.study, plot_df.n_case, plot_df.n_ctrl)], fontsize=8)
ax.axvline(0, color="#444", lw=0.8)
ax.set_xlabel("Cohen's d (cases − controls, deviation D)", fontsize=10)
ax.set_title("Effect size (red = p < 0.05)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#f0f0f0", lw=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "en_fig4_deviation.png"), bbox_inches="tight")
plt.close()
print("en_fig4 ok")
