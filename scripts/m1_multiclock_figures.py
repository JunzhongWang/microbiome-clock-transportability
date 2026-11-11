#!/usr/bin/env python3
"""跨实现验证的图（投稿用 4 面板合成图 + 附加文件图）

图 2（主图，170 mm × 约 130 mm，300 dpi）
  a  随机 5 折 CV：六种实现的年龄段 R²
  b  按研究分组 5 折 CV：同上（更严格）
  c  留一研究投影：逐研究 r 的分布
  d  校准门通过率与投影偏移
附加文件图 S1  实现间预测一致性
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results/m1_multiclock")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.7

LABEL = {"EN": "Elastic Net", "RF": "Random Forest", "GBM": "Gradient Boosting",
         "DNN": "Deep NN", "SVR": "SVR (RBF)", "RFg": "Random Forest (genus)"}
COLOR = {"EN": "#2F6DB5", "RF": "#1D9E75", "GBM": "#D9822B",
         "DNN": "#A32D2D", "SVR": "#6B5CA5", "RFg": "#8C6D31"}
ORDER = ["EN", "RF", "GBM", "DNN", "SVR", "RFg"]
BANDS = ["儿童 <18", "青年 18-40", "中年 40-70", "老年 60+", "全跨度 0-107"]
BANDS_EN = ["<18", "18–40", "40–70", "60+", "0–107"]

band = pd.read_csv(os.path.join(RES, "band_performance.csv"))
loso = pd.read_csv(os.path.join(RES, "loso_projection.csv"))
summ = pd.read_csv(os.path.join(RES, "summary.csv")).set_index("impl")
conc = pd.read_csv(os.path.join(RES, "interclock_concordance.csv"))

# 170 mm 宽 × 132 mm 高（BMC 满页宽）
fig, axes = plt.subplots(2, 2, figsize=(6.69, 5.20), dpi=300)
(axA, axB), (axC, axD) = axes

for ax, scheme, tag in [(axA, "random", "a"), (axB, "group", "b")]:
    for k in ORDER:
        vals = []
        for b in BANDS:
            s = band[(band.impl == k) & (band.cv == scheme) & (band.band == b)]
            vals.append(s.r2.iloc[0] if len(s) else np.nan)
        ax.plot(range(len(BANDS)), vals, "-o", color=COLOR[k], lw=1.1, ms=2.8,
                label=LABEL[k], zorder=3, markeredgewidth=0)
    ax.axhline(0, color="#777", lw=0.7, zorder=1)
    ax.axhline(0.444, color="#A32D2D", ls="--", lw=0.8, zorder=1)
    ax.axvspan(1.5, 2.5, color="#F0F0F0", zorder=0)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS_EN, fontsize=7.5)
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(-0.74, 0.72)
    ax.set_xlabel("Age band (years)", fontsize=8)
    ax.set_ylabel("Cross-validated $R^2$", fontsize=8)
    ax.set_title(f"({tag}) {'Random' if scheme=='random' else 'Study-grouped'} 5-fold CV",
                 fontsize=8.5, pad=4, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#EFEFEF", lw=0.5)
    ax.set_axisbelow(True)
axA.text(-0.25, 0.415, "published clock: 0.444", fontsize=6, color="#A32D2D",
         ha="left", va="top")
axA.legend(fontsize=6.2, frameon=False, loc="lower left", handlelength=1.4,
           labelspacing=0.25, borderpad=0.1)

# (c) LOSO
rng = np.random.RandomState(0)
for i, k in enumerate(ORDER):
    v = loso[loso.impl == k]["r_proj"].values
    axC.scatter(v, np.full_like(v, i) + rng.uniform(-0.13, 0.13, len(v)),
                s=5, color=COLOR[k], alpha=0.5, edgecolors="none", zorder=3)
    axC.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=COLOR[k], lw=1.8, zorder=4)
axC.axvline(0.3, color="#A32D2D", ls="--", lw=0.8)
axC.axvline(0, color="#777", lw=0.7)
axC.text(0.315, 5.42, "gate 0.3", fontsize=6.2, color="#A32D2D")
axC.set_yticks(range(len(ORDER)))
axC.set_yticklabels(["EN", "RF", "GBM", "DNN", "SVR", "RFg"], fontsize=7)
axC.set_ylim(-0.6, 5.8)
axC.set_xlim(-0.5, 0.85)
axC.set_xlabel("Within-study $r$ (predicted vs. chronological age)", fontsize=8)
axC.set_title("(c) Leave-one-study-out projection", fontsize=8.5, pad=4, loc="left")
axC.spines[["top", "right"]].set_visible(False)
axC.tick_params(labelsize=7)
axC.grid(axis="x", color="#EFEFEF", lw=0.5)
axC.set_axisbelow(True)

# (d) gate pass + bias SD
gp = [summ.loc[k, "gate_pass"] * 100 for k in ORDER]
bs = [summ.loc[k, "bias_sd"] for k in ORDER]
x = np.arange(len(ORDER))
axD.bar(x, gp, color=[COLOR[k] for k in ORDER], width=0.62)
for i, (g, b) in enumerate(zip(gp, bs)):
    axD.text(i, g + 1.6, f"{g:.0f}%", ha="center", fontsize=6.4)
    axD.text(i, 2.2, f"{b:.0f}", ha="center", fontsize=5.6, rotation=90,
             color="white" if g > 25 else "#333")
axD.set_xticks(x)
axD.set_xticklabels(["EN", "RF", "GBM", "DNN", "SVR", "RFg"], fontsize=7)
axD.set_ylim(0, 58)
axD.set_ylabel("Studies passing gate (%)", fontsize=8)
axD.set_title("(d) Calibration-gate pass rate", fontsize=8.5, pad=4, loc="left")
axD.spines[["top", "right"]].set_visible(False)
axD.tick_params(labelsize=7)
axD.grid(axis="y", color="#EFEFEF", lw=0.5)
axD.set_axisbelow(True)
axD.text(2.0, 46, "numbers inside bars:\nSD of projection bias (years)",
         fontsize=5.8, ha="center", color="#444", linespacing=1.2)

fig.text(0.5, 0.005,
         "Grey band marks 40–70 y, the age range of most chronic liver disease cohorts",
         ha="center", fontsize=6.4, color="#555", style="italic")
plt.tight_layout(rect=(0, 0.018, 1, 1), h_pad=1.1, w_pad=1.2)
plt.savefig(os.path.join(RES, "fig2_multiclock.png"), bbox_inches="tight")
plt.savefig(os.path.join(RES, "fig2_multiclock.pdf"), bbox_inches="tight")
plt.close()

# 原单图另存，便于查看
for f in ["fig5_multiclock_bands.png", "fig6_loso_gate.png"]:
    pass

# 附加文件图 S1
M = pd.DataFrame(np.eye(len(ORDER)), index=ORDER, columns=ORDER)
for _, r in conc.iterrows():
    M.loc[r.impl_a, r.impl_b] = r.r
    M.loc[r.impl_b, r.impl_a] = r.r
fig, ax = plt.subplots(figsize=(3.35, 2.9), dpi=300)
im = ax.imshow(M.values, cmap="YlGnBu", vmin=0.6, vmax=1.0)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(["EN", "RF", "GBM", "DNN", "SVR", "RFg"], fontsize=7)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([LABEL[k] for k in ORDER], fontsize=6.5)
for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        v = M.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                color="white" if v > 0.9 else "#222")
ax.set_title("Agreement between implementations (OOF r)", fontsize=7.5, pad=6)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6)
cb.set_label("Pearson r", fontsize=6.5)
plt.tight_layout()
plt.savefig(os.path.join(RES, "figS1_concordance.png"), bbox_inches="tight")
plt.savefig(os.path.join(RES, "figS1_concordance.pdf"), bbox_inches="tight")
plt.close()

print("已生成：fig2_multiclock.(png|pdf)、figS1_concordance.(png|pdf)")
