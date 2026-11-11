#!/usr/bin/env python3
"""M1 收尾：从已有 CSV 产出全部图（不重跑训练）

图 1  年龄分层条形图：时钟预测力随年龄带衰减
图 2  队列间投影偏移 vs 文献 mBAG 效应量
图 3  校准门阈值扫描：通过率 vs 阈值
图 4  54 个研究的 r_age 分布（校准门前后对比）

输出到 results/m1_metaphlan/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/m1_metaphlan")

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(os.path.join(OUT, "age_stratified_clock.csv"))
tr = pd.read_csv(os.path.join(OUT, "clock_transportability.csv"))

# ================= 图 1：年龄分层 =================
cv = df[(df["mode"] == "A_loose") & df.gbdt_r2.notna()].copy()
order = ["全体 0-107", "成人 18+", "青年 18-40", "中年 40-70", "老年 60+", "儿童 <18"]
cv["tag"] = pd.Categorical(cv["tag"], categories=order, ordered=True)
cv = cv.sort_values("tag")

fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=150)
y = np.arange(len(cv))[::-1]
ax.barh(y + 0.2, cv.ridge_r2, height=0.36, color="#378ADD", label="Ridge")
ax.barh(y - 0.2, cv.gbdt_r2, height=0.36, color="#1D9E75", label="GBDT")
ax.set_yticks(y)
ax.set_yticklabels([f"{t}\n n={n:,} · 年龄SD={s:.1f} 岁" for t, n, s in zip(cv.tag, cv.n, cv.age_sd)],
                   fontsize=9)
ax.axvline(0, color="#888780", lw=0.8)
ax.axvline(0.444, color="#A32D2D", ls="--", lw=1.2)
ax.text(0.452, len(cv) - 0.8, "Metabolism 2027\n报告值 0.444", fontsize=8, color="#A32D2D", va="top")
for i, r2 in enumerate(cv.ridge_r2):
    ax.text(max(r2, 0) + 0.006, y[i] + 0.2, f"{r2:.3f}", va="center", fontsize=8, color="#378ADD")
ax.set_xlabel("交叉验证 R²（用菌群预测年龄）", fontsize=11)
ax.set_xlim(-0.02, 0.62)
ax.set_title("菌群「衰老时钟」的预测力高度依赖年龄跨度\n成人窄带（慢性肝病所处区间）内 R² 仅 0.10–0.15",
             fontsize=12, pad=12)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#ececec", lw=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_age_stratified.png"), bbox_inches="tight")
plt.close()
print("fig1 ok")

# ================= 图 2：偏移 vs 效应量 =================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})

ax = axes[0]
b = tr.bias.dropna()
ax.hist(b, bins=24, color="#85B7EB", edgecolor="white")
ax.axvline(0, color="#444", lw=1)
for v, c, lab in [(3, "#1D9E75", "文献 mBAG 效应下限 3 岁"),
                  (7, "#A32D2D", "文献 mBAG 效应上限 7 岁")]:
    ax.axvline(v, color=c, ls="--", lw=1.4)
    ax.axvline(-v, color=c, ls="--", lw=1.4)
ax.text(7.4, ax.get_ylim()[1] * 0.92, "±7 岁\n效应上限", fontsize=8, color="#A32D2D")
ax.text(-11.5, ax.get_ylim()[1] * 0.92, "±3 岁\n效应下限", fontsize=8, color="#1D9E75")
ax.set_xlabel("投影到单个研究后的年龄偏移 pred − age（岁）", fontsize=10)
ax.set_ylabel("研究数", fontsize=10)
ax.set_title(f"54 个健康研究的队列偏移\nSD={b.std():.1f} 岁，是效应量的 2.3 倍", fontsize=11)
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
ax.set_xlabel("校准门阈值 ρ（研究内 pred~age 的 r）", fontsize=10)
ax.set_ylabel("通过研究的比例 (%)", fontsize=10)
ax.set_title("「投影前校准门」阈值扫描\n建议 ρ≥0.3（仅 30% 研究通过）", fontsize=11)
ax.set_ylim(-4, 106)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#f0f0f0", lw=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_bias_and_gate.png"), bbox_inches="tight")
plt.close()
print("fig2 ok")

# ================= 图 3：r_age 分布 + 肝病位置 =================
fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=150)
r = tr.r_age.dropna()
ax.hist(r, bins=20, color="#B5D4F4", edgecolor="white")
ax.axvline(0.3, color="#A32D2D", ls="--", lw=1.4, label="校准门 ρ=0.3")
ax.axvline(0.036, color="#A32D2D", lw=2.2, label="QinN_2014（唯一肝病队列）r=0.036")
ax.set_xlabel("研究内 pred_age ~ age 的相关系数 r（54 个研究）", fontsize=10)
ax.set_ylabel("研究数", fontsize=10)
ax.set_title("大多数研究的时钟投影都无法正常追踪年龄", fontsize=12)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_rage_distribution.png"), bbox_inches="tight")
plt.close()
print("fig3 ok")
