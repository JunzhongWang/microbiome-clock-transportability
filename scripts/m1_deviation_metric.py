#!/usr/bin/env python3
"""M1 收尾 第三步：健康流形偏离度（不依赖年龄的替代度量）

逻辑：
  时钟（pred_age→mBAG）在肝病队列上失效，是因为它依赖年龄建模、且是在
  健康人群的年龄梯度上学习的。改问一个更简单的问题——
  「这个样本的菌群离健康人群的中心有多远」——不需要年龄。

方法：
  1. 在健康训练集上：特征筛选 → log → 标准化 → PCA(前 30 个 PC)
  2. 定义健康流形距离 D = sqrt( Σ (PC_i / SD_i)^2 )，SD 取健康集内各 PC 的标准差
     （对角马氏距离，PC 空间内标准化）
  3. 把每个肝病样本投影进来，计算 D
  4. 关键对照（防止「中国菌群天然偏离欧美训练集」的混杂）：
     - 训练集内部：每个研究自己的 D 分布（研究间天然偏移有多大）
     - QinN 对照组 vs 训练集：基线偏移
     - QinN 病例 vs QinN 对照：真正的疾病信号

输出:
  results/m1_metaphlan/deviation_scores.csv
  results/m1_metaphlan/deviation_report.md
  results/m1_metaphlan/fig4_deviation.png
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
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/metaphlan")
OUT = os.path.join(ROOT, "results/m1_metaphlan")
SEED = 42
N_PC = 30
MIN_PREVALENCE = 0.10
PSEUDO = 1e-6


def load_panel():
    meta = pd.read_csv(os.path.join(DATA, "sample_metadata.tsv.gz"), sep="\t",
                       index_col=0, compression="infer")
    meta.index = meta.index.astype(str)
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA, "per_study", "*.tsv.gz"))):
        df = pd.read_csv(p, sep="\t", index_col=0)
        df.index = df.index.astype(str)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df.apply(pd.to_numeric, errors="coerce").fillna(0.0))
    X = pd.concat(frames, axis=0, sort=False)
    common = X.index.intersection(meta.index)
    X, meta = X.loc[common], meta.loc[common]
    X = X.fillna(0.0)
    X = X.loc[:, X.sum(axis=0) > 0]
    return X, meta


X, meta = load_panel()
meta.columns = [c.lower() for c in meta.columns]
lab = meta["study_condition"].astype(str).str.lower()
dis = meta["disease"].astype(str).str.lower()
age = pd.to_numeric(meta["age"], errors="coerce")

# ---------- 1. 健康训练集 ----------
healthy_mask = dis.eq("healthy") & age.notna()
Xh_raw = X.loc[healthy_mask]
prev = (Xh_raw > 0).mean(axis=0)
Xh_raw = Xh_raw.loc[:, prev >= MIN_PREVALENCE]
feat = Xh_raw.columns
print(f"[健康训练集] {Xh_raw.shape[0]} 例 × {Xh_raw.shape[1]} 特征")

scaler = StandardScaler()
Xh_log = np.log10(Xh_raw.values.astype(float) + PSEUDO)
Xh_s = scaler.fit_transform(Xh_log)
pca = PCA(n_components=N_PC, random_state=SEED)
PC = pca.fit_transform(Xh_s)
sd_pc = PC.std(axis=0)
evr = pca.explained_variance_ratio_.sum()
print(f"[PCA] {N_PC} 个主成分，累计解释方差 {evr:.1%}")


def deviation(Xsub_raw):
    """对任意样本子集计算健康流形距离（对角马氏, PC 空间标准化）"""
    Xl = np.log10(Xsub_raw[feat].values.astype(float) + PSEUDO)
    Xs = scaler.transform(Xl)
    pc = pca.transform(Xs)
    return np.sqrt(((pc / sd_pc) ** 2).sum(axis=1)), pc


# ---------- 2. 训练集内参照：研究间天然偏移 ----------
D_train, _ = deviation(Xh_raw)
ref = pd.DataFrame({"sampleID": Xh_raw.index, "D": D_train})
ref = ref.join(meta[["study"]], on="sampleID")
per_study_ref = ref.groupby("study")["D"].agg(["mean", "median", "count"])
print(f"[训练集内 D] 中位 {np.median(D_train):.2f} | P95 {np.percentile(D_train, 95):.2f}")
print(f"[训练集内研究间 D 中位的 SD] {per_study_ref['median'].std():.2f} "
      f"（这就是纯人群/批次的天然偏移幅度）")

# ---------- 3. 肝病样本 ----------
liver_mask = lab.str.contains("cirrhos|nafld|nash|hepat|fibros|steato|hcc|liver", na=False)
liver_studies = meta.loc[liver_mask, "study"].unique()
rows, per_sample = [], []
for st in sorted(liver_studies):
    smask = (meta["study"] == st)
    Xs = X.loc[smask]
    if len(Xs) < 20:
        continue
    D, _ = deviation(Xs)
    sub = meta.loc[smask]
    case = sub["study_condition"].astype(str).str.lower().str.contains(
        "cirrhos|nafld|nash|hepat|fibros|steato|hcc|liver")
    ctrl = sub["study_condition"].astype(str).str.lower().str.contains("healthy|control")
    d_case, d_ctrl = D[case.values], D[ctrl.values]
    if len(d_case) < 10 or len(d_ctrl) < 10:
        continue
    pooled = np.concatenate([d_case, d_ctrl])
    sp = np.sqrt((d_case.var() + d_ctrl.var()) / 2)
    d = (d_case.mean() - d_ctrl.mean()) / sp if sp > 0 else np.nan
    u, p = stats.mannwhitneyu(d_case, d_ctrl, alternative="two-sided")
    rows.append(dict(
        study=st, n_case=int(case.sum()), n_ctrl=int(ctrl.sum()),
        D_case=round(float(np.mean(d_case)), 2), D_ctrl=round(float(np.mean(d_ctrl)), 2),
        diff=round(float(d_case.mean() - d_ctrl.mean()), 2),
        cohens_d=round(float(d), 3),
        p_mannwhitney=float(p),
    ))
    per_sample.append(pd.DataFrame({
        "sampleID": Xs.index, "study": st, "D": D,
        "is_case": case.values, "age": age.loc[smask].values}))

res = pd.DataFrame(rows).sort_values("cohens_d", ascending=False)
res.to_csv(os.path.join(OUT, "deviation_scores.csv"), index=False)
per_sample = pd.concat(per_sample, ignore_index=True) if per_sample else pd.DataFrame()
if len(per_sample):
    per_sample.to_csv(os.path.join(OUT, "deviation_per_sample.csv"), index=False)

print("\n" + "=" * 86)
print("健康流形偏离度：肝病 vs 同队列对照")
print("=" * 86)
if len(res):
    print(res.to_string(index=False))
    sig = res[res.p_mannwhitney < 0.05]
    print(f"\n显著（p<0.05）: {len(sig)}/{len(res)} 个队列")
    print(f"训练集内研究间 D 中位 SD = {per_study_ref['median'].std():.2f}（天然偏移）")

# ---------- 4. 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)

ax = axes[0]
ax.hist(D_train, bins=40, color="#D3D1C7", alpha=0.9, label=f"健康训练集 n={len(D_train):,}")
for _, r in res.iterrows():
    sub = per_sample[per_sample.study == r.study]
    c = sub[sub.is_case]["D"]; t = sub[~sub.is_case]["D"]
    ax.hist(t, bins=30, histtype="step", lw=1.6, color="#1D9E75", alpha=0.85,
            label=f"{r.study} 对照 n={len(t)}")
    ax.hist(c, bins=30, histtype="step", lw=1.6, color="#A32D2D", alpha=0.85, ls="--",
            label=f"{r.study} 病例 n={len(c)}")
ax.set_xlabel("健康流形偏离度 D", fontsize=10)
ax.set_ylabel("样本数", fontsize=10)
ax.set_title("菌群偏离健康中心的距离分布", fontsize=11)
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
ax.set_xlabel("Cohen's d（病例 − 对照 的偏离度差异）", fontsize=10)
ax.set_title("效应量（红 = p<0.05）", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#f0f0f0", lw=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_deviation.png"), bbox_inches="tight")
plt.close()
print(f"\n→ {os.path.join(OUT, 'fig4_deviation.png')}")
