#!/usr/bin/env python3
"""M1 收尾 第二步（修订版）：年龄分层时钟实验

上一版内部口径与已发表报告不一致（特征 107 vs 178，全体 R² 0.214 vs 0.486）。
本版完全复刻 m1_metaphlan_mbag.py 的口径，并同时跑两种健康定义：

  口径 A（宽松，与旧报告一致）: study_condition 含 healthy|control  -> 7657 例
  口径 B（严格）             : disease == healthy                   -> 7534 例

核心问题：已发表时钟的 R²=0.44-0.49 是在 0-107 岁全跨度上得到的。
慢性肝病患者几乎全部落在 40-70 岁。若时钟在该窄带内预测力接近 0，
所有"肝病患者菌群加速/减速衰老 N 岁"的报道都缺乏测量学基础。

输出:
  results/m1_metaphlan/age_stratified_clock.csv
  results/m1_metaphlan/fig_age_stratified.png
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/metaphlan")
OUT = os.path.join(ROOT, "results/m1_metaphlan")
SEED = 42
MIN_PREVALENCE = 0.10
VAR_PERCENTILE = 10
PSEUDO = 1e-6


# ---------------- 载入（复刻旧脚本） ----------------
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
    print(f"[载入] {X.shape[0]} 样本 × {X.shape[1]} 物种, {len(frames)} 个研究文件")
    return X, meta


def select_features(X):
    prev = (X > 0).mean(axis=0)
    X1 = X.loc[:, prev >= MIN_PREVALENCE]
    var = X1.var(axis=0)
    thr = np.percentile(var, VAR_PERCENTILE)
    X2 = X1.loc[:, var >= thr]
    return X2


def log_transform(X):
    return np.log10(np.asarray(X, dtype=float) + PSEUDO)


def make_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=15, random_state=SEED)


def cv_eval(Xl, y, tag, mode):
    """StratifiedKFold(按年龄分箱)，复刻旧口径"""
    n_splits = 5
    if len(y) < 100:
        return None
    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    except Exception:
        bins = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(skf.split(Xl, bins))
    rec = dict(mode=mode, tag=tag, n=len(y), n_feat=Xl.shape[1],
               age_min=round(float(y.min())), age_max=round(float(y.max())),
               age_sd=round(float(y.std()), 1))
    for kind in ["ridge", "gbdt"]:
        pred = np.full(len(y), np.nan)
        for tr, te in splits:
            m = make_model(kind).fit(Xl[tr], y[tr])
            pred[te] = m.predict(Xl[te])
        r, _ = stats.pearsonr(y, pred)
        rec[f"{kind}_r2"] = round(r2_score(y, pred), 3)
        rec[f"{kind}_mae"] = round(mean_absolute_error(y, pred), 2)
        rec[f"{kind}_r"] = round(r, 3)
    return rec


def transfer_eval(Xl, y, tr_idx, te_idx, tag, mode):
    if tr_idx.sum() < 100 or te_idx.sum() < 50:
        return None
    m = make_model("ridge").fit(Xl[tr_idx], y[tr_idx])
    pred = m.predict(Xl[te_idx])
    yte = y[te_idx]
    r, _ = stats.pearsonr(yte, pred)
    return dict(mode=mode, tag=tag, n=int(te_idx.sum()), n_feat=Xl.shape[1],
                age_min=round(float(yte.min())), age_max=round(float(yte.max())),
                age_sd=round(float(yte.std()), 1),
                ridge_r2=round(r2_score(yte, pred), 3),
                ridge_mae=round(mean_absolute_error(yte, pred), 2),
                ridge_r=round(r, 3), gbdt_r2=np.nan, gbdt_mae=np.nan, gbdt_r=np.nan)


# ---------------- 主流程 ----------------
X, meta = load_panel()
meta.columns = [c.lower() for c in meta.columns]
age = pd.to_numeric(meta["age"], errors="coerce")
lab = meta["study_condition"].astype(str).str.lower()
dis = meta["disease"].astype(str).str.lower()

defs = {
    "A_loose": (age.notna() & lab.str.contains("healthy|control", na=False)),
    "B_strict": (age.notna() & dis.eq("healthy")),
}

rows = []
for mode, keep in defs.items():
    Xtr = select_features(X.loc[keep])
    Xl = log_transform(Xtr)
    y = age.loc[keep].to_numpy(dtype=float)
    print(f"\n[{mode}] {len(y)} 例 × {Xtr.shape[1]} 特征, 年龄 {y.min():.0f}-{y.max():.0f}")

    bands = {
        "全体 0-107": np.ones(len(y), bool),
        "成人 18+": y >= 18,
        "中年 40-70": (y >= 40) & (y <= 70),
        "青年 18-40": (y >= 18) & (y < 40),
        "老年 60+": y >= 60,
        "儿童 <18": y < 18,
    }
    for tag, m in bands.items():
        r = cv_eval(Xl[m], y[m], tag, mode)
        if r:
            rows.append(r)
            print(f"   {tag:<12} n={r['n']:<5} sd={r['age_sd']:5.1f} | "
                  f"Ridge R²={r['ridge_r2']:6.3f} MAE={r['ridge_mae']:5.2f} | "
                  f"GBDT R²={r['gbdt_r2']:6.3f}")

    # 跨带迁移
    tr_all = np.ones(len(y), bool)
    for tag, tri, tei in [
        ("全体→中年", tr_all, (y >= 40) & (y <= 70)),
        ("儿童→成人", y < 18, y >= 18),
        ("成人→儿童", y >= 18, y < 18),
    ]:
        r = transfer_eval(Xl, y, tri, tei, tag, mode)
        if r:
            rows.append(r)
            print(f"   {tag:<12} n_test={r['n']:<5} R²={r['ridge_r2']:8.3f} MAE={r['ridge_mae']:6.2f} r={r['ridge_r']:6.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "age_stratified_clock.csv"), index=False)
print(f"\n→ {os.path.join(OUT, 'age_stratified_clock.csv')}  ({len(df)} 行)")

# ---------------- 图：只用严格口径的 CV 结果 ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

cv = df[(df.mode == "B_strict") & df.gbdt_r2.notna()].copy()
fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=150)
ypos = np.arange(len(cv))[::-1]
ax.barh(ypos + 0.2, cv.ridge_r2, height=0.36, color="#378ADD", label="Ridge")
ax.barh(ypos - 0.2, cv.gbdt_r2, height=0.36, color="#1D9E75", label="GBDT")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{t}\nn={n:,} · SD={s:.1f}岁"
                    for t, n, s in zip(cv.tag, cv.n, cv.age_sd)], fontsize=9)
ax.axvline(0, color="#888780", lw=0.8)
ax.axvline(0.444, color="#A32D2D", ls="--", lw=1.2)
ax.text(0.452, len(cv) - 0.6, "Metabolism 2027\n报告值 0.444", fontsize=8,
        color="#A32D2D", va="top")
ax.set_xlabel("交叉验证 R²（用菌群预测年龄）", fontsize=11)
ax.set_xlim(-0.02, 0.56)
ax.set_title("菌群衰老时钟的预测力高度依赖年龄跨度：成人带内 R² < 0.05", fontsize=12, pad=12)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#ececec", lw=0.6)
ax.set_axisbelow(True)
for i, r2 in enumerate(cv.ridge_r2):
    ax.text(max(r2, 0) + 0.006, ypos[i] + 0.2, f"{r2:.3f}", va="center",
            fontsize=8, color="#378ADD")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_age_stratified.png"), bbox_inches="tight")
print(f"→ {os.path.join(OUT, 'fig_age_stratified.png')}")
