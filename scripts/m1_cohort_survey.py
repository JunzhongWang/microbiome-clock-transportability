#!/usr/bin/env python3
"""M1 收尾 第一步：队列普查 + 队列内自训时钟可行性

回答两个问题：
  Q1 每个研究有多少样本、多少有精确年龄、年龄跨度多大 —— 决定能否自训时钟
  Q2 那些样本量够的队列，自己训一个时钟能达到多少 R² —— 与全局时钟投影对照

输出:
  results/m1_metaphlan/cohort_survey.csv
  results/m1_metaphlan/intracohort_clock.csv
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_STUDY = os.path.join(ROOT, "data/metaphlan/per_study")
META = os.path.join(ROOT, "data/metaphlan/sample_metadata.tsv.gz")
OUT = os.path.join(ROOT, "results/m1_metaphlan")

MIN_N_AGE = 60          # 自训时钟的最低样本量
MIN_AGE_SD = 10.0       # 年龄跨度门槛（太窄则无法学）
PREV_THRESH = 0.20      # 研究内物种流行率门槛
MAX_FEAT = 150          # 特征上限（防过拟合）
SEED = 42

meta = pd.read_csv(META, sep="\t").set_index("sampleID")
print(f"metadata: {meta.shape[0]} 样本, {meta['study'].nunique()} 研究")

survey_rows, clock_rows = [], []

for path in sorted(glob.glob(os.path.join(PER_STUDY, "*.tsv.gz"))):
    study = os.path.basename(path).replace(".tsv.gz", "")
    try:
        ab = pd.read_csv(path, sep="\t", index_col=0)
    except Exception as e:
        print(f"  [skip] {study}: {e}")
        continue
    if ab.shape[0] < 5:
        continue

    # 统一：行=样本, 列=物种
    common = ab.index.intersection(meta.index)
    if len(common) < 5:
        # 有时表是 物种x样本，转置再试
        ab = ab.T
        common = ab.index.intersection(meta.index)
    if len(common) < 5:
        continue

    ab = ab.loc[common]
    md = meta.loc[common]
    n_total = len(common)
    ages = pd.to_numeric(md["age"], errors="coerce")
    has_age = ages.notna()
    n_age = int(has_age.sum())
    age_v = ages[has_age]

    is_case = md["study_condition"].astype(str).str.lower().isin(
        ["case", "cirrhosis", "nafld", "nash", "ibd", "cancer", "adenoma"]
    )
    n_case = int(is_case.sum())
    n_ctrl = int((~is_case).sum())

    survey_rows.append(dict(
        study=study, n_total=n_total, n_age=n_age,
        age_mean=round(float(age_v.mean()), 1) if n_age else np.nan,
        age_sd=round(float(age_v.std()), 1) if n_age > 1 else np.nan,
        age_min=float(age_v.min()) if n_age else np.nan,
        age_max=float(age_v.max()) if n_age else np.nan,
        n_case=n_case, n_ctrl=n_ctrl, n_species=ab.shape[1],
    ))

    # ---- Q2: 队列内自训时钟 ----
    if n_age < MIN_N_AGE:
        continue
    sd = float(age_v.std())
    if not np.isfinite(sd) or sd < MIN_AGE_SD:
        continue

    X_raw = ab.loc[has_age]
    y = age_v.values.astype(float)

    # 研究内特征筛选
    X = X_raw.astype(float).values
    prev = (X > 0).mean(axis=0)
    var = X.var(axis=0)
    keep = (prev >= PREV_THRESH) & (var > 0)
    if keep.sum() < 10:
        keep = var > 0
    X = X[:, keep]
    if X.shape[1] > MAX_FEAT:
        order = np.argsort(X.var(axis=0))[::-1][:MAX_FEAT]
        X = X[:, order]
    n_feat = X.shape[1]

    # log 变换 + 标准化
    Xl = np.log10(X + 1e-4)
    mu, sdd = Xl.mean(0), Xl.std(0)
    sdd[sdd == 0] = 1.0
    Xs = (Xl - mu) / sdd

    # 年龄分箱分层
    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    except Exception:
        bins = np.zeros_like(y, dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = np.zeros_like(y)
    ok = True
    for tr, te in skf.split(Xs, bins):
        try:
            m = RidgeCV(alphas=np.logspace(-2, 3, 30))
            m.fit(Xs[tr], y[tr])
            pred[te] = m.predict(Xs[te])
        except Exception:
            ok = False
            break
    if not ok:
        continue

    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    r, _ = stats.pearsonr(y, pred)
    clock_rows.append(dict(
        study=study, n=n_age, n_feat=n_feat, age_sd=round(sd, 1),
        r2=round(r2, 3), mae=round(mae, 2), r=round(r, 3),
        alpha_used="cv",
    ))
    print(f"  {study:<28} n={n_age:<4} feat={n_feat:<4} R2={r2:5.3f} MAE={mae:5.2f} r={r:5.3f}")

survey = pd.DataFrame(survey_rows).sort_values("n_age", ascending=False)
survey.to_csv(os.path.join(OUT, "cohort_survey.csv"), index=False)

clock = pd.DataFrame(clock_rows).sort_values("r2", ascending=False)
clock.to_csv(os.path.join(OUT, "intracohort_clock.csv"), index=False)

print("\n" + "=" * 78)
print("普查汇总")
print("=" * 78)
print(f"研究总数 {len(survey)} | 有年龄样本 {int(survey.n_age.sum())} 例")
print(f"可自训时钟（n_age>={MIN_N_AGE} 且 age_sd>={MIN_AGE_SD}）: {len(clock)} 个研究")
if len(clock):
    print(f"  中位 R² {clock.r2.median():.3f} | 中位 MAE {clock.mae.median():.1f} 岁 | 中位 r {clock.r.median():.3f}")
    print(f"  R²>0.3 的研究: {(clock.r2 > 0.3).sum()}/{len(clock)}")
