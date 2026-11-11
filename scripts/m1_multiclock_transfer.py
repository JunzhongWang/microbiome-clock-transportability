#!/usr/bin/env python3
"""跨带迁移实验（修正版）

问题
----
早期实现（m1_age_stratified_clock.py 与本目录首版）的"全体→中年 40-70"实验，
训练掩码取 np.ones(len(y))，即训练集**包含**被测试的中年样本，属于样本内评估。
后果：容量大的模型（RF/GBM）在样本内可贴合训练标签，给出虚高的 R²（如 RF 0.67），
而线性模型因预测被压缩到全局均值附近，给出虚低的负值——两者都不是"迁移"。

修正
----
训练集与测试带严格不相交：
  T1 非中年（＜40 或 ＞70） → 中年 40-70      带内插值
  T2 儿童（＜18）           → 成人（≥18）     外推
  T3 中年（40-70）          → 非中年          带内模型用于带外

输出
----
  results/m1_multiclock/transfer_fixed.csv
  并就地替换 band_performance.csv 中的 cv == "transfer" 行
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results/m1_multiclock")
spec = importlib.util.spec_from_file_location(
    "mc", os.path.join(ROOT, "scripts/m1_multiclock_validation.py"))
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

X, meta = mc.load_panel()
meta.columns = [c.lower() for c in meta.columns]
age = pd.to_numeric(meta["age"], errors="coerce")
dis = meta["disease"].astype(str).str.lower()
keep = age.notna() & dis.eq("healthy")

Xs = mc.select_features(X.loc[keep])
Xg = mc.select_features(mc.to_genus(X.loc[keep]))
FEATS = {"species": mc.log_transform(Xs), "genus": mc.log_transform(Xg)}
y = age.loc[keep].to_numpy(dtype=float)
print(f"[数据] n={len(y)}, 物种 {Xs.shape[1]} / 属 {Xg.shape[1]}")

TRANSFERS = [
    ("非中年→40-70", (y < 40) | (y > 70), (y >= 40) & (y <= 70)),
    ("儿童→成人", y < 18, y >= 18),
    ("中年→非中年", (y >= 40) & (y <= 70), (y < 40) | (y > 70)),
]

rows = []
for kind in mc.IMPLS:
    Xl = FEATS[mc.SPACE[kind]]
    for tag, trm, tem in TRANSFERS:
        assert not np.any(trm & tem), f"{tag} 训练/测试重叠"
        pred = mc.fit_predict(kind, Xl[trm], y[trm], Xl[tem])
        yt = y[tem]
        r, _ = stats.pearsonr(yt, pred) if np.std(pred) > 0 else (0.0, 1.0)
        lo, hi = mc.boot_r2_ci(yt, pred)
        rows.append(dict(mode="B_strict", impl=kind, family=mc.FAMILY[kind],
                         cv="transfer_fixed", band=tag, n_feat=Xl.shape[1],
                         age_sd=round(float(np.std(yt, ddof=1)), 1),
                         r2_lo=round(lo, 3), r2_hi=round(hi, 3),
                         tag=tag, n=int(tem.sum()),
                         r2=float(r2_score(yt, pred)),
                         mae=float(mean_absolute_error(yt, pred)),
                         r=float(r), p=np.nan,
                         resid_sd=float(np.std(yt - pred, ddof=1)),
                         pred_sd=float(np.std(pred, ddof=1))))
        print(f"  {kind:4s} {tag:<12} n={tem.sum():<5} R²={rows[-1]['r2']:8.3f} "
              f"[{lo:7.3f},{hi:7.3f}] MAE={rows[-1]['mae']:6.2f} r={r:6.3f}")

new = pd.DataFrame(rows)
new.to_csv(os.path.join(RES, "transfer_fixed.csv"), index=False)

old = pd.read_csv(os.path.join(RES, "band_performance.csv"))
old = old[old.cv != "transfer"]
merged = pd.concat([old, new], ignore_index=True, sort=False)
merged.to_csv(os.path.join(RES, "band_performance.csv"), index=False)
print(f"\n→ transfer_fixed.csv（{len(new)} 行）；band_performance.csv 已替换旧 transfer 行")
