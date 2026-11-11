#!/usr/bin/env python3
"""M1 基线时钟: 训练 + Study 分层交叉验证。

对标 Metabolism 2027 3.2 节: 全寿命队列 R2=0.444, MAE=13.09, Pearson r=0.667
做法: log1p -> StandardScaler -> Ridge, GroupKFold(5) 按 Study 分组(防同研究泄漏)
用法: python3 scripts/m1_train_clock.py [p10|p20]
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "p10"
X = pd.read_csv(ROOT / f"results/m1/train_features_{SUFFIX}.csv", index_col=0)
M = pd.read_csv(ROOT / f"results/m1/train_meta_{SUFFIX}.csv")
M = M.set_index("Sample").loc[X.index]
y = M["Age"].astype(float).values
groups = M["Study"].astype(str).values

print(f"features: {X.shape} | age {y.min():.0f}-{y.max():.0f} (median {np.median(y):.0f})")
print(f"groups(Study): {len(set(groups))} 个, 最大组 {pd.Series(groups).value_counts().iloc[0]} 例")

pipe = make_pipeline(
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-1, 3, 12)),
)
cv = GroupKFold(n_splits=5)
pred = cross_val_predict(pipe, X.values, y, groups=groups, cv=cv, n_jobs=-1)

r2, mae, r = r2_score(y, pred), mean_absolute_error(y, pred), pearsonr(y, pred)[0]
print("\n=== 交叉验证结果 (GroupKFold by Study, 5 折) ===")
print(f"  R2   = {r2:.3f}   (Metabolism 2027 原文 0.444)")
print(f"  MAE  = {mae:.2f} 岁 (原文 13.09)")
print(f"  r    = {r:.3f}   (原文 0.667)")

# 对照: 随机样本级划分(会因同研究样本泄漏而虚高, 仅用于与文献口径对齐)
pred_rand = cross_val_predict(pipe, X.values, y, cv=KFold(5, shuffle=True, random_state=42), n_jobs=-1)
print("\n=== 对照: 随机 5 折(样本级, 存在研究泄漏) ===")
print(f"  R2   = {r2_score(y, pred_rand):.3f} | MAE = {mean_absolute_error(y, pred_rand):.2f} | r = {pearsonr(y, pred_rand)[0]:.3f}")
print("  注: 与分组结果的落差 = 研究批次效应贡献的虚高部分, 论文中必须两者都报")

# 分折指标
fold_rows = []
for i, (tr, te) in enumerate(cv.split(X.values, y, groups)):
    fold_rows.append({"fold": i + 1, "n_train": len(tr), "n_test": len(te),
                      "r2": r2_score(y[te], pred[te]), "mae": mean_absolute_error(y[te], pred[te])})
folds = pd.DataFrame(fold_rows)
print("\n各折:", " | ".join(f"F{r.fold}: R2={r.r2:.2f} MAE={r.mae:.1f}" for r in folds.itertuples()))

out = pd.DataFrame({"Sample": X.index, "Study": groups, "Age": y, "pred_age": pred,
                    "BAG": pred - y})
out.to_csv(ROOT / f"results/m1/clock_baseline_cv_{SUFFIX}.csv", index=False)
folds.to_csv(ROOT / f"results/m1/clock_baseline_folds_{SUFFIX}.csv", index=False)
print(f"\nsaved: results/m1/clock_baseline_cv_{SUFFIX}.csv ({len(out)} 行, 含 OOF 预测与 BAG)")

# 年龄组分层误差: 检验中年段是否更难预测(原文的核心论点)
bins = pd.cut(y, [0, 18, 35, 50, 65, 120], labels=["0-18", "19-35", "36-50", "51-65", "65+"])
seg = out.groupby(bins, observed=True).apply(
    lambda d: pd.Series({"n": len(d), "MAE": mean_absolute_error(d.Age, d.pred_age),
                         "r": pearsonr(d.Age, d.pred_age)[0] if len(d) > 5 else np.nan}),
    include_groups=False)
print("\n按年龄段:")
print(seg.round(3).to_string())
