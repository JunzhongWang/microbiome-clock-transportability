#!/usr/bin/env python3
"""汇总分批结果：band_performance.csv / summary.csv / interclock_concordance.csv"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results/m1_multiclock")
spec = importlib.util.spec_from_file_location(
    "mc", os.path.join(ROOT, "scripts/m1_multiclock_validation.py"))
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

band = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RES, "part_band_*.csv")))],
                 ignore_index=True, sort=False)
loso = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RES, "part_loso_*.csv")))],
                 ignore_index=True, sort=False)
band.to_csv(os.path.join(RES, "band_performance.csv"), index=False)
loso.to_csv(os.path.join(RES, "loso_projection.csv"), index=False)

rows = []
for kind in mc.IMPLS:
    b = band[(band.impl == kind) & (band["mode"] == "B_strict")]
    rnd = b[b.cv == "random"].set_index("band")
    grp = b[b.cv == "group"].set_index("band")
    tr = b[b.cv == "transfer"].set_index("band")
    l = loso[loso.impl == kind]
    rows.append(dict(
        impl=kind, family=mc.FAMILY[kind], space=mc.SPACE[kind],
        n_feat=int(rnd.loc["全跨度 0-107", "n_feat"]),
        r2_full=rnd.loc["全跨度 0-107", "r2"],
        mae_full=rnd.loc["全跨度 0-107", "mae"],
        r2_child=rnd.loc["儿童 <18", "r2"],
        mae_child=rnd.loc["儿童 <18", "mae"],
        r2_young=rnd.loc["青年 18-40", "r2"],
        r2_40_70=rnd.loc["中年 40-70", "r2"],
        mae_40_70=rnd.loc["中年 40-70", "mae"],
        r2_60p=rnd.loc["老年 60+", "r2"],
        r2_full_group=grp.loc["全跨度 0-107", "r2"],
        r2_40_70_group=grp.loc["中年 40-70", "r2"],
        delta=rnd.loc["全跨度 0-107", "r2"] - rnd.loc["中年 40-70", "r2"],
        transfer_40_70=tr.loc["非中年→40-70", "r2"] if "非中年→40-70" in tr.index else np.nan,
        loso_r_median=float(l.r_proj.median()),
        loso_r_q25=float(l.r_proj.quantile(0.25)),
        loso_r_q75=float(l.r_proj.quantile(0.75)),
        loso_r_max=float(l.r_proj.max()),
        gate_pass=float((l.r_proj >= 0.3).mean()),
        n_study=int(len(l)),
        bias_sd=float(l.bias.std(ddof=1)),
        bias_absmed=float(l.bias.abs().median()),
        bias_absmax=float(l.bias.abs().max()),
    ))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RES, "summary.csv"), index=False)

# 实现间一致性（OOF 预测）
y = np.load(os.path.join(RES, "_y_strict.npy"))
oof = {k: np.load(os.path.join(RES, f"_oof_{k}.npy")) for k in mc.IMPLS
       if os.path.exists(os.path.join(RES, f"_oof_{k}.npy"))}
conc = []
keys = list(oof)
for i, a in enumerate(keys):
    for b_ in keys[i + 1:]:
        pa, pb = oof[a], oof[b_]
        ok = np.isfinite(pa) & np.isfinite(pb)
        r, _ = stats.pearsonr(pa[ok], pb[ok])
        conc.append(dict(impl_a=a, impl_b=b_, r=float(r), n=int(ok.sum())))
pd.DataFrame(conc).to_csv(os.path.join(RES, "interclock_concordance.csv"), index=False)

print("=" * 104)
print("表  六种时钟实现在同一健康训练集上的表现（严格口径，n = 7,534）")
print("=" * 104)
print(f"{'实现':<26}{'R²全':>7}{'MAE':>6}{'R²<18':>7}{'R²40-70':>8}{'ΔR²':>7}"
      f"{'R²40-70组':>10}{'LOSO中位':>9}{'门通过':>7}{'biasSD':>7}")
print("-" * 104)
for _, r in summ.iterrows():
    print(f"{r.impl+' ('+r.space+')':<26}{r.r2_full:>7.3f}{r.mae_full:>6.1f}{r.r2_child:>7.3f}"
          f"{r.r2_40_70:>8.3f}{r.delta:>7.3f}{r.r2_40_70_group:>10.3f}"
          f"{r.loso_r_median:>9.3f}{r.gate_pass*100:>6.0f}%{r.bias_sd:>7.1f}")
print("=" * 104)
print(f"实现间 OOF 预测相关 r：{min(c['r'] for c in conc):.2f} – {max(c['r'] for c in conc):.2f}")
print(f"跨实现 R²(全跨度) 范围 {summ.r2_full.min():.3f}–{summ.r2_full.max():.3f}；"
      f"R²(40-70) 范围 {summ.r2_40_70.min():.3f}–{summ.r2_40_70.max():.3f}；"
      f"ΔR² 最小 {summ.delta.min():.3f}")
