#!/usr/bin/env python3
"""分批执行跨实现验证（规避长任务被回收）

用法：
  python3 scripts/m1_multiclock_run.py EN,GBM
  python3 scripts/m1_multiclock_run.py RF
  ...
每个实现单独落盘，最后由 m1_multiclock_merge.py 汇总。
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results/m1_multiclock")
os.makedirs(RES, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "mc", os.path.join(ROOT, "scripts/m1_multiclock_validation.py"))
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

only = sys.argv[1].split(",") if len(sys.argv) > 1 else mc.IMPLS
stage = sys.argv[2] if len(sys.argv) > 2 else "all"   # all | cv | loso
do_cv = stage in ("all", "cv")
do_loso = stage in ("all", "loso")

X, meta = mc.load_panel()
meta.columns = [c.lower() for c in meta.columns]
age = pd.to_numeric(meta["age"], errors="coerce")
lab = meta["study_condition"].astype(str).str.lower()
dis = meta["disease"].astype(str).str.lower()
study = meta["study"].astype(str)

MASKS = {
    "B_strict": age.notna() & dis.eq("healthy"),
    "A_loose": age.notna() & lab.str.contains("healthy|control", na=False),
}

FEATS = {}
_FEAT_CACHE = os.path.join(RES, "_feats_cache.npz")
if os.path.exists(_FEAT_CACHE):
    _z = np.load(_FEAT_CACHE, allow_pickle=True)
    FEATS = {m: {"species": _z[f"{m}_species"], "genus": _z[f"{m}_genus"]}
             for m in ("B_strict", "A_loose")}
    print("[特征缓存] 已从磁盘载入，跳过重复筛选")
else:
    for mode, keep in MASKS.items():
        Xs = mc.select_features(X.loc[keep])
        Xg = mc.select_features(mc.to_genus(X.loc[keep]))
        FEATS[mode] = {"species": mc.log_transform(Xs), "genus": mc.log_transform(Xg)}
    np.savez_compressed(_FEAT_CACHE,
                        **{f"{m}_{s}": FEATS[m][s]
                           for m in FEATS for s in ("species", "genus")})
    print(f"[特征缓存] 已构建并写入 {_FEAT_CACHE}")
if True:  # 保留原行为：落盘 y/g 以便 merge 使用
    for mode, keep in MASKS.items():
        if mode == "B_strict":
            np.save(os.path.join(RES, "_y_strict.npy"), age.loc[keep].to_numpy(float))
            np.save(os.path.join(RES, "_g_strict.npy"), study.loc[keep].to_numpy())

for kind in only:
    t0 = time.time()
    band_rows, loso_rows = [], []
    for mode in ["B_strict", "A_loose"]:
        keep = MASKS[mode]
        y = age.loc[keep].to_numpy(float)
        g = study.loc[keep].to_numpy()
        Xl = FEATS[mode][mc.SPACE[kind]]
        bands = {
            "全跨度 0-107": np.ones(len(y), bool),
            "儿童 <18": y < 18,
            "青年 18-40": (y >= 18) & (y < 40),
            "中年 40-70": (y >= 40) & (y <= 70),
            "老年 60+": y >= 60,
            "成人 18+": y >= 18,
        }
        schemes = ["random", "group"] if mode == "B_strict" else ["random"]
        mc.CUR_KIND = kind

        for scheme in (schemes if do_cv else []):
            for btag, bm in bands.items():
                if bm.sum() < mc.MIN_BAND_N:
                    continue
                pred = mc.cv_oof(Xl[bm], y[bm], g[bm], scheme)
                mt = mc.metrics(y[bm], pred, btag)
                lo, hi = mc.boot_r2_ci(y[bm], pred)
                band_rows.append(dict(
                    mode=mode, impl=kind, family=mc.FAMILY[kind], cv=scheme,
                    band=btag, n_feat=Xl.shape[1],
                    age_sd=round(float(np.std(y[bm], ddof=1)), 1),
                    r2_lo=round(lo, 3), r2_hi=round(hi, 3), **mt))
                if mode == "B_strict" and scheme == "random" and btag == "全跨度 0-107":
                    np.save(os.path.join(RES, f"_oof_{kind}.npy"), pred)
                print(f"  {mode} {scheme:6s} {btag:<12} R²={mt['r2']:6.3f} "
                      f"MAE={mt['mae']:5.2f} r={mt['r']:6.3f}")

        # 严格带外迁移（训练/测试不相交）
        if mode == "B_strict" and do_cv:
            for ttag, trm, tem in [
                ("非中年→40-70", (y < 40) | (y > 70), (y >= 40) & (y <= 70)),
                ("儿童→成人", y < 18, y >= 18),
                ("中年→非中年", (y >= 40) & (y <= 70), (y < 40) | (y > 70)),
            ]:
                assert not np.any(trm & tem)
                pred = mc.fit_predict(kind, Xl[trm], y[trm], Xl[tem])
                yt = y[tem]
                r, _ = stats.pearsonr(yt, pred) if np.std(pred) > 0 else (0.0, 1.0)
                lo, hi = mc.boot_r2_ci(yt, pred)
                band_rows.append(dict(
                    mode=mode, impl=kind, family=mc.FAMILY[kind], cv="transfer",
                    band=ttag, n_feat=Xl.shape[1],
                    age_sd=round(float(np.std(yt, ddof=1)), 1),
                    r2_lo=round(lo, 3), r2_hi=round(hi, 3), tag=ttag,
                    n=int(tem.sum()), r2=float(mc.r2_score(yt, pred)),
                    mae=float(mean_absolute_error(yt, pred)), r=float(r), p=np.nan,
                    resid_sd=float(np.std(yt - pred, ddof=1)),
                    pred_sd=float(np.std(pred, ddof=1))))
                print(f"  {mode} transfer {ttag:<12} R²={band_rows[-1]['r2']:8.3f} "
                      f"MAE={band_rows[-1]['mae']:6.2f}")

        if mode == "B_strict" and do_loso:
            for s in pd.unique(g):
                tem = g == s
                trm = ~tem
                if tem.sum() < mc.MIN_STUDY_N or trm.sum() < 500:
                    continue
                pred = mc.fit_predict(kind, Xl[trm], y[trm], Xl[tem])
                yt = y[tem]
                if np.std(pred) == 0 or np.std(yt) == 0:
                    continue
                r, p = stats.pearsonr(yt, pred)
                loso_rows.append(dict(mode=mode, impl=kind, study=s,
                                      n=int(tem.sum()), r_proj=float(r),
                                      p=float(p),
                                      bias=float(pred.mean() - yt.mean()),
                                      mae=float(mean_absolute_error(yt, pred))))

    if do_cv:
        pd.DataFrame(band_rows).to_csv(os.path.join(RES, f"part_band_{kind}.csv"), index=False)
    if do_loso:
        pd.DataFrame(loso_rows).to_csv(os.path.join(RES, f"part_loso_{kind}.csv"), index=False)
    print(f"[{kind}] 完成，{len(loso_rows)} 个研究 LOSO，用时 {time.time()-t0:.0f}s")
