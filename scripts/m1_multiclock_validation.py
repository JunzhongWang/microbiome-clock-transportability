#!/usr/bin/env python3
"""M1 补充实验：跨时钟实现的年龄分层失效验证

目的
----
前序工作（m1_age_stratified_clock.py）只在 Ridge 与 GBDT 两种实现上发现
"全跨度 R²≈0.48、40-70 岁带内 R²≈0.12" 的落差。单一实现无法排除
"是我们选的模型不行"，因此本脚本复刻已发表时钟所用的全部主流算法族，
检验该落差是否跨实现成立。

实现清单（6 种，覆盖 4 个算法族 × 2 个分类层级）
------------------------------------------------
  EN       Elastic Net       物种级   Galkin 2020 对比模型之一
  RF       Random Forest     物种级   Galkin 2020 对比模型之一
  GBM      Gradient Boosting 物种级   Galkin 2020 对比模型之一
  DNN      深度神经网络       物种级   Galkin 2020 报告的最优架构
  SVR      RBF 核支持向量机   物种级   核方法，独立的归纳偏置
  RFg      Random Forest     属级     Wilmanski 2021 的建模范式

  参考：Galkin 2020 (iScience 23:101199) 对 1165 份健康样本比较了
  Elastic Net / Random Forest / Gradient Boosting / Neural Network，
  最优 DNN 外部验证 MAE = 5.91 年。

协议
----
  P1 随机 5 折 CV（按年龄分箱分层）        —— 与已发表做法一致
  P2 按研究分组 5 折 CV（GroupKFold）      —— 跨研究泛化的诚实估计
  P3 留一研究投影（LOSO）                  —— 跨队列可迁移性
  P4 跨带迁移（全跨度/成人→40-70 岁）      —— 已发表时钟能否直接用于肝病人群

输出
----
  results/m1_multiclock/band_performance.csv
  results/m1_multiclock/loso_projection.csv
  results/m1_multiclock/summary.csv
  results/m1_multiclock/interclock_concordance.csv
"""
import glob
import os
import pickle
import sys
import time
import warnings

# ---- 线程限制：必须在 numpy/sklearn 导入前设置，否则不生效 ----
# n_jobs=-1 会占满全部 8 核（活动监视器 ~700% CPU，风扇全速）。
# 限制为 4 线程：RF/ElasticNetCV 结果与 n_jobs 无关（树序列由 random_state 决定），
# 仅降低并行度（约慢 40-60%），但整机占用从 ~700% 降到 ~400%。
MAX_JOBS = int(os.environ.get("M1_MAX_JOBS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(MAX_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(MAX_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(MAX_JOBS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(MAX_JOBS))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/metaphlan")
OUT = os.path.join(ROOT, "results/m1_multiclock")
os.makedirs(OUT, exist_ok=True)
SEED = 42
MIN_PREVALENCE = 0.10
VAR_PERCENTILE = 10
PSEUDO = 1e-6
N_SPLITS = 5
BOOT = 400
MIN_STUDY_N = 20      # LOSO 只对 n>=20 的研究做
MIN_BAND_N = 150      # 年龄带 CV 的最小样本量

# 已发表参考值（用于核对实现是否"忠实"）
PUBLISHED = {
    "R2_metabolism2027": 0.444,   # 项目锚点文献报告值
    "MAE_galkin2020": 5.91,       # Galkin 2020 外部验证 MAE
}


# ------------------------------------------------------------------ 数据
def load_panel():
    """返回 (样本×物种相对丰度 DataFrame, 元数据 DataFrame)"""
    cache = os.path.join(DATA, "_panel_cache.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            X, meta = pickle.load(fh)
        print(f"[缓存] {X.shape[0]} 样本 × {X.shape[1]} 物种")
        return X, meta

    meta = pd.read_csv(os.path.join(DATA, "sample_metadata.tsv.gz"), sep="\t",
                       index_col=0, compression="infer")
    meta.index = meta.index.astype(str)
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA, "per_study", "*.tsv.gz"))):
        study = os.path.basename(p).replace(".tsv.gz", "")
        df = pd.read_csv(p, sep="\t", index_col=0)
        # 复合键：不同研究存在 sampleID 撞号（如 LeChatelierE_2013 与
        # NielsenHB_2014 都用 MH0001..MH0177），裸 ID 对齐会错配 177 例
        df.index = [f"{study}::{str(i).strip()}" for i in df.index]
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df.apply(pd.to_numeric, errors="coerce").fillna(0.0))
    X = pd.concat(frames, axis=0, sort=False)
    X = X[~X.index.duplicated(keep="first")]

    key = meta["study"].astype(str) + "::" + meta.index.astype(str)
    meta = meta.copy()
    meta["_key"] = key.to_numpy()
    common = X.index.intersection(pd.Index(meta["_key"]))
    X = X.loc[common].fillna(0.0)
    meta = meta.set_index("_key").loc[common]
    X = X.loc[:, X.sum(axis=0) > 0]
    with open(cache, "wb") as fh:
        pickle.dump((X, meta), fh)
    print(f"[载入] {X.shape[0]} 样本 × {X.shape[1]} 物种")
    return X, meta


def to_genus(X):
    """s__Bacteroides_vulgatus -> g__Bacteroides；按属求和

    防御：非物种行（k__/p__/u__ 等门纲目科层级或未注释行）不能按
    split("_")[0] 映射——那会产出 "g__k__Bacteria" 这类伪属并污染求和。
    当前 curatedMetagenomicData 物种表无此类列（已验证 2022 列全部为
    s__/t__ 前缀），此处仍保留防御，避免未来数据口径变化时静默出错。
    """
    mapping = {}
    for c in X.columns:
        s = str(c)
        if s.startswith(("s__", "t__")):
            for pre in ("s__", "t__"):
                if s.startswith(pre):
                    s = s[len(pre):]
                    break
            genus = s.split("_")[0]
        else:
            # 保留原标签作为独立分组，不并入任何属
            genus = str(c)
        mapping[c] = "g__" + genus
    G = X.T.groupby(pd.Series(mapping)).sum().T
    return G


def select_features(X):
    """复刻已发表口径：出现率 >=10%，再保留方差前 90%

    注（已知局限，刻意保留）：筛选在全部健康样本上进行，未按 CV 折划分，
    严格意义上存在轻度信息泄漏（测试折的 prevalence/variance 参与了筛选）。
    这与被复现的已发表流程一致（原文同样在全数据上筛选后做 CV），
    为保证"忠实复现"而保留；据此得到的跨带对比结论不受影响，但绝对值
    可能略偏乐观。
    """
    prev = (X > 0).mean(axis=0)
    X1 = X.loc[:, prev >= MIN_PREVALENCE]
    if X1.shape[1] == 0:
        X1 = X
    var = X1.var(axis=0)
    thr = np.percentile(var, VAR_PERCENTILE)
    X2 = X1.loc[:, var >= thr]
    return X2 if X2.shape[1] > 0 else X1


def log_transform(X):
    return np.log10(np.asarray(X, dtype=float) + PSEUDO)


# ------------------------------------------------------------------ 模型
def make_model(kind):
    if kind == "EN":
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0], cv=3,
                         n_jobs=MAX_JOBS, max_iter=3000, random_state=SEED))
    if kind == "RF":
        return RandomForestRegressor(n_estimators=400, max_features=0.33,
                                     n_jobs=MAX_JOBS, random_state=SEED)
    if kind == "GBM":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=15, random_state=SEED)
    if kind == "DNN":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                         solver="adam", alpha=1e-4, batch_size=64,
                         learning_rate_init=1e-3, max_iter=400,
                         early_stopping=True, n_iter_no_change=15,
                         validation_fraction=0.1, random_state=SEED))
    if kind == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale",
                                                   epsilon=1.0, cache_size=800))
    if kind == "RFg":
        return RandomForestRegressor(n_estimators=400, max_features=0.5,
                                     n_jobs=MAX_JOBS, random_state=SEED)
    raise ValueError(kind)


IMPLS = ["EN", "RF", "GBM", "DNN", "SVR", "RFg"]
SPACE = {"EN": "species", "RF": "species", "GBM": "species",
         "DNN": "species", "SVR": "species", "RFg": "genus"}
FAMILY = {"EN": "线性（L1+L2）", "RF": "装袋树", "GBM": "提升树",
          "DNN": "神经网络", "SVR": "核方法", "RFg": "装袋树·属级"}
LABEL = {"EN": "Elastic Net（物种）", "RF": "Random Forest（物种）",
         "GBM": "Gradient Boosting（物种）", "DNN": "DNN 256-128-64（物种）",
         "SVR": "SVR-RBF（物种）", "RFg": "Random Forest（属级）"}


# ------------------------------------------------------------------ 评估
def metrics(y, pred, tag):
    r, p = stats.pearsonr(y, pred) if np.std(pred) > 0 else (0.0, 1.0)
    return dict(tag=tag, n=int(len(y)), r2=float(r2_score(y, pred)),
                mae=float(mean_absolute_error(y, pred)),
                r=float(r), p=float(p),
                resid_sd=float(np.std(y - pred, ddof=1)),
                pred_sd=float(np.std(pred, ddof=1)))


def boot_r2_ci(y, pred, n=BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        yy = y[idx]
        if np.std(yy) == 0:
            continue
        vals.append(r2_score(yy, pred[idx]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def cv_oof(Xl, y, groups, scheme):
    """返回 OOF 预测；scheme ∈ {random, group}"""
    n = len(y)
    pred = np.full(n, np.nan)
    if scheme == "random":
        try:
            bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        except Exception:
            bins = np.zeros(n, dtype=int)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(Xl, bins))
    else:
        gkf = GroupKFold(n_splits=N_SPLITS)
        splits = list(gkf.split(Xl, y, groups))
    for tr, te in splits:
        m = make_model(CUR_KIND).fit(Xl[tr], y[tr])
        pred[te] = m.predict(Xl[te])
    return pred


def fit_predict(kind, Xtr, ytr, Xte):
    m = make_model(kind).fit(Xtr, ytr)
    return m.predict(Xte)


# ------------------------------------------------------------------ 主流程
def main():
    global CUR_KIND

    X, meta = load_panel()
    meta.columns = [c.lower() for c in meta.columns]
    age = pd.to_numeric(meta["age"], errors="coerce")
    lab = meta["study_condition"].astype(str).str.lower()
    dis = meta["disease"].astype(str).str.lower()
    study = meta["study"].astype(str)

    MASKS = {
        "B_strict": age.notna() & dis.eq("healthy"),
        "A_loose": age.notna() & lab.str.contains("healthy|control", na=False),
    }

    # ---------- 预构建特征矩阵 ----------
    print("\n[构建特征空间]")
    feats = {}
    for mode, keep in MASKS.items():
        Xs = select_features(X.loc[keep])
        Xg = select_features(to_genus(X.loc[keep]))
        feats[mode] = {"species": (log_transform(Xs), Xs.shape[1]),
                       "genus": (log_transform(Xg), Xg.shape[1])}
        print(f"  {mode}: 物种 {Xs.shape[1]} 特征 / 属 {Xg.shape[1]} 特征, "
              f"n={int(keep.sum())}")

    band_rows, loso_rows, summary_rows = [], [], []
    oof_store = {}

    for mode in ["B_strict", "A_loose"]:
        keep = MASKS[mode]
        y = age.loc[keep].to_numpy(dtype=float)
        g = study.loc[keep].to_numpy()
        bands = {
            "全跨度 0-107": np.ones(len(y), bool),
            "儿童 <18": y < 18,
            "青年 18-40": (y >= 18) & (y < 40),
            "中年 40-70": (y >= 40) & (y <= 70),
            "老年 60+": y >= 60,
            "成人 18+": y >= 18,
        }
        schemes = ["random", "group"] if mode == "B_strict" else ["random"]

        for kind in IMPLS:
            Xl, nf = feats[mode][SPACE[kind]]
            CUR_KIND = kind
            t0 = time.time()
            print(f"\n[{mode} | {kind} | {nf} 特征]")

            for scheme in schemes:
                for btag, bm in bands.items():
                    if bm.sum() < MIN_BAND_N:
                        continue
                    yb, Xb, gb = y[bm], Xl[bm], g[bm]
                    pred = cv_oof(Xb, yb, gb, scheme)
                    mt = metrics(yb, pred, btag)
                    lo, hi = boot_r2_ci(yb, pred)
                    row = dict(mode=mode, impl=kind, family=FAMILY[kind],
                               cv=scheme, band=btag, n_feat=nf,
                               age_sd=round(float(np.std(yb, ddof=1)), 1),
                               r2_lo=round(lo, 3), r2_hi=round(hi, 3), **mt)
                    band_rows.append(row)
                    if mode == "B_strict" and scheme == "random" and btag == "全跨度 0-107":
                        oof_store[kind] = pred.copy()
                    print(f"   {scheme:6s} {btag:<12} n={mt['n']:<5} "
                          f"R²={mt['r2']:6.3f} [{lo:5.3f},{hi:5.3f}] "
                          f"MAE={mt['mae']:5.2f} r={mt['r']:6.3f}")

            # ---------- P4 跨带迁移（仅在严格口径、随机 CV 下） ----------
            # 注意：训练集与测试带必须严格不相交。早期版本曾用
            # ("成人→40-70", y>=18, 40-70) —— 训练集包含测试带，属样本内
            # 评估，R² 虚高（RF 曾达 0.67），是已修正的方法学错误。
            if mode == "B_strict":
                non_mid = (y < 40) | (y > 70)
                mid = (y >= 40) & (y <= 70)
                trans = [
                    ("非中年→40-70", non_mid, mid),
                    ("儿童→成人", y < 18, y >= 18),
                    ("中年→非中年", mid, non_mid),
                ]
                for ttag, trm, tem in trans:
                    if trm.sum() < MIN_BAND_N or tem.sum() < MIN_BAND_N:
                        continue
                    assert not np.any(trm & tem), f"迁移实验训练/测试集重叠: {ttag}"
                    pred = fit_predict(kind, Xl[trm], y[trm], Xl[tem])
                    mt = metrics(y[tem], pred, ttag)
                    lo, hi = boot_r2_ci(y[tem], pred)
                    band_rows.append(dict(mode=mode, impl=kind, family=FAMILY[kind],
                                          cv="transfer", band=ttag, n_feat=nf,
                                          age_sd=round(float(np.std(y[tem], ddof=1)), 1),
                                          r2_lo=round(lo, 3), r2_hi=round(hi, 3), **mt))
                    print(f"   transfer {ttag:<12} n={mt['n']:<5} "
                          f"R²={mt['r2']:7.3f} MAE={mt['mae']:5.2f}")

            # ---------- P3 留一研究投影（严格口径） ----------
            if mode == "B_strict":
                for s in pd.unique(g):
                    tem = g == s
                    trm = ~tem
                    if tem.sum() < MIN_STUDY_N or trm.sum() < 500:
                        continue
                    pred = fit_predict(kind, Xl[trm], y[trm], Xl[tem])
                    yt = y[tem]
                    if np.std(pred) == 0 or np.std(yt) == 0:
                        continue
                    r, p = stats.pearsonr(yt, pred)
                    loso_rows.append(dict(mode=mode, impl=kind, study=s,
                                          n=int(tem.sum()), r_proj=float(r),
                                          p=float(p),
                                          bias=float(pred.mean() - yt.mean()),
                                          mae=float(mean_absolute_error(yt, pred))))
                print(f"   LOSO 完成 {len([r for r in loso_rows if r['impl']==kind])} 个研究"
                      f" | 用时 {time.time()-t0:.0f}s")

    band = pd.DataFrame(band_rows)
    loso = pd.DataFrame(loso_rows)

    # ---- 防覆盖保护：正式结果由 m1_multiclock_run.py 分批产出 ----
    # 若已有正式结果且与本脚本的迁移设计不同，拒绝覆盖（提示先备份）。
    final_band = os.path.join(OUT, "band_performance.csv")
    if os.path.exists(final_band):
        old = pd.read_csv(final_band)
        old_trans = set(old.loc[old.cv == "transfer", "band"].unique())
        new_trans = set(band.loc[band.cv == "transfer", "band"].unique())
        if old_trans and old_trans != new_trans:
            raise RuntimeError(
                "检测结果冲突：已存在的 band_performance.csv 的迁移方向 "
                f"{sorted(old_trans)} 与本脚本 {sorted(new_trans)} 不同。\n"
                "正式结果由 m1_multiclock_run.py 产出；如确要重算，"
                "请先备份旧结果再删除该文件。")

    band.to_csv(final_band, index=False)
    loso.to_csv(os.path.join(OUT, "loso_projection.csv"), index=False)

    # ---------- 汇总表 ----------
    for kind in IMPLS:
        b = band[(band.impl == kind) & (band.mode == "B_strict")]
        rnd = b[b.cv == "random"].set_index("band")
        grp = b[b.cv == "group"].set_index("band")
        l = loso[loso.impl == kind]
        gate = (l.r_proj >= 0.3).mean() if len(l) else np.nan
        summary_rows.append(dict(
            impl=kind, family=FAMILY[kind], space=SPACE[kind],
            n_feat=int(rnd.loc["全跨度 0-107", "n_feat"]) if "全跨度 0-107" in rnd.index else np.nan,
            r2_full=rnd.loc["全跨度 0-107", "r2"] if "全跨度 0-107" in rnd.index else np.nan,
            mae_full=rnd.loc["全跨度 0-107", "mae"] if "全跨度 0-107" in rnd.index else np.nan,
            r2_40_70=rnd.loc["中年 40-70", "r2"] if "中年 40-70" in rnd.index else np.nan,
            mae_40_70=rnd.loc["中年 40-70", "mae"] if "中年 40-70" in rnd.index else np.nan,
            r2_child=rnd.loc["儿童 <18", "r2"] if "儿童 <18" in rnd.index else np.nan,
            r2_full_group=grp.loc["全跨度 0-107", "r2"] if "全跨度 0-107" in grp.index else np.nan,
            r2_40_70_group=grp.loc["中年 40-70", "r2"] if "中年 40-70" in grp.index else np.nan,
            delta=rnd.loc["全跨度 0-107", "r2"] - rnd.loc["中年 40-70", "r2"],
            loso_r_median=float(l.r_proj.median()) if len(l) else np.nan,
            loso_r_q25=float(l.r_proj.quantile(0.25)) if len(l) else np.nan,
            loso_r_q75=float(l.r_proj.quantile(0.75)) if len(l) else np.nan,
            gate_pass=float(gate),
            bias_sd=float(l.bias.std(ddof=1)) if len(l) > 2 else np.nan,
            bias_absmax=float(l.bias.abs().max()) if len(l) else np.nan,
        ))
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # ---------- 实现间一致性 ----------
    conc = []
    keys = list(oof_store)
    y_full = age.loc[MASKS["B_strict"]].to_numpy(dtype=float)
    for i, a in enumerate(keys):
        for b_ in keys[i + 1:]:
            pa, pb = oof_store[a], oof_store[b_]
            ok = np.isfinite(pa) & np.isfinite(pb)
            if ok.sum() < 100:
                continue
            r, _ = stats.pearsonr(pa[ok], pb[ok])
            conc.append(dict(impl_a=a, impl_b=b_, r=float(r), n=int(ok.sum())))
    pd.DataFrame(conc).to_csv(os.path.join(OUT, "interclock_concordance.csv"), index=False)

    # ---------- 控制台汇总 ----------
    print("\n" + "=" * 100)
    print("跨实现汇总（严格健康口径，随机 5 折 CV）")
    print("=" * 100)
    hdr = f"{'实现':<28}{'特征':>5}{'R²全跨度':>10}{'MAE':>7}{'R²40-70':>9}{'ΔR²':>8}{'LOSO中位r':>11}{'门通过':>8}{'bias SD':>9}"
    print(hdr)
    print("-" * 100)
    for _, r in summ.iterrows():
        print(f"{LABEL[r.impl]:<28}{int(r.n_feat):>5}{r.r2_full:>10.3f}{r.mae_full:>7.2f}"
              f"{r.r2_40_70:>9.3f}{r.delta:>8.3f}{r.loso_r_median:>11.3f}"
              f"{r.gate_pass*100:>7.0f}%{r.bias_sd:>9.2f}")
    print("=" * 100)
    print(f"\n输出目录: {OUT}")
    return summ


CUR_KIND = "EN"

if __name__ == "__main__":
    main()
