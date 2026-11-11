#!/usr/bin/env python3
"""
m1_metaphlan_mbag.py — M1「零下载」路线：MetaPhlAn 口径的菌群衰老时钟 + mBAG 计算

设计要点（为什么这样做，见 docs/M1零下载流程.md）：
  1. 训练与投影使用同一套定量口径（MetaPhlAn relative_abundance），避免 UHGG(训练)
     vs MetaPhlAn(投影) 的口径混用导致的假阳性；
  2. 主分析单位是「同一研究内的病例 vs 对照 mBAG 差」，而不是跨研究直接比较绝对值
     ——这是消除研究批次效应最稳妥的做法；
  3. 跨研究结果用随机效应（DerSimonian-Laird）meta 分析合并，报告 I²。

子命令：
  train    训练时钟（两种 CV：随机分层 / 按研究分组）
  project  投影到肝病队列，算 mBAG，做研究内对比 + meta 分析
  demo     用合成数据跑通全流程（无真实数据时验证代码）

用法：
  python3 scripts/m1_metaphlan_mbag.py demo
  python3 scripts/m1_metaphlan_mbag.py train   --data data/metaphlan --out results/m1_metaphlan
  python3 scripts/m1_metaphlan_mbag.py project --data data/metaphlan --out results/m1_metaphlan
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
TARGET_COL = "age"
GROUP_COL = "study"
LABEL_COL = "study_condition"
COUNTRY_COL = "country"
PSEUDO = 1e-6          # log 变换的伪计数（相对丰度量级）
MIN_PREVALENCE = 0.10  # 物种流行率门槛
VAR_PERCENTILE = 10    # 方差过滤分位数


# ----------------------------------------------------------------------------
# 数据载入
# ----------------------------------------------------------------------------
def read_study_table(path: Path) -> pd.DataFrame:
    """读单个研究的宽表（行=样本，列=物种），第一列为 sampleID。"""
    df = pd.read_csv(path, sep="\t", index_col=0, compression="infer")
    df.index = df.index.astype(str)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_panel(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """载入 R 导出的全部研究丰度表 + 元数据，拼成一个大矩阵（列的并集）。"""
    per_study = sorted(glob.glob(str(data_dir / "per_study" / "*.tsv.gz")))
    meta_path = data_dir / "sample_metadata.tsv.gz"
    if not per_study:
        sys.exit(f"[错误] 在 {data_dir/'per_study'} 找不到任何 *.tsv.gz，请先运行 m1_fetch_metaphlan.R")
    if not meta_path.exists():
        sys.exit(f"[错误] 找不到 {meta_path}，请先运行 m1_fetch_metaphlan.R")

    meta = pd.read_csv(meta_path, sep="\t", index_col=0, compression="infer")
    meta.index = meta.index.astype(str)

    frames = []
    for p in per_study:
        df = read_study_table(Path(p))
        frames.append(df)
        hit = meta.index.intersection(df.index)
        if len(hit):
            meta.loc[hit, "source_file"] = Path(p).name
    X = pd.concat(frames, axis=0, sort=False)
    # 只保留有元数据的样本
    common = X.index.intersection(meta.index)
    if len(common) < len(X):
        print(f"[提示] {len(X)-len(common)} 个样本缺元数据，已剔除")
    X, meta = X.loc[common], meta.loc[common]
    # 去掉全零列并补零
    X = X.fillna(0.0)
    X = X.loc[:, X.sum(axis=0) > 0]
    print(f"[载入] {X.shape[0]} 样本 × {X.shape[1]} 物种，来自 {len(per_study)} 个研究文件")
    return X, meta


def select_features(X: pd.DataFrame) -> pd.DataFrame:
    """流行率 + 方差过滤，返回过滤后的矩阵。"""
    prev = (X > 0).mean(axis=0)
    X1 = X.loc[:, prev >= MIN_PREVALENCE]
    if X1.shape[1] == 0:
        sys.exit("[错误] 流行率过滤后没有剩余物种，检查输入是否为空")
    var = X1.var(axis=0)
    thr = np.percentile(var, VAR_PERCENTILE)
    X2 = X1.loc[:, var >= thr]
    print(f"[特征] 流行率≥{MIN_PREVALENCE:.0%}: {X1.shape[1]} 个 → 方差≥P{VAR_PERCENTILE}: {X2.shape[1]} 个")
    return X2


def log_transform(X: pd.DataFrame) -> np.ndarray:
    return np.log10(np.asarray(X, dtype=float) + PSEUDO)


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
def make_model(kind: str = "ridge"):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor

    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    if kind == "gbdt":
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, random_state=42
        )
    raise ValueError(kind)


def _oof_predict(make, X, y, splits):
    """按给定 split 做折外预测（自己实现，避免 StratifiedKFold 不接受连续型 y）。"""
    pred = np.full(len(y), np.nan, dtype=float)
    for tr, te in splits:
        m = make().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def cv_report(X, y, groups, kind="ridge"):
    """两种 CV 口径各跑一遍：随机分层（乐观）与按研究分组（诚实）。"""
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy.stats import pearsonr

    rows = []
    bins = pd.cut(y, bins=[-1, 20, 35, 50, 65, 200], labels=False).astype(int)

    schemes = [
        ("random (StratifiedKFold by age bin)",
         StratifiedKFold(5, shuffle=True, random_state=42).split(X, bins)),
        ("grouped (GroupKFold by study)",
         GroupKFold(5).split(X, y, groups)),
    ]
    for name, splits in schemes:
        pred = _oof_predict(lambda: make_model(kind), X, y, splits)
        r = {
            "scheme": name,
            "model": kind,
            "n": int(len(y)),
            "r2": round(float(r2_score(y, pred)), 4),
            "mae": round(float(mean_absolute_error(y, pred)), 3),
            "pearson_r": round(float(pearsonr(y, pred)[0]), 4),
        }
        rows.append(r)
        print(f"  {name:42s} R²={r['r2']:+.3f}  MAE={r['mae']:5.2f}  r={r['pearson_r']:+.3f}")
    return pd.DataFrame(rows), pred


# ----------------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot=2000, seed=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        d = cohens_d(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
        if np.isfinite(d):
            ds.append(d)
    if not ds:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(ds, [2.5, 97.5]))


def random_effects_meta(effects, variances):
    """DerSimonian-Laird 随机效应合并。返回 dict。"""
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    if k == 0:
        return {"k": 0}
    w = 1.0 / v
    y_fe = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - y_fe) ** 2))
    dfree = k - 1
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 and k > 1 else 0.0
    wr = 1.0 / (v + tau2)
    y_re = float(np.sum(wr * y) / np.sum(wr))
    se = float(np.sqrt(1.0 / np.sum(wr)))
    I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 and k > 1 else 0.0
    return {
        "k": k,
        "pooled_d": round(y_re, 4),
        "se": round(se, 4),
        "ci_low": round(y_re - 1.96 * se, 4),
        "ci_high": round(y_re + 1.96 * se, 4),
        "p": round(float(2 * (1 - _norm_cdf(abs(y_re / se)))), 5) if se > 0 else float("nan"),
        "I2_percent": round(I2, 1) if k > 1 else None,
        "I2_note": None if k > 1 else "k=1，异质性无法估计",
        "tau2": round(tau2, 4),
        "Q": round(Q, 2),
    }


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def cmd_train(args):
    data_dir = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X, meta = load_panel(data_dir)
    meta.columns = [c.lower() for c in meta.columns]
    if TARGET_COL not in meta.columns:
        sys.exit(f"[错误] 元数据里没有 age 列，现有列：{list(meta.columns)[:15]}")

    age = pd.to_numeric(meta[TARGET_COL], errors="coerce")
    healthy = meta[LABEL_COL].astype(str).str.lower().str.contains("healthy|control", na=False) if LABEL_COL in meta.columns else pd.Series(True, index=meta.index)
    keep = age.notna() & healthy
    if keep.sum() < 200:
        sys.exit(f"[错误] 健康且有精确年龄的样本只有 {keep.sum()} 例，不足以训钟")

    Xtr = select_features(X.loc[keep])
    y = age.loc[keep].to_numpy(dtype=float)
    groups = meta.loc[keep, GROUP_COL].astype(str).to_numpy() if GROUP_COL in meta.columns else np.arange(keep.sum())
    print(f"[训练集] {Xtr.shape[0]} 例 × {Xtr.shape[1]} 物种，{len(set(groups))} 个研究，年龄 {y.min():.0f}–{y.max():.0f}")

    Xl = log_transform(Xtr)
    print("[CV] 两种验证口径：")
    tbl_r, _ = cv_report(Xl, y, groups, "ridge")
    tbl_g, _ = cv_report(Xl, y, groups, "gbdt")
    cvtab = pd.concat([tbl_r, tbl_g], ignore_index=True)
    cvtab.to_csv(out / "clock_cv.csv", index=False)
    print(f"  → {out/'clock_cv.csv'}")

    # 全量拟合最终模型（后续投影用 ridge：更稳、可解释、外推行为好）
    model = make_model("ridge").fit(Xl, y)
    import pickle
    with open(out / "clock_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": list(Xtr.columns)}, f)
    print(f"  → {out/'clock_model.pkl'}（{len(Xtr.columns)} 个特征）")

    # 训练集上的个体 BAG 分布留档（供投影时对比参考）
    pd.DataFrame({"sampleID": Xtr.index, "age": y, "pred": model.predict(Xl),
                  "bag": model.predict(Xl) - y}).to_csv(out / "train_bag.csv", index=False)
    return 0


# ----------------------------------------------------------------------------
# project
# ----------------------------------------------------------------------------
def cmd_project(args):
    import pickle

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clock_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, features = bundle["model"], bundle["features"]

    X, meta = load_panel(data_dir)
    meta.columns = [c.lower() for c in meta.columns]

    # ---- 1. 选出肝病研究：同研究内同时有病例与对照 ----
    if LABEL_COL not in meta.columns:
        sys.exit("[错误] 元数据缺少 study_condition 列，无法区分病例/对照")
    lab = meta[LABEL_COL].astype(str).str.lower()
    is_ctrl = lab.str.contains("healthy|control", na=False)
    is_case = lab.str.contains("cirrhos|nafld|nash|hepat|fibros|steato|hcc|hepatocellular|liver", na=False)
    meta = meta.assign(_case=is_case, _ctrl=is_ctrl)

    usable = []
    for s, g in meta.groupby(GROUP_COL):
        if g["_case"].sum() >= 10 and g["_ctrl"].sum() >= 5:
            usable.append(s)
    if not usable:
        sys.exit("[错误] 没有任何研究同时具备 ≥10 例病例与 ≥5 例对照，无法做研究内对比")
    print(f"[肝病队列] 可用于研究内对比的研究：{len(usable)} 个")
    sel = meta[meta[GROUP_COL].isin(usable) & (meta["_case"] | meta["_ctrl"])]

    # ---- 2. 特征对齐 ----
    avail = [c for c in features if c in X.columns]
    missing = len(features) - len(avail)
    print(f"[对齐] 训练特征 {len(features)} 个，投影队列可匹配 {len(avail)} 个（缺 {missing} 个，按 0 处理）")
    if len(avail) < 0.5 * len(features):
        print("[警告] 特征匹配率低于 50%，投影结果可能不可靠，请检查物种命名口径")

    Xp = X.loc[sel.index, avail].copy()
    for c in features:
        if c not in Xp.columns:
            Xp[c] = 0.0
    Xp = Xp[features]
    pred = model.predict(log_transform(Xp))

    res = sel[[GROUP_COL, LABEL_COL]].copy()
    res["pred_age"] = pred
    res["_case"] = sel["_case"].to_numpy()
    if TARGET_COL in meta.columns:
        res["age"] = pd.to_numeric(sel[TARGET_COL], errors="coerce")
        res["mbag"] = res["pred_age"] - res["age"]
    else:
        res["age"] = np.nan
    res.to_csv(out / "projected_samples.csv")

    # ---- 3. 研究内病例 vs 对照 ----
    rows = []
    for s, g in res.groupby(GROUP_COL):
        case = g.loc[g["_case"], "pred_age"].to_numpy()
        ctrl = g.loc[~g["_case"], "pred_age"].to_numpy()
        if len(case) < 10 or len(ctrl) < 5:
            continue
        # 若年龄可得，用 mBAG 对比；否则用预测年龄（同研究内年龄分布相近时等价）
        use_mbag = g["mbag"].notna().sum() > 0.8 * len(g)
        a = g.loc[g["_case"], "mbag" if use_mbag else "pred_age"].to_numpy(dtype=float)
        b = g.loc[~g["_case"], "mbag" if use_mbag else "pred_age"].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 5:
            continue
        from scipy.stats import mannwhitneyu
        d = cohens_d(a, b)
        lo, hi = bootstrap_ci(a, b)
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        n1, n2 = len(a), len(b)
        var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)) if np.isfinite(d) else np.nan
        rows.append({
            "study": s, "metric": "mBAG" if use_mbag else "pred_age",
            "n_case": n1, "n_ctrl": n2,
            "mean_case": round(float(np.mean(a)), 2), "mean_ctrl": round(float(np.mean(b)), 2),
            "diff": round(float(np.mean(a) - np.mean(b)), 2),
            "cohens_d": round(d, 3), "d_ci_low": round(lo, 3), "d_ci_high": round(hi, 3),
            "p_mannwhitney": p, "var_d": var_d,
        })

    per_study = pd.DataFrame(rows).sort_values("cohens_d", ascending=False)
    per_study.to_csv(out / "mbag_by_study.csv", index=False)
    print("\n[研究内对比]")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(per_study.drop(columns=["var_d"]).to_string(index=False))

    # ---- 4. 随机效应 meta 分析 ----
    # 只合并 mBAG：pred_age 与 mBAG 是两种不同指标，混在同一个 meta 里无效
    # （pred_age 未扣除年龄差异，无法解释为"衰老加速度"）。
    ok = per_study[per_study["metric"] == "mBAG"]
    skipped = per_study[per_study["metric"] != "mBAG"]
    if len(skipped):
        print(f"\n[注意] {len(skipped)} 个研究缺精确年龄，只能比较 pred_age，"
              f"不纳入 mBAG 主分析（已列在 mbag_by_study.csv 供参考）:")
        for _, r in skipped.iterrows():
            print(f"    {r['study']}: pred_age 差 {r['diff']:+.2f} 岁（无年龄→无法算 mBAG）")
    if len(ok) < 1:
        sys.exit("[错误] 没有任何研究能计算 mBAG，无法做 meta 分析")
    meta_res = random_effects_meta(ok["cohens_d"], ok["var_d"])
    with open(out / "meta_analysis.json", "w") as f:
        json.dump(meta_res, f, ensure_ascii=False, indent=2)
    print("\n[随机效应 meta 分析]")
    print(json.dumps(meta_res, ensure_ascii=False, indent=2))

    # ---- 5. 图 ----
    try:
        make_figures(res, per_study, out, meta_res)
    except Exception as e:  # 画图失败不应中断分析
        print(f"[提示] 绘图跳过：{e}")
    return 0


def make_figures(res: pd.DataFrame, per_study: pd.DataFrame, out: Path, meta_res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：各研究预测年龄（病例 vs 对照）
    studies = per_study["study"].tolist()
    data = [res.loc[res[GROUP_COL] == s, "pred_age"].to_numpy() for s in studies]
    try:  # matplotlib >=3.9 把 labels 改名为 tick_labels
        axes[0].boxplot(data, tick_labels=studies, showfliers=False)
    except TypeError:
        axes[0].boxplot(data, labels=studies, showfliers=False)
    axes[0].set_ylabel("Predicted microbiome age (years)")
    axes[0].set_title("Predicted age by study (all samples)")
    axes[0].tick_params(axis="x", rotation=30)

    # 右：森林图
    y = np.arange(len(per_study))
    axes[1].errorbar(per_study["cohens_d"], y,
                     xerr=[per_study["cohens_d"] - per_study["d_ci_low"],
                           per_study["d_ci_high"] - per_study["cohens_d"]],
                     fmt="o", color="#c0392b", capsize=3)
    axes[1].axvline(0, ls="--", c="grey", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(per_study["study"], fontsize=8)
    # 合并效应（菱形，黑色）
    if meta_res.get("k", 0) > 0:
        yy = len(per_study) - 0.6
        i2 = meta_res.get("I2_percent")
        lab = (f"pooled d={meta_res['pooled_d']:.2f}" if i2 is None
               else f"pooled d={meta_res['pooled_d']:.2f} (I²={i2:.0f}%)")
        axes[1].errorbar([meta_res["pooled_d"]], [yy],
                         xerr=[[meta_res["pooled_d"] - meta_res["ci_low"]],
                               [meta_res["ci_high"] - meta_res["pooled_d"]]],
                         fmt="D", color="black", capsize=4, markersize=7,
                         label=lab)
        axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_xlabel("Cohen's d (case vs control, within study)")
    axes[1].set_title("Microbiome age acceleration by cohort")
    fig.tight_layout()
    fig.savefig(out / "mbag_forest.png", dpi=200)
    print(f"  → {out/'mbag_forest.png'}")


# ----------------------------------------------------------------------------
# demo：合成数据跑通全流程
# ----------------------------------------------------------------------------
def cmd_demo(args):
    """合成数据：物种丰度携带明确的年龄信号，肝病病例额外"加速"15 年，
    用来验证「训钟 → 投影 → mBAG → meta 分析」整条链路是否给出预期方向。"""
    rng = np.random.default_rng(7)
    base = Path("results/m1_metaphlan_demo")
    data = base / "data"
    (data / "per_study").mkdir(parents=True, exist_ok=True)

    n_species, n_age_sp = 300, 60          # 前 60 个物种携带年龄信号
    base_log = rng.normal(-6, 1.0, n_species)
    slope = np.zeros(n_species)
    slope[:n_age_sp] = 0.08                # 每增加 1 岁，log 丰度 +0.08
    meta_rows = []

    def synth(study, n, age_mean, condition="healthy", accel=0.0, batch=0.0, tag=None):
        ages = rng.normal(age_mean, 12, n).clip(18, 90)
        eff = ages + accel
        logc = base_log[None, :] + slope[None, :] * (eff[:, None] - 50)
        logc[:, n_age_sp:] += batch        # 批次效应只作用在非年龄物种上
        logc += rng.normal(0, 0.35, logc.shape)   # 采样噪声
        p = np.exp(logc - logc.max(axis=1, keepdims=True))
        mat = p / p.sum(axis=1, keepdims=True)
        ids = [f"{tag or study}_{i}" for i in range(n)]
        pd.DataFrame(mat, index=ids, columns=[f"s__Species_{i:03d}" for i in range(n_species)]) \
            .to_csv(data / "per_study" / f"{tag or study}.tsv.gz", sep="\t", compression="gzip")
        for i, sid in enumerate(ids):
            meta_rows.append({"sampleID": sid, "study": study, "age": round(float(ages[i]), 1),
                              "study_condition": condition})

    # 三个健康研究（训钟用），各自带不同批次偏移
    for s, n, b in [("HealthyA", 220, 0.0), ("HealthyB", 200, 0.6), ("HealthyC", 240, -0.5)]:
        synth(s, n, 45, "healthy", batch=b)

    # 两个肝病研究：同一研究内含病例 + 对照，病例额外"衰老" 15 年
    for s, ncase, nctrl, label, b in [("CirrhosisX", 60, 45, "cirrhosis", 0.4),
                                      ("NAFLDY", 50, 40, "NAFLD", -0.3)]:
        synth(s, nctrl, 55, "healthy", batch=b, tag=f"{s}_ctrl")
        synth(s, ncase, 55, label, accel=15.0, batch=b, tag=f"{s}_case")

    pd.DataFrame(meta_rows).set_index("sampleID").to_csv(
        data / "sample_metadata.tsv.gz", sep="\t", compression="gzip")

    print("=" * 70)
    print("DEMO 数据已生成（病例被人为设置 +15 年菌群年龄加速）：", data)
    print("=" * 70)
    ns = argparse.Namespace(data=str(data), out=str(base / "out"))
    cmd_train(ns)
    print("\n" + "=" * 70)
    cmd_project(ns)
    print("\nDEMO 完成，产物在", base / "out")
    return 0


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="M1 零下载：MetaPhlAn 口径菌群衰老时钟 + mBAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, helptext in [
        ("train", cmd_train, "训练时钟"),
        ("project", cmd_project, "投影肝病队列 + mBAG + meta 分析"),
        ("demo", cmd_demo, "合成数据跑通全流程"),
    ]:
        p = sub.add_parser(name, help=helptext)
        if name != "demo":
            p.add_argument("--data", default="data/metaphlan", help="R 导出的数据目录")
            p.add_argument("--out", default="results/m1_metaphlan", help="输出目录")
        p.set_defaults(func=fn)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
