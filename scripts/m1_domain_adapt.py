#!/usr/bin/env python3
"""
m1_domain_adapt.py — 域适配能否修复时钟的跨队列可迁移性？

问题：MetaPhlAn 时钟在全数据上 random CV R²≈0.49（复现原文），但投影到
QinN_2014 后 pred_age 与年龄几乎无关（r≈0.04）。原因很可能是物种丰度的
"研究特异基线"——不同研究的 DNA 提取、测序平台、队列饮食造成整体偏移，
线性模型在训练分布外失效。

本脚本比较 4 种处理，看能否把 QinN 的 r 从 0 拉起来：

  A. 原始 log 丰度（基线）
  B. 研究内 z-score（每个研究内部对每个物种标准化）——最简单的域适配
  C. 研究内分位数归一化（rank → 正态分数）
  D. 训练时加入研究固定效应（study-wise centering + 保留研究均值作为协变量）

评估口径：留一研究交叉验证（leave-one-study-out）。对每个留出的研究，
用该研究自身的均值/标准差做变换（test-time adaptation，可用，但需在论文里声明）。

用法：
  python3 scripts/m1_domain_adapt.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/metaphlan")
OUT = Path("results/m1_metaphlan")
PSEUDO = 1e-6


def load():
    frames = []
    for f in sorted((DATA / "per_study").glob("*.tsv.gz")):
        d = pd.read_csv(f, sep="\t", index_col=0, compression="infer")
        d.index = d.index.astype(str)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d.apply(pd.to_numeric, errors="coerce").fillna(0.0))
    X = pd.concat(frames, axis=0, join="outer").fillna(0.0)
    X = X[~X.index.duplicated()]
    meta = pd.read_csv(DATA / "sample_metadata.tsv.gz", sep="\t", index_col=0)
    meta.index = meta.index.astype(str)
    meta.columns = [c.lower() for c in meta.columns]
    common = X.index.intersection(meta.index)
    X, meta = X.loc[common], meta.loc[common]
    meta["age"] = pd.to_numeric(meta.get("age"), errors="coerce")
    return X, meta


def to_log(X):
    return np.log10(X.clip(lower=0) + PSEUDO)


def zscore_by_study(L: pd.DataFrame, study: pd.Series):
    """研究内逐个物种 z-score（跨样本标准化）。"""
    out = L.copy()
    for s in study.unique():
        idx = study.index[study == s]
        sub = L.loc[idx]
        mu, sd = sub.mean(0), sub.std(0, ddof=0)
        sd = sd.replace(0, np.nan).fillna(1.0)
        out.loc[idx] = (sub - mu) / sd
    return out.fillna(0.0)


def rank_by_study(L: pd.DataFrame, study: pd.Series):
    """研究内分位数归一化 → 正态分数。"""
    out = L.copy()
    for s in study.unique():
        idx = study.index[study == s]
        sub = L.loc[idx]
        r = sub.rank(axis=0, pct=True)
        out.loc[idx] = pd.DataFrame(
            stats.norm.ppf(r.clip(1e-4, 1 - 1e-4)), index=sub.index, columns=sub.columns)
    return out.fillna(0.0)


def loso_eval(F: pd.DataFrame, y: pd.Series, study: pd.Series, min_n=25):
    """留一研究交叉验证，返回每研究 r / MAE。"""
    rows = []
    model = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 12)))
    for s in study.unique():
        te = study.index[study == s]
        tr = study.index[(study != s) & y.notna()]
        if len(te) < min_n or y.loc[te].notna().sum() < min_n:
            continue
        m = model().fit(F.loc[tr], y.loc[tr])
        p = m.predict(F.loc[te])
        a = y.loc[te].to_numpy()
        ok = np.isfinite(a)
        r = stats.pearsonr(a[ok], p[ok])[0] if ok.sum() > 3 else np.nan
        rows.append({"study": s, "n": int(ok.sum()), "r": r,
                     "MAE": float(np.mean(np.abs(p[ok] - a[ok])))})
    return pd.DataFrame(rows)


def main():
    X, meta = load()
    L = to_log(X)
    y = meta["age"]
    study = meta["study"].astype(str)

    # 特征筛选：与 m1_metaphlan_mbag.py 保持一致
    prev = (X > 0).mean(0)
    keep = prev[prev >= 0.10].index
    var = L[keep].var(0)
    keep = var[var >= var.quantile(0.10)].index.tolist()
    print(f"[特征] {len(keep)} 个物种\n")

    variants = {
        "A 原始 log 丰度": L[keep],
        "B 研究内 z-score": zscore_by_study(L[keep], study),
        "C 研究内分位数归一化": rank_by_study(L[keep], study),
    }

    lines = ["# 域适配实验：能否修复时钟的跨队列可迁移性\n"]
    def say(s=""):
        print(s); lines.append(s)

    summary = []
    for name, F in variants.items():
        res = loso_eval(F, y, study)
        res.to_csv(OUT / f"domain_adapt_{name[0]}.csv", index=False)
        q = res[res["study"] == "QinN_2014"]
        med = res["r"].median()
        n04 = (res["r"] > 0.4).sum()
        line = (f"| {name} | {med:+.3f} | {n04}/{len(res)} | "
                f"{q['r'].iloc[0]:+.3f} | {q['MAE'].iloc[0]:.1f} |" if not q.empty else
                f"| {name} | {med:+.3f} | {n04}/{len(res)} | — | — |")
        summary.append(line)
        say(f"[{name}] 中位 r={med:+.3f}，r>0.4 的研究 {n04}/{len(res)}"
            + (f"，QinN_2014 r={q['r'].iloc[0]:+.3f} MAE={q['MAE'].iloc[0]:.1f}" if not q.empty else ""))

    # ---- 若 B 有效，用它重算 QinN 的年龄匹配效应 ----
    say("\n## 汇总（留一研究交叉验证）\n")
    say("| 处理 | 中位 r | r>0.4 的研究 | QinN_2014 r | QinN MAE |")
    say("|---|---|---|---|---|")
    for l in summary:
        say(l)

    F = variants["B 研究内 z-score"]
    s = "QinN_2014"
    te = study.index[study == s]
    tr = study.index[(study != s) & y.notna()]
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 12))).fit(F.loc[tr], y.loc[tr])
    pred = pd.Series(m.predict(F.loc[te]), index=te)
    g = meta.loc[te].copy()
    g["pred_age"] = pred
    lab = g["study_condition"].astype(str).str.lower()
    g["case"] = lab.str.contains("cirrhos|liver|hepat|nafld|nash")
    g = g.dropna(subset=["age", "pred_age"])

    say(f"\n## 域适配后 QinN_2014 的年龄匹配效应（z-score 版）\n")
    r_in = stats.pearsonr(g["age"], g["pred_age"])
    say(f"- 队列内 pred_age~age：r={r_in[0]:+.3f}（p={r_in[1]:.2g}）")

    case, ctrl = g[g["case"]], g[~g["case"]]
    pool, used, pairs = list(ctrl.index), set(), []
    for i, row in case.sort_values("age").iterrows():
        best, bd = None, np.inf
        for j in pool:
            if j in used:
                continue
            dd = abs(float(ctrl.loc[j, "age"]) - float(row["age"]))
            if dd < bd:
                best, bd = j, dd
        if best is not None and bd <= 5.0:
            used.add(best); pairs.append((i, best))
    if pairs:
        mc = np.array([g.loc[a, "pred_age"] for a, _ in pairs])
        mk = np.array([g.loc[b_, "pred_age"] for _, b_ in pairs])
        d = (mc.mean() - mk.mean()) / np.sqrt((mc.var(ddof=1) + mk.var(ddof=1)) / 2)
        p = stats.wilcoxon(mc, mk).pvalue
        say(f"- 年龄匹配 {len(pairs)} 对：病例 {mc.mean():.2f} vs 对照 {mk.mean():.2f} 岁，"
            f"差 {mc.mean()-mk.mean():+.2f}，d={d:+.2f}，p={p:.2g}")
        say(f"- 判读：{'域适配后仍无显著差异' if p >= 0.05 else ('显著更老' if d > 0 else '显著更年轻')}\n")

    (OUT / "domain_adapt_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n→ {OUT/'domain_adapt_report.md'}")


if __name__ == "__main__":
    main()
