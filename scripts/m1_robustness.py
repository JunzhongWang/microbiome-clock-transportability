#!/usr/bin/env python3
"""
m1_robustness.py — 对 M1 主结果做混杂与稳健性检查

背景：真实数据下 QinN_2014（肝硬化 vs 对照）得到 mBAG 差 -11.46 岁
（d=-0.87, p=3.9e-10），即肝硬化患者微生物组看起来"更年轻"——与
"加速衰老"的直觉假设方向相反。这个结果无论如何都要先排除技术/混杂解释。

本脚本检查 4 件事：
  1. 投影校准：同一研究内对照组 mBAG 是否接近 0（若接近，说明跨队列投影本身无系统偏移）
  2. 年龄响应：mBAG 与真实年龄的关系（若强相关，说明时钟只是没学好年龄）
  3. 多样性混杂：Shannon 多样性在两组间的差异，以及 mBAG 对多样性的依赖
  4. 调整后效应：把 mBAG 对多样性做残差化后，组间差是否仍在

用法：
  python3 scripts/m1_robustness.py --study QinN_2014
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def shannon(X: pd.DataFrame) -> pd.Series:
    p = X.clip(lower=0).to_numpy(dtype=float)
    p = p / np.where(p.sum(1, keepdims=True) > 0, p.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return pd.Series(h, index=X.index)


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="QinN_2014")
    ap.add_argument("--data", default="data/metaphlan")
    ap.add_argument("--out", default="results/m1_metaphlan")
    args = ap.parse_args()

    out = Path(args.out)
    proj = pd.read_csv(out / "projected_samples.csv", index_col=0)
    sub = proj[proj["study"] == args.study].copy()
    if sub.empty:
        raise SystemExit(f"{args.study} 不在投影结果中")

    # 载入该研究的原始丰度（用于多样性）
    X = pd.read_csv(Path(args.data) / "per_study" / f"{args.study}.tsv.gz",
                    sep="\t", index_col=0, compression="infer")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    div = shannon(X)
    sub["shannon"] = div.reindex(sub.index)
    sub["case"] = sub["_case"].astype(bool)

    case = sub[sub["case"]]
    ctrl = sub[~sub["case"]]
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say(f"# {args.study}：M1 主结果稳健性检查\n")
    say(f"样本：病例 {len(case)} / 对照 {len(ctrl)}\n")

    # ---- 1. 校准 ----
    say("## 1. 投影校准（对照组 mBAG 是否接近 0）\n")
    say(f"- 对照 mBAG 均值 = **{ctrl['mbag'].mean():+.2f} 岁**（中位 {ctrl['mbag'].median():+.2f}）")
    say(f"- 病例 mBAG 均值 = **{case['mbag'].mean():+.2f} 岁**（中位 {case['mbag'].median():+.2f}）")
    say(f"- 结论：对照组偏差 {abs(ctrl['mbag'].mean()):.1f} 岁，"
        f"{'投影无明显系统偏移，差异可解释为条件效应' if abs(ctrl['mbag'].mean()) < 5 else '存在明显系统偏移，差异需谨慎解读'}\n")

    # ---- 2. 年龄响应 ----
    say("## 2. 时钟对真实年龄的响应\n")
    for nm, g in [("病例", case), ("对照", ctrl), ("合计", sub)]:
        r, p = stats.pearsonr(g["age"], g["pred_age"])
        slope = np.polyfit(g["age"], g["pred_age"], 1)[0]
        say(f"- {nm}：pred_age ~ age  r={r:+.3f} (p={p:.2g})，斜率 {slope:.2f}")
    say("")

    # ---- 3. 多样性混杂 ----
    say("## 3. Shannon 多样性\n")
    d_case, d_ctrl = case["shannon"].to_numpy(), ctrl["shannon"].to_numpy()
    d = cohens_d(d_case, d_ctrl)
    p_div = stats.mannwhitneyu(d_case, d_ctrl, alternative="two-sided").pvalue
    say(f"- 病例 {np.mean(d_case):.2f} vs 对照 {np.mean(d_ctrl):.2f}，"
        f"Cohen's d={d:+.2f}，p={p_div:.2g}")
    r_mb, p_mb = stats.pearsonr(sub["shannon"], sub["mbag"])
    say(f"- mBAG ~ Shannon：r={r_mb:+.3f} (p={p_mb:.2g})"
        f"{'  ← 强相关，多样性是潜在混杂' if abs(r_mb) > 0.4 else ''}\n")

    # ---- 4. 调整后效应 ----
    say("## 4. 对多样性做残差化后的组间差\n")
    s = sub.dropna(subset=["mbag", "shannon", "age"])
    # 用对照组拟合 mBAG ~ shannon（只用对照，避免把病例效应吸收进模型）
    c0 = s[~s["case"]]
    beta = np.polyfit(c0["shannon"], c0["mbag"], 1)
    s["mbag_adj"] = s["mbag"] - np.polyval(beta, s["shannon"])
    a1 = s.loc[s["case"], "mbag"].to_numpy()
    b1 = s.loc[~s["case"], "mbag"].to_numpy()
    a2 = s.loc[s["case"], "mbag_adj"].to_numpy()
    b2 = s.loc[~s["case"], "mbag_adj"].to_numpy()
    d_raw, d_adj = cohens_d(a1, b1), cohens_d(a2, b2)
    p_raw = stats.mannwhitneyu(a1, b1, alternative="two-sided").pvalue
    p_adj = stats.mannwhitneyu(a2, b2, alternative="two-sided").pvalue
    say(f"- 原始：差 {np.mean(a1)-np.mean(b1):+.2f} 岁，d={d_raw:+.2f}，p={p_raw:.2g}")
    say(f"- 多样性残差化后：差 {np.mean(a2)-np.mean(b2):+.2f} 岁，d={d_adj:+.2f}，p={p_adj:.2g}")
    shrink = (1 - d_adj / d_raw) * 100 if d_raw else float("nan")
    say(f"- 效应衰减 {shrink:.0f}%"
        f"{'  ← 大部分效应可由多样性差异解释' if shrink > 50 else '  ← 多样性不足以解释，效应独立存在'}\n")

    # ---- 5. 同时调整年龄与多样性 ----
    say("## 5. 多变量线性模型（mBAG ~ 病例 + 年龄 + 多样性）\n")
    try:
        import statsmodels.api as sm
        Xd = pd.DataFrame({
            "case": s["case"].astype(float),
            "age": s["age"],
            "shannon": s["shannon"],
        })
        Xd = sm.add_constant(Xd)
        m = sm.OLS(s["mbag"], Xd).fit()
        say(f"- 病例系数 = **{m.params['case']:+.2f} 岁** (95% CI "
            f"{m.conf_int().loc['case',0]:+.2f} ~ {m.conf_int().loc['case',1]:+.2f}, "
            f"p={m.pvalues['case']:.2g})")
        say(f"- 年龄系数 = {m.params['age']:+.2f}，多样性系数 = {m.params['shannon']:+.2f}，"
            f"R²={m.rsquared:.3f}\n")
    except Exception as e:  # statsmodels 未装时退化为手动 OLS
        A = np.column_stack([np.ones(len(s)), s["case"].astype(float), s["age"], s["shannon"]])
        coef, *_ = np.linalg.lstsq(A, s["mbag"].to_numpy(), rcond=None)
        resid = s["mbag"].to_numpy() - A @ coef
        dof = len(s) - A.shape[1]
        s2 = resid @ resid / dof
        se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
        t = coef[1] / se[1]
        p = 2 * stats.t.sf(abs(t), dof)
        say(f"- 病例系数 = **{coef[1]:+.2f} 岁** (SE {se[1]:.2f}, t={t:.2f}, p={p:.2g})")
        say(f"- 年龄系数 = {coef[2]:+.2f}，多样性系数 = {coef[3]:+.2f}\n")

    (out / f"robustness_{args.study}.md").write_text("\n".join(lines), encoding="utf-8")
    s.to_csv(out / f"robustness_{args.study}_samples.csv")
    print(f"→ {out/f'robustness_{args.study}.md'}")


if __name__ == "__main__":
    main()
