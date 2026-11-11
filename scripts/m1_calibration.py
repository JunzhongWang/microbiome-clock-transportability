#!/usr/bin/env python3
"""
m1_calibration.py — 时钟的跨队列可迁移性 + 年龄匹配的病例对照比较

为什么必须做这两件事：
  第一版 M1 结果给出 QinN_2014 肝硬化 mBAG = -11.46 岁（p=3.9e-10），
  但稳健性检查发现该队列内 pred_age 与真实年龄几乎不相关（r=0.004）。
  这意味着 mBAG 在该队列里退化成「-年龄 + 常数」，组间差主要反映
  「病例组年龄更大」，而不是衰老加速/减速。必须用两件事纠正：

  1) 可迁移性谱：对每个有年龄的研究单独算 pred_age~age 的相关系数与 MAE，
     画出分布 —— 回答「这个钟到底在哪些队列里有效」；
  2) 年龄匹配：在 QinN_2014 内按年龄 1:1 最近邻匹配病例与对照，
     再比较 pred_age —— 这才是「同等年龄下，肝硬化患者微生物组
     是否显得更老/更年轻」的正确问法。

用法：
  python3 scripts/m1_calibration.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("data/metaphlan")
OUT = Path("results/m1_metaphlan")


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    return np.log10(X.clip(lower=0) + 1e-6)


def load_all():
    files = sorted((DATA / "per_study").glob("*.tsv.gz"))
    frames = []
    for f in files:
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
    return X.loc[common], meta.loc[common]


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def main():
    with open(OUT / "clock_model.pkl", "rb") as f:
        b = pickle.load(f)
    model, features = b["model"], b["features"]

    X, meta = load_all()
    for c in features:
        if c not in X.columns:
            X[c] = 0.0
    pred = pd.Series(model.predict(log_transform(X[features])), index=X.index)

    age = pd.to_numeric(meta.get("age"), errors="coerce")
    lines = []

    def say(s=""):
        print(s); lines.append(s)

    # ---- 1. 可迁移性谱 ----
    say("# 时钟跨队列可迁移性\n")
    rows = []
    for s, g in meta.groupby("study"):
        idx = g.index
        a = age.reindex(idx)
        p = pred.reindex(idx)
        ok = a.notna() & p.notna()
        if ok.sum() < 20:
            continue
        r, pv = stats.pearsonr(a[ok], p[ok])
        rows.append({
            "study": s, "n": int(ok.sum()),
            "r_age": round(float(r), 3), "p": pv,
            "MAE": round(float(np.mean(np.abs(p[ok] - a[ok]))), 2),
            "age_mean": round(float(a[ok].mean()), 1),
            "pred_mean": round(float(p[ok].mean()), 1),
            "bias": round(float(np.mean(p[ok] - a[ok])), 2),
        })
    cal = pd.DataFrame(rows).sort_values("r_age", ascending=False)
    cal.to_csv(OUT / "clock_transportability.csv", index=False)

    # 训练涉及的研究（用样本数近似判断：训练集里样本多的研究）
    say(f"在 {len(cal)} 个有足够年龄样本的研究中，pred_age 与真实年龄的相关性：\n")
    say(f"- 中位 r = **{cal['r_age'].median():+.3f}**（范围 {cal['r_age'].min():+.3f} ~ {cal['r_age'].max():+.3f}）")
    say(f"- r>0.4 的研究：{(cal['r_age'] > 0.4).sum()} / {len(cal)}")
    say(f"- r<0.1 的研究：{(cal['r_age'] < 0.1).sum()} / {len(cal)}")
    q = cal[cal["study"] == "QinN_2014"]
    if not q.empty:
        say(f"- **QinN_2014（唯一年龄+病例对照齐全的肝病队列）：r = {q.iloc[0]['r_age']:+.3f}，"
            f"MAE = {q.iloc[0]['MAE']:.1f} 岁，系统偏差 {q.iloc[0]['bias']:+.1f} 岁**")
    say("\n相关性最高的 5 个研究：")
    for _, r in cal.head(5).iterrows():
        say(f"  {r['study']:<24s} n={int(r['n']):4d}  r={r['r_age']:+.3f}  MAE={r['MAE']:.1f}")
    say("\n相关性最低的 5 个研究：")
    for _, r in cal.tail(5).iterrows():
        say(f"  {r['study']:<24s} n={int(r['n']):4d}  r={r['r_age']:+.3f}  MAE={r['MAE']:.1f}")

    # ---- 2. 年龄匹配的病例对照比较 ----
    say("\n# QinN_2014 年龄匹配分析\n")
    s = "QinN_2014"
    g = meta[meta["study"] == s].copy()
    g["pred_age"] = pred.reindex(g.index)
    g["age"] = age.reindex(g.index)
    lab = g["study_condition"].astype(str).str.lower()
    g["case"] = lab.str.contains("cirrhos|liver|hepat|nafld|nash")
    g = g.dropna(subset=["age", "pred_age"])
    case = g[g["case"]].copy()
    ctrl = g[~g["case"]].copy()
    say(f"匹配前：病例 {len(case)}（年龄 {case['age'].mean():.1f}±{case['age'].std():.1f}），"
        f"对照 {len(ctrl)}（年龄 {ctrl['age'].mean():.1f}±{ctrl['age'].std():.1f}）")
    t0 = stats.ttest_ind(case["age"], ctrl["age"], equal_var=False)
    say(f"年龄差异：**{case['age'].mean()-ctrl['age'].mean():+.1f} 岁**（p={t0.pvalue:.2g}）"
        f" ← 这就是 mBAG 差异的真实来源\n")

    # 1:1 最近邻匹配（不放回），按年龄差从小到大贪心
    pairs = []
    pool = list(ctrl.index)
    used = set()
    for i, row in case.sort_values("age").iterrows():
        best, bestd = None, np.inf
        for j in pool:
            if j in used:
                continue
            dd = abs(float(ctrl.loc[j, "age"]) - float(row["age"]))
            if dd < bestd:
                best, bestd = j, dd
        if best is not None and bestd <= 5.0:   # 允许 ±5 岁
            used.add(best)
            pairs.append((i, best, bestd))
    say(f"匹配成功 {len(pairs)} 对（卡尺 ±5 岁，1:1 不放回）")

    if pairs:
        mc = np.array([float(case.loc[a, "pred_age"]) for a, _, _ in pairs])
        mk = np.array([float(ctrl.loc[b_, "pred_age"]) for _, b_, _ in pairs])
        d = cohens_d(mc, mk)
        w = stats.wilcoxon(mc, mk)
        say(f"- 匹配后年龄：病例 {np.mean([case.loc[a,'age'] for a,_,_ in pairs]):.1f} "
            f"vs 对照 {np.mean([ctrl.loc[b_,'age'] for _,b_,_ in pairs]):.1f}")
        say(f"- 预测年龄：病例 {mc.mean():.2f} vs 对照 {mk.mean():.2f}，"
            f"差 **{mc.mean()-mk.mean():+.2f} 岁**，Cohen's d={d:+.2f}，Wilcoxon p={w.pvalue:.2g}")
        say(f"- 结论：同等年龄下，肝硬化患者微生物组"
            f"{'显著更老' if (w.pvalue < 0.05 and d > 0) else ('显著更年轻' if (w.pvalue < 0.05 and d < 0) else '与对照无显著差异')}"
            f"（效应量 |d|={abs(d):.2f}）\n")
        pd.DataFrame({"case_id": [a for a, _, _ in pairs],
                      "ctrl_id": [b_ for _, b_, _ in pairs],
                      "age_diff": [dd for _, _, dd in pairs],
                      "pred_case": mc, "pred_ctrl": mk}).to_csv(
            OUT / "matched_pairs_QinN_2014.csv", index=False)

    (OUT / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 3. 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        ax[0].hist(cal["r_age"], bins=20, color="#4a7ebb", edgecolor="white")
        ax[0].axvline(0, ls="--", c="grey", lw=1)
        ax[0].axvline(cal["r_age"].median(), ls="-", c="#c0392b", lw=1.5,
                      label=f"median r = {cal['r_age'].median():.2f}")
        ax[0].axvline(0.4, ls=":", c="grey", lw=1, label="r = 0.4")
        ax[0].set_xlabel("Pearson r (predicted vs chronological age), per study")
        ax[0].set_ylabel("Number of studies")
        ax[0].set_title("Clock transportability across cohorts")
        ax[0].legend(fontsize=8)
        if not q.empty:
            rx = cal.loc[cal["study"] == "QinN_2014", "r_age"].iloc[0]
            ax[0].axvline(rx, ls="-", c="#e67e22", lw=2, label="QinN_2014 (cirrhosis)")
            ax[0].legend(fontsize=8)
        if pairs:
            ax[1].scatter(mk, mc, s=22, alpha=0.7, color="#c0392b")
            lim = [min(mk.min(), mc.min()) - 2, max(mk.max(), mc.max()) + 2]
            ax[1].plot(lim, lim, ls="--", c="grey", lw=1)
            ax[1].set_xlabel("Predicted age — control (matched)")
            ax[1].set_ylabel("Predicted age — cirrhosis (matched)")
            ax[1].set_title(f"Age-matched pairs, QinN_2014 (n={len(pairs)})")
        fig.tight_layout()
        fig.savefig(OUT / "clock_calibration.png", dpi=200)
        print(f"  → {OUT/'clock_calibration.png'}")
    except Exception as e:
        print("[提示] 绘图跳过：", e)

    print(f"→ {OUT/'calibration_report.md'}")


if __name__ == "__main__":
    main()
