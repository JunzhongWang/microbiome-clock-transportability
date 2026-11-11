#!/usr/bin/env python3
"""M1 第一步: 构建时钟训练特征矩阵。

真实数据格式(2026-11-09 核实):
  uhgg_abundance_filtered.csv = 基因组 x 样本 的转置矩阵
    行: GUT_GENOME000001 ... (4,612 个 UHGG 基因组)
    列: Genome | Lineage | Status | 11,115 个样本(ERS/SRS/DRS 混合, 与 Table S1 Sample 一一对应)
    值: read counts

处理流程(对齐 Metabolism 2027 2.2 节):
  1. 按 Lineage 的 s__ 字段把基因组聚合成物种
  2. 转置为 样本 x 物种
  3. 过滤: 流行率 >= 20%, 方差 >= 第 10 百分位
  4. 与 Table S1 的健康 + 有精确年龄样本(4,551)取交集

输出:
  results/m1/train_features.csv            4,551 x 物种 的 read count 矩阵
  results/m1/train_meta.csv                对应 metadata
  results/m1/train_meta_agegroup_only.csv  仅有年龄分组的健康样本(敏感性分析)
  results/m1/species_prevalence.csv        物种流行率(供后续特征筛选参考)
"""
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABUND = ROOT / "data/uhgg_abundance/uhgg_abundance_filtered.csv"
META = ROOT / "data/metadata/table_s1.csv"
OUT = ROOT / "results/m1"
OUT.mkdir(parents=True, exist_ok=True)

# 流行率阈值可调: python3 scripts/m1_prepare_features.py 0.10
PREV_THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
SUFFIX = f"p{int(PREV_THRESHOLD*100):02d}"

# ---- 0. 前置检查 ----
if not ABUND.exists() or ABUND.stat().st_size < 1_000_000:
    raise SystemExit(
        "丰度矩阵缺失或损坏（应为 113 MB）。下载: https://figshare.com/ndownloader/files/57348290\n"
        f"放置到: {ABUND}"
    )
if not META.exists():
    raise SystemExit(f"缺少 metadata: {META}")
print(f"abundance: {ABUND.stat().st_size/1e6:.0f} MB | metadata: {META.stat().st_size/1e3:.0f} KB")

# ---- 1. 读转置矩阵, 基因组 -> 物种 ----
print("loading abundance matrix (genomes x samples) ...")
g = pd.read_csv(ABUND, index_col=0)
print(f"  genomes: {g.shape[0]}, samples: {g.shape[1] - 2}")

species = g["Lineage"].str.extract(r"s__([^;]+)")[0].fillna("unassigned")
counts = g.drop(columns=["Lineage", "Status"])
counts["species"] = species.values

sp = counts.groupby("species").sum()          # species x samples
mat = sp.T.astype("float64")                  # samples x species
mat.index.name = "Sample"
print(f"  species: {mat.shape[1]}, samples: {mat.shape[0]}")

# ---- 2. 过滤(流行率/方差) ----
prev = (mat > 0).mean(axis=0)
mat = mat.loc[:, prev >= PREV_THRESHOLD]
var_thr = mat.var(axis=0).quantile(0.10)
mat = mat.loc[:, mat.var(axis=0) >= var_thr]
print(f"  after prevalence>={PREV_THRESHOLD:.0%} & variance>=P10: {mat.shape}")

pd.DataFrame({"prevalence": prev}).to_csv(OUT / "species_prevalence.csv")

# ---- 3. 与 Table S1 连接, 筛健康 + 有精确年龄 ----
meta = pd.read_csv(META, encoding="utf-8-sig")   # WPS/Excel 导出带 BOM
ID_COL, HEALTH_COL, AGE_COL, AGEGRP_COL = "Sample", "Health state", "Age", "Age group"
meta = meta.dropna(subset=[ID_COL]).drop_duplicates(subset=[ID_COL])

healthy = meta[meta[HEALTH_COL].astype(str).str.strip().str.lower() == "healthy"]
with_age = healthy[healthy[AGE_COL].notna()]
agegrp_only = healthy[healthy[AGE_COL].isna() & healthy[AGEGRP_COL].notna()]
print(f"healthy: {len(healthy)} | with exact age: {len(with_age)} | age-group only: {len(agegrp_only)}")

common = mat.index.intersection(with_age[ID_COL].astype(str))
sub = mat.loc[common]
sub_meta = with_age.set_index(ID_COL).loc[common].reset_index()
print(f"  matched training samples: {len(sub)}  (预期 4551)")

# ---- 4. 落盘 ----
sub.to_csv(OUT / f"train_features_{SUFFIX}.csv")
sub_meta.to_csv(OUT / f"train_meta_{SUFFIX}.csv", index=False)
agegrp_only.to_csv(OUT / "train_meta_agegroup_only.csv", index=False)

print(f"\nsaved: train_features_{SUFFIX}.csv {sub.shape[0]} samples x {sub.shape[1]} species")
print(f"saved: train_meta_{SUFFIX}.csv {len(sub_meta)} rows | train_meta_agegroup_only.csv {len(agegrp_only)} rows")
print(f"age range in training set: {sub_meta[AGE_COL].min():.0f}-{sub_meta[AGE_COL].max():.0f}")
print("下一步: log1p + StandardScaler -> 时钟训练(按 Study 分层交叉验证, 防同研究泄漏)")
