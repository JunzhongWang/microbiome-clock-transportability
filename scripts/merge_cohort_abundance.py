#!/usr/bin/env python3
"""把 metaMap 的基因组级 read counts 合并成"样本 x 物种"矩阵, 并与训练特征对齐。

输入:
  data/liver/<cohort>/results/summary/bwa_counts-total.csv   (genome x sample, metaMap 输出)
  data/uhgg_abundance/uhgg_abundance_filtered.csv            (提供 Genome -> Lineage 物种映射)
  results/m1/train_features_p10.csv                          (提供训练特征列顺序)

输出:
  results/liver/<cohort>_species_counts.csv   样本 x 训练特征物种(缺失列补 0)
  results/liver/<cohort>_qc.csv               每样本 QC(总 reads / 检出物种数)

用法: python3 scripts/merge_cohort_abundance.py [cohort ...]
"""
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVER = ROOT / "data/liver"
OUT = ROOT / "results/liver"
OUT.mkdir(parents=True, exist_ok=True)

# Genome -> 物种 (与训练矩阵同一套 Lineage 解析规则)
print("loading genome->species map ...")
ann = pd.read_csv(ROOT / "data/uhgg_abundance/uhgg_abundance_filtered.csv",
                  usecols=["Genome", "Lineage"], index_col=0)
sp_map = ann["Lineage"].str.extract(r"s__([^;]+)")[0].fillna("unassigned")
sp_map = sp_map[~sp_map.index.duplicated()]

# 训练特征列(对齐用)
train_cols = pd.read_csv(ROOT / "results/m1/train_features_p10.csv", nrows=0).columns[1:].tolist()
print(f"training feature space: {len(train_cols)} species")

cohorts = sys.argv[1:] or [d.name for d in LIVER.iterdir() if d.is_dir()]
for name in cohorts:
    f = LIVER / name / "results/summary/bwa_counts-total.csv"
    if not f.exists():
        print(f"跳过 {name}: 缺少 {f.relative_to(ROOT)}")
        continue

    cnt = pd.read_csv(f, index_col=0)
    # metaMap 输出的行名可能是 genome_1 形式, 取第一个下划线前的 genome id
    cnt.index = cnt.index.astype(str).str.replace(r"_\d+$", "", regex=True)
    cnt = cnt.groupby(level=0).sum()

    sp = cnt.groupby(sp_map.reindex(cnt.index)).sum()   # species x sample
    mat = sp.T                                          # sample x species

    # 对齐训练特征空间: 缺失物种补 0, 多余物种保留在 *_extra 列
    extra = mat.columns.difference(train_cols)
    aligned = mat.reindex(columns=train_cols, fill_value=0)
    if len(extra):
        print(f"  {name}: {len(extra)} 个物种不在训练特征内(已保留到 _extra 文件)")
        mat[extra].to_csv(OUT / f"{name}_species_counts_extra.csv")

    aligned.index.name = "Sample"
    aligned.to_csv(OUT / f"{name}_species_counts.csv")

    qc = pd.DataFrame({
        "total_counts": aligned.sum(axis=1),
        "species_detected": (aligned > 0).sum(axis=1),
        "species_in_training_space": (aligned[train_cols] > 0).sum(axis=1),
    })
    qc.to_csv(OUT / f"{name}_qc.csv")

    print(f"✓ {name}: {aligned.shape[0]} 样本 x {aligned.shape[1]} 物种 | "
          f"总 reads 中位 {qc.total_counts.median()/1e6:.1f}M | "
          f"检出物种中位 {qc.species_in_training_space.median():.0f}")
    low = qc[qc.total_counts < 1e7]
    if len(low):
        print(f"  ⚠ {len(low)} 个样本总 reads <10M (建议排除或标注): {', '.join(low.index[:5])}")

print("\n下一步: ComBat 批次校正 -> 投影基线时钟 -> 计算 mBAG")
