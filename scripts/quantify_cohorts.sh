#!/usr/bin/env bash
# 肝病队列丰度定量 (metaMap / Snakemake)
#
# 前置:
#   1. bash scripts/fetch_liver_cohorts.sh          # 已下载 fastq + samplesheet
#   2. bash scripts/build_uhgg_ref.sh               # 已建 UHGG 参考库与 bwa index
#   3. conda install -c bioconda -c conda-forge snakemake bwa samtools
#
# 用法:
#   bash scripts/quantify_cohorts.sh                 # 处理 data/liver 下所有队列
#   JOBS=16 bash scripts/quantify_cohorts.sh epos_nafld
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-8}
REF=$(pwd)/data/ref/uhgg/uhgg_reps.fa
MM=$(pwd)/reference/metamap

[ -f "$REF.bwt" ] || { echo "✗ 缺少 bwa 索引, 先跑 scripts/build_uhgg_ref.sh"; exit 1; }
command -v snakemake >/dev/null || { echo "✗ 缺少 snakemake"; exit 1; }

TARGETS=${*:-$(ls data/liver 2>/dev/null)}

for NAME in $TARGETS; do
  DIR=$(pwd)/data/liver/$NAME
  # 每次定量前重建 samplesheet(处理下载中断/多 run 样本)
  bash scripts/make_samplesheet.sh "$NAME" >/dev/null 2>&1 || true
  SS="$DIR/samplesheet.tsv"
  [ -s "$SS" ] || { echo "跳过 $NAME (无 samplesheet: 请先下载 fastq)"; continue; }
  N=$(wc -l < "$SS" | tr -d ' ')
  echo "=============================================="
  echo "[$NAME] $N 个样本"
  echo "=============================================="

  # 写临时 config (不动 metaMap 原仓库里的文件)
  CFG="$DIR/metamap.config.yml"
  cat > "$CFG" <<YAML
input:
    "$SS"
output:
    "$DIR/results"
database:
    bwa_db: "$REF"
    type: "contigs"
YAML

  # 在 metaMap 目录下运行, 结果写入 cohort 目录
  ( cd "$MM" && snakemake --configfile "$CFG" --use-conda -k -j "$JOBS" \
      --rerun-incomplete --latency-wait 60 2>&1 | tail -20 )

  echo "  ✓ 汇总文件: $DIR/results/summary/"
  ls -1 "$DIR/results/summary/" 2>/dev/null | sed 's/^/    /' || echo "    (未生成, 检查上方日志)"
done

echo
echo "下一步: python3 scripts/merge_cohort_abundance.py"
