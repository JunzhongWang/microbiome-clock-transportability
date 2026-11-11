#!/usr/bin/env bash
# 从 manifest 生成 metaMap 输入 samplesheet, 并处理同一 sample 跨多个 run 的情况
# (多 run 样本先物理拼接, 避免把同一样本当独立样本)
#
# 用法: bash scripts/make_samplesheet.sh [cohort ...]
set -euo pipefail
cd "$(dirname "$0")/.."

COHORTS=${*:-$(ls data/liver 2>/dev/null)}
for NAME in $COHORTS; do
  DIR=$(pwd)/data/liver/$NAME
  MF="$DIR/manifest.tsv"
  [ -s "$MF" ] || { echo "跳过 $NAME (无 manifest)"; continue; }

  mkdir -p "$DIR/fastq" "$DIR/merged"
  # sample -> run 列表
  tail -n +2 "$MF" | cut -f1,2 | awk -F'\t' '{a[$2]=a[$2]" "$1} END{for(s in a) print s"\t"substr(a[s],2)}' \
    | sort > "$DIR/_sample_runs.tsv"

  : > "$DIR/samplesheet.tsv"
  multi=0
  while IFS=$'\t' read -r s runs; do
    set -- $runs
    if [ $# -eq 1 ]; then
      f1="$DIR/fastq/$1_1.fastq.gz"; f2="$DIR/fastq/$1_2.fastq.gz"
      if [ -s "$f1" ] && [ -s "$f2" ]; then
        printf "%s\t%s\t%s\n" "$s" "$f1" "$f2" >> "$DIR/samplesheet.tsv"
      fi
    else
      multi=$((multi+1))
      f1="$DIR/merged/${s}_1.fastq.gz"; f2="$DIR/merged/${s}_2.fastq.gz"
      if [ ! -s "$f1" ] || [ ! -s "$f2" ]; then
        ok=1; for r in $runs; do [ -s "$DIR/fastq/${r}_1.fastq.gz" ] || ok=0; done
        if [ $ok -eq 1 ]; then
          : > "$f1"; : > "$f2"
          for r in $runs; do
            cat "$DIR/fastq/${r}_1.fastq.gz" >> "$f1"
            cat "$DIR/fastq/${r}_2.fastq.gz" >> "$f2"
          done
        fi
      fi
      [ -s "$f1" ] && printf "%s\t%s\t%s\n" "$s" "$f1" "$f2" >> "$DIR/samplesheet.tsv"
    fi
  done < "$DIR/_sample_runs.tsv"

  printf "%-22s samplesheet: %s 行 | 多 run 样本: %s 个\n" "$NAME" "$(wc -l < "$DIR/samplesheet.tsv" | tr -d ' ')" "$multi"
done
