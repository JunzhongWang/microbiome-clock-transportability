#!/usr/bin/env bash
# 肝病公共 shotgun 队列批量下载 (ENA 直连, 无需 VPN)
#
# 用法:
#   bash scripts/fetch_liver_cohorts.sh                # 下载全部默认队列
#   bash scripts/fetch_liver_cohorts.sh --list         # 只生成 run 清单, 不下载
#   bash scripts/fetch_liver_cohorts.sh PRJEB6337      # 只处理指定队列
#   JOBS=8 bash scripts/fetch_liver_cohorts.sh         # 并发 8
#
# 输出结构:
#   data/liver/<cohort>/manifest.tsv       run 级清单(含大小/策略/平台)
#   data/liver/<cohort>/fastq/<run>_1.fastq.gz, _2.fastq.gz
#   data/liver/<cohort>/samplesheet.tsv    metaMap 输入(两列: fwd, rev)
set -uo pipefail
cd "$(dirname "$0")/.."

# 默认队列: 项目编号:短名
DEFAULT="PRJEB6337:qin2014_cirrhosis PRJNA373901:loomba2017_nafld PRJEB55534:epos_nafld"
JOBS=${JOBS:-4}
LIST_ONLY=0
PROJECTS=""

for a in "$@"; do
  case "$a" in
    --list) LIST_ONLY=1 ;;
    PRJ*) PROJECTS="$PROJECTS $a" ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

if [ -z "$PROJECTS" ]; then
  COHORTS="$DEFAULT"
else
  COHORTS=""
  for p in $PROJECTS; do
    # 在默认表里找短名, 找不到就用项目号当短名
    hit=$(echo "$DEFAULT" | tr ' ' '\n' | grep "^${p}:" || true)
    COHORTS="$COHORTS ${hit:-${p}:${p}}"
  done
fi

FIELDS="run_accession,sample_accession,fastq_ftp,fastq_bytes,fastq_md5,library_strategy,instrument_model,read_count,base_count"

for pair in $COHORTS; do
  ACC=${pair%%:*}; NAME=${pair##*:}
  OUT="data/liver/$NAME"
  mkdir -p "$OUT/fastq"

  echo "=============================================="
  echo "[$NAME] $ACC"
  echo "=============================================="

  # 1) 取 run 级清单 (ENA portal API)
  RAW="$OUT/_ena_raw.tsv"
  curl -s -m 120 "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${ACC}&result=read_run&fields=${FIELDS}&format=tsv" -o "$RAW"
  if [ ! -s "$RAW" ] || ! head -1 "$RAW" | grep -q run_accession; then
    echo "  ✗ 无法获取清单(项目号错误或 ENA 接口异常): $RAW"; continue
  fi

  # 2) 只保留 WGS(排除 16S/AMPLICON) 且双端可下载的 run
  {
    echo -e "run\tsample\tfwd\trev\tbytes\tmd5\tstrategy\tinstrument"
    tail -n +2 "$RAW" | awk -F'\t' 'BEGIN{OFS="\t"}
      $3!="" && $3 ~ /;/ && $6=="WGS" {
        split($3,f,";"); split($4,b,";"); split($5,m,";")
        print $1,$2,f[1],f[2],b[1]+b[2],m[1],$6,$7
      }'
  } > "$OUT/manifest.tsv"

  N=$(($(wc -l < "$OUT/manifest.tsv") - 1))
  GB=$(awk -F'\t' 'NR>1{s+=$5} END{printf "%.1f", s/1e9}' "$OUT/manifest.tsv")
  TOTAL=$(($(wc -l < "$RAW") - 1))
  echo "  run 总数 $TOTAL | 纳入 WGS 双端 $N 个 | 合计 ${GB} GB"

  if [ "$N" -eq 0 ]; then echo "  ✗ 无可用 run"; continue; fi
  if [ "$LIST_ONLY" -eq 1 ]; then
    echo "  清单已写入 $OUT/manifest.tsv (未下载)"
    continue
  fi

  # 3) 生成 metaMap 输入 samplesheet
  awk -F'\t' 'NR>1{print $1"\t"ENVIRON["PWD"]"/"ENVIRON["OUT"]"/fastq/"$1"_1.fastq.gz\t"ENVIRON["PWD"]"/"ENVIRON["OUT"]"/fastq/"$1"_2.fastq.gz"}' \
      OUT="$OUT" "$OUT/manifest.tsv" > "$OUT/samplesheet.tsv"

  # 4) 并发下载 (断点续传)
  echo "  开始下载 (并发 $JOBS) ..."
  tail -n +2 "$OUT/manifest.tsv" | awk -F'\t' -v out="$OUT/fastq" '{
      print $1"\thttps://"$3"\t"out"/"$1"_1.fastq.gz"
    }' | while IFS=$'\t' read -r run url dest; do
      [ -s "$dest" ] || echo -e "$url\t$dest"
    done > "$OUT/_dl1.txt"

  tail -n +2 "$OUT/manifest.tsv" | awk -F'\t' -v out="$OUT/fastq" '{
      print $1"\thttps://"$4"\t"out"/"$1"_2.fastq.gz"
    }' | while IFS=$'\t' read -r run url dest; do
      [ -s "$dest" ] || echo -e "$url\t$dest"
    done > "$OUT/_dl2.txt"

  cat "$OUT/_dl1.txt" "$OUT/_dl2.txt" | \
    xargs -P "$JOBS" -n 2 sh -c 'wget -c -q --tries=3 --timeout=60 -O "$1.part" "$0" && mv "$1.part" "$1"' 2>/dev/null || true

  DONE=$(ls "$OUT/fastq"/*_1.fastq.gz 2>/dev/null | wc -l | tr -d ' ')
  echo "  ✓ 完成: $DONE/$N 个样本(左端文件)"
  rm -f "$OUT/_dl1.txt" "$OUT/_dl2.txt" "$RAW"
done

echo
echo "全部完成。查看: ls data/liver/*/manifest.tsv"
