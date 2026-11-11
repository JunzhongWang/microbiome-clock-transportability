#!/usr/bin/env bash
# 构建 metaMap 用的 UHGG v1.0 参考库 (仅 4,612 个 species representative genomes,
# 与训练矩阵的行完全一致, 保证定量结果可直接与训练特征对齐)
#
# 用法: bash scripts/build_uhgg_ref.sh [并发数, 默认 6]
# 产物: data/ref/uhgg/uhgg_reps.fa (+ bwa index), 约 15 GB 磁盘
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${1:-6}
REF=data/ref/uhgg
FNA=$REF/fna
mkdir -p "$FNA"
BASE="https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v1.0"

# ---- 1. 生成下载清单 (genome id -> MGYG id -> URL) ----
if [ ! -s "$REF/urls.tsv" ]; then
  echo "[1/4] 生成 URL 清单 ..."
  curl -s -m 300 "$BASE/genomes-all_metadata.tsv" -o "$REF/genomes-all_metadata.tsv"
  python3 - <<'PY'
import csv, pathlib
root = pathlib.Path("data/ref/uhgg")
# 训练矩阵里的基因组列表(第一列)
with open("data/uhgg_abundance/uhgg_abundance_filtered.csv") as f:
    want = {line.split(",", 1)[0] for i, line in enumerate(f) if i > 0}
base = "https://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v1.0"
rows, n = [], 0
with open(root / "genomes-all_metadata.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        g = r["Genome"]
        if g in want:
            mgyg = r["MGnify_accession"]           # MGYG-HGUT-00001
            grp = mgyg[:13]                         # MGYG-HGUT-000
            url = f"{base}/uhgg_catalogue/{grp}/{mgyg}/genome/{mgyg}.fna"
            rows.append((g, mgyg, url)); n += 1
with open(root / "urls.tsv", "w") as o:
    for g, m, u in rows:
        o.write(f"{g}\t{m}\t{u}\n")
print(f"  {n} 个基因组待下载 (矩阵共 {len(want)} 行)")
PY
fi

# ---- 2. 并行下载 (断点续传) ----
echo "[2/4] 下载基因组 FASTA (并发 $JOBS) ..."
cut -f1,3 "$REF/urls.tsv" | while IFS=$'\t' read -r g url; do
  [ -s "$FNA/$g.fna" ] || echo -e "$url\t$FNA/$g.fna"
done | xargs -P "$JOBS" -n 2 sh -c 'wget -c -q --tries=3 --timeout=60 -O "$1.part" "$0" && mv "$1.part" "$1"' || true
echo "  已下载: $(ls "$FNA"/*.fna 2>/dev/null | wc -l | tr -d ' ') 个"

# ---- 3. 合并为单一 FASTA, 序列头改为 genome_1, genome_2 ... (metaMap contigs 模式要求) ----
echo "[3/4] 合并 FASTA 并重命名序列头 ..."
: > "$REF/uhgg_reps.fa"
while IFS=$'\t' read -r g _ _; do
  f="$FNA/$g.fna"
  [ -s "$f" ] || continue
  awk -v g="$g" '/^>/{n++; print ">"g"_"n; next} {print}' "$f" >> "$REF/uhgg_reps.fa"
done < "$REF/urls.tsv"

# ---- 4. BWA 建索引 ----
echo "[4/4] BWA 建索引 (耗时 1-3 小时, 内存约 30 GB) ..."
if [ ! -f "$REF/uhgg_reps.fa.bwt" ]; then
  command -v bwa >/dev/null || { echo "✗ 缺少 bwa, 请先 conda install -c bioconda bwa"; exit 1; }
  bwa index "$REF/uhgg_reps.fa"
fi

du -sh "$REF/uhgg_reps.fa"* | sed 's/^/  /'
echo "完成。参考库: $REF/uhgg_reps.fa (type=contigs)"
