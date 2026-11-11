#!/usr/bin/env bash
# M1 训练数据一键下载脚本
# 用法: bash scripts/download_training_data.sh
# 注意: 本机直连 figshare 可能被 Cloudflare 拦截(403)。
#       在单位服务器 / HPC 登录节点 / 挂代理环境运行即可。
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/uhgg_abundance/uhgg_abundance_filtered.csv
EXPECT_MD5=c3517e8a7fe3bd3d7112bcea845c8aff

echo "[1/3] 下载 UHGG 丰度矩阵 (113 MB, doi:10.6084/m9.figshare.29963597.v1)"
mkdir -p data/uhgg_abundance
if [ -f "$OUT" ]; then
  echo "  已存在: $OUT (跳过下载)"
else
  # 备选镜像依次尝试
  for URL in \
    "https://ndownloader.figshare.com/files/57348290" \
    "https://ndownloader.figshare.com/files/57348290?download=1"; do
    echo "  尝试: $URL"
    if curl -fL --retry 3 --connect-timeout 20 -A "Mozilla/5.0" -o "$OUT.tmp" "$URL"; then
      mv "$OUT.tmp" "$OUT"; break
    fi
  done
  [ -f "$OUT" ] || { echo "下载失败: 请挂 VPN / 换网络环境后重试,或改走 README 中的『备用路线B』"; exit 1; }
fi

echo "[2/3] MD5 校验 (期望 $EXPECT_MD5)"
ACTUAL=$(md5sum "$OUT" | awk '{print $1}') || ACTUAL=$(md5 -q "$OUT")
if [ "$ACTUAL" != "$EXPECT_MD5" ]; then
  echo "MD5 不匹配: $ACTUAL"; exit 1
fi
echo "  校验通过"

echo "[3/3] 完成。下一步: python3 scripts/m1_prepare_features.py"
