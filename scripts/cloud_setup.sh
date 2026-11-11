#!/usr/bin/env bash
# =============================================================================
# cloud_setup.sh — 在全新的云服务器（Ubuntu 22.04/24.04）上装齐 M1/M3 所需环境
#
# 用法:
#   bash scripts/cloud_setup.sh            # 首次安装
#   bash scripts/cloud_setup.sh --check    # 只检查不安装
#
# 装的东西: 系统依赖 / Miniconda / bwa / samtools / snakemake / aria2c / python 包
# 幂等: 重复执行不会破坏已有安装
# =============================================================================
set -euo pipefail

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="liver-mb"
LOG_PREFIX="[cloud_setup]"

say() { printf "%s %s\n" "$LOG_PREFIX" "$*"; }
die() { printf "%s ERROR: %s\n" "$LOG_PREFIX" "$*" >&2; exit 1; }

# ---------- 0. 基本环境体检 ----------
say "系统信息: $(uname -srm)"
[ "$(uname -s)" = "Linux" ] || die "本脚本面向 Linux 云服务器；macOS 请直接用 conda 装 bwa/samtools"

DISK_AVAIL_GB=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
CPU_N=$(nproc)
say "可用磁盘: ${DISK_AVAIL_GB} GB | 内存: ${MEM_GB} GB | CPU: ${CPU_N} 核"

if [ "$DISK_AVAIL_GB" -lt 300 ]; then
  say "警告: 可用磁盘不足 300 GB。全量 2.2 TB 建议挂独立数据盘，或改用 S3/OSS 中转 + 分批处理。"
fi
if [ "$MEM_GB" -lt 14 ]; then
  say "警告: 内存低于 14 GB，BWA 比对 UHGG 参考（约 14 Gbp）可能 OOM，建议 ≥16 GB。"
fi

# ---------- 1. 系统依赖 ----------
if [ "$CHECK_ONLY" -eq 0 ]; then
  say "安装系统依赖 (apt)..."
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq \
    build-essential wget curl git aria2 pigz unzip ca-certificates \
    libbz2-dev liblzma-dev zlib1g-dev libcurl4-openssl-dev time >/dev/null
  say "系统依赖完成"
else
  for c in wget curl git aria2c pigz; do
    command -v "$c" >/dev/null 2>&1 && say "OK  $c" || say "缺失 $c"
  done
fi

# ---------- 2. Miniconda ----------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  [ "$CHECK_ONLY" -eq 1 ] && { say "缺失 conda ($CONDA_DIR)"; }
  if [ "$CHECK_ONLY" -eq 0 ]; then
    say "安装 Miniconda..."
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O /tmp/miniconda.sh \
      || die "Miniconda 下载失败，检查网络"
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" >/dev/null || die "Miniconda 安装失败"
    rm -f /tmp/miniconda.sh
    say "Miniconda 安装完成: $CONDA_DIR"
  fi
else
  say "已存在 conda: $("$CONDA_DIR/bin/conda" --version)"
fi

# ---------- 3. 生信环境 ----------
if [ -x "$CONDA_DIR/bin/conda" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    [ "$CHECK_ONLY" -eq 1 ] && say "缺失 conda 环境: $ENV_NAME"
    if [ "$CHECK_ONLY" -eq 0 ]; then
      say "创建 conda 环境 $ENV_NAME（bwa/samtools/snakemake）..."
      conda create -y -q -n "$ENV_NAME" -c conda-forge -c bioconda \
        python=3.11 bwa=0.7.17 samtools=1.19 snakemake-minimal=7.32 \
        pigz aria2 seqtk >/dev/null || die "conda 环境创建失败"
      say "conda 环境创建完成"
    fi
  else
    say "已存在 conda 环境: $ENV_NAME"
    if [ "$CHECK_ONLY" -eq 0 ]; then
      conda activate "$ENV_NAME"
      for t in bwa samtools snakemake; do
        command -v "$t" >/dev/null 2>&1 || die "环境内缺少 $t，请重建环境"
      done
    fi
  fi
fi

# ---------- 4. Python 分析包 ----------
if [ "$CHECK_ONLY" -eq 0 ] && [ -x "$CONDA_DIR/bin/conda" ]; then
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  say "安装 python 分析包 (pandas/numpy/scikit-learn/statsmodels)..."
  pip install -q --no-input \
    pandas numpy scipy scikit-learn statsmodels openpyxl matplotlib \
    || die "pip 安装失败"
  python - <<'PY'
import pandas, numpy, sklearn, statsmodels
print(f"[cloud_setup] python 包就绪: pandas {pandas.__version__}, sklearn {sklearn.__version__}")
PY
fi

# ---------- 5. 汇总 ----------
say "--------------- 环境摘要 ---------------"
[ -x "$CONDA_DIR/bin/conda" ] && source "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate "$ENV_NAME" 2>/dev/null || true
for t in bwa samtools snakemake aria2c pigz; do
  if command -v "$t" >/dev/null 2>&1; then
    say "OK   $t"
  else
    say "缺失 $t"
  fi
done
say "----------------------------------------"
if [ "$CHECK_ONLY" -eq 0 ]; then
  cat <<'TIP'

下一步（在项目根目录执行）：
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate liver-mb

  # 1) 先看清单，确认样本量与体积
  bash scripts/fetch_liver_cohorts.sh --list

  # 2) 只跑 Loomba（292 GB）验证全流程
  bash scripts/fetch_liver_cohorts.sh loomba2017

  # 3) 建参考库（15 GB，一次性）
  bash scripts/build_uhgg_ref.sh 8

  # 4) 定量
  JOBS=12 bash scripts/quantify_cohorts.sh

  # 5) 合并
  python3 scripts/merge_cohort_abundance.py

⚠️  收工前务必：删原始 fastq、终止所有竞价实例、释放数据盘 EBS/云盘
TIP
fi
