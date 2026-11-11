#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# m1_export_metaphlan.R — 把 curatedMetagenomicData 的 .rda 批量导出为
#                         下游 Python 可直接吃的宽表（行=样本，列=物种）
#
# 为什么不用 curatedMetagenomicData / SummarizedExperiment：
#   本机 R 为 x86_64（Rosetta），而系统编译工具链只有 arm64，
#   任何需源码编译的 Bioconductor 包都装不上。
#   实测发现 .rda 内实际存的是**普通数值矩阵**（taxa × samples，单位为百分数），
#   不是 S4 对象 —— 因此只用 base R 的 load() 即可，零依赖。
#
# 输出：
#   data/metaphlan/per_study/<study>.tsv.gz     行=样本，列=物种（相对丰度，列和=1）
#   data/metaphlan/sample_metadata.tsv.gz       列: sampleID/study/study_condition/age/...
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
ROOT     <- "."
RDA_DIR  <- file.path(ROOT, "data/metaphlan/rda")
OUT_DIR  <- file.path(ROOT, "data/metaphlan")
PER_DIR  <- file.path(OUT_DIR, "per_study")
META_RDA <- file.path(OUT_DIR, "sampleMetadata.rda")
dir.create(PER_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(META_RDA)) {
  stop("找不到 ", META_RDA, "，请先运行 scripts/fetch_curated_resources.py")
}

# ---- 1. 元数据（data.frame，base R 直接可读）-------------------------------
e <- new.env(); nm <- load(META_RDA, envir = e)
sm <- get(nm[1], envir = e)
# 真实列名为 sample_id（curatedMetagenomicData 3.14）
id_col <- if ("sample_id" %in% names(sm)) "sample_id" else "sampleID"
sm$sampleID <- as.character(sm[[id_col]])
cat(sprintf("sampleMetadata: %d 样本 × %d 列\n", nrow(sm), ncol(sm)))

# 研究名列：不同版本叫 study_name 或 study
study_col <- if ("study_name" %in% names(sm)) "study_name" else "study"
sm$study <- as.character(sm[[study_col]])

# ---- 2. 逐个 .rda 导出 ----------------------------------------------------
files <- list.files(RDA_DIR, pattern = "\\.rda$", full.names = TRUE)
if (length(files) == 0) stop("rda 目录为空，请先运行 scripts/fetch_curated_resources.py")

# 只要指定研究时用命令行参数过滤
if (length(args) > 0) {
  keep <- args
  files <- files[basename(files) %in% paste0(keep, ".rda")]
  cat("按参数只导出", length(files), "个研究\n")
}

export_one <- function(f) {
  study <- sub("\\.rda$", "", basename(f))
  out_f <- file.path(PER_DIR, paste0(study, ".tsv.gz"))
  if (file.exists(out_f) && file.size(out_f) > 1000) {
    return(list(study = study, status = "skip", n = NA_integer_, k = NA_integer_))
  }
  res <- tryCatch({
    ee <- new.env(); nn <- load(f, envir = ee)
    m <- get(nn[1], envir = ee)
    if (!is.matrix(m) || !is.numeric(m)) stop("不是数值矩阵: ", typeof(m))
    # 只保留物种层（MetaPhlAn 的 s__ 行）；若该研究无物种层则保留全部层级
    idx <- grepl("\\|s__|^s__", rownames(m))
    if (sum(idx) >= 10) m <- m[idx, , drop = FALSE]
    # 每列归一化为相对丰度（原始为百分数，且含多层级行）
    cs <- colSums(m, na.rm = TRUE)
    cs[cs <= 0 | is.na(cs)] <- 1
    m <- sweep(m, 2, cs, "/")
    m[!is.finite(m)] <- 0
    # 转置为 样本 × 物种，并清理物种名（去掉前缀层级，统一为 s__X）
    t_m <- t(m)
    colnames(t_m) <- sub("^.*\\|", "", colnames(t_m))
    df <- data.frame(sampleID = rownames(t_m), t_m,
                     check.names = FALSE, stringsAsFactors = FALSE)
    con <- gzfile(out_f, "w")
    write.table(df, con, sep = "\t", quote = FALSE, row.names = FALSE)
    close(con)
    list(study = study, status = "ok", n = nrow(df), k = ncol(df) - 1L)
  }, error = function(err) {
    list(study = study, status = paste("FAIL:", conditionMessage(err)), n = NA_integer_, k = NA_integer_)
  })
  res
}

cat("\n开始导出", length(files), "个研究...\n")
res <- lapply(files, export_one)
for (r in res) {
  if (r$status == "ok") {
    cat(sprintf("  [完成] %-28s %5d 样本 × %5d 物种\n", r$study, r$n, r$k))
  } else if (r$status == "skip") {
    cat(sprintf("  [跳过] %-28s 已存在\n", r$study))
  } else {
    cat(sprintf("  [%s] %s\n", r$status, r$study))
  }
}

# ---- 3. 元数据筛选到已导出的样本 ------------------------------------------
exported <- sub("\\.tsv\\.gz$", "", list.files(PER_DIR, pattern = "\\.tsv\\.gz$"))
if (length(exported) == 0) stop("没有成功导出任何研究")
sm2 <- sm[sm$study %in% exported, , drop = FALSE]

keep_cols <- intersect(c("sampleID", "study", "study_condition", "disease", "age",
                         "gender", "sex", "country", "BMI", "sequencing_platform",
                         "number_reads", "days_from_first_collection"),
                       names(sm2))
sm2 <- sm2[, keep_cols, drop = FALSE]
if ("age" %in% names(sm2)) sm2$age <- suppressWarnings(as.numeric(as.character(sm2$age)))
sm2 <- sm2[!is.na(sm2$sampleID) & sm2$sampleID != "", , drop = FALSE]
sm2 <- sm2[!duplicated(sm2$sampleID), , drop = FALSE]

con <- gzfile(file.path(OUT_DIR, "sample_metadata.tsv.gz"), "w")
write.table(sm2, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)

cat(sprintf("\n元数据已写出: %d 样本 × %d 列 → data/metaphlan/sample_metadata.tsv.gz\n",
            nrow(sm2), ncol(sm2)))
if ("age" %in% names(sm2)) {
  cat(sprintf("  有精确年龄: %d\n", sum(!is.na(sm2$age))))
}
cat("  研究数:", length(unique(sm2$study)), "\n")
cat("  study_condition 分布:\n")
print(head(sort(table(sm2$study_condition), decreasing = TRUE), 15))
