#!/usr/bin/env python3
"""导出投稿用 Additional files 1-7（机器可读 CSV + 1 个 PDF）

背景
----
稿件 Additional files 列表里写的文件此前只存在于 results/ 的中间表，
从未真正导出为投稿格式；本脚本把它们统一生成到 results/additional_files/。

口径约定（与正文一致）
--------------------
* 分析面板 = per_study 丰度表与 sampleMetadata 的复合键（study::sampleID）交集
  → 74 个研究、14,898 样本、2,022 物种（去重后口径）。
  注意：data/metaphlan/per_study 下有 80 个 tsv，其中 6 个研究
  （BedarfJR_2017、FrankelAE_2017、LeeKA_2022、MetaCardis_2020_a、
  PetersBA_2019、WindTT_2020）来自较新 release，样本 ID 不在本次使用的
  sampleMetadata 里，实际贡献 0 个样本，故不进入任何统计。
* 年龄/健康定义、病例计数一律在去重后面板上重算，保证列和可核对：
  n_samples_analysed 合计 14,898；n_with_exact_age 合计 10,578；
  loose healthy+age = 7,657；strict healthy+age = 7,534。

输出（results/additional_files/）
  Additional_file_1_cohort_survey.csv        74 研究普查 + PMID + accession
  Additional_file_2_band_performance.csv     6 实现 × 3 CV 方案 × 年龄带
  Additional_file_3_loso_projection.csv      264 次留一研究投影
  Additional_file_4_interclock_concordance.csv 15 对实现间一致性
  Additional_file_5_deviation_per_sample.csv 逐样本健康流形偏离度 D
  Additional_file_6_Figure_S1.pdf            Figure S1
  Additional_file_7_sample_accessions.csv    14,898 样本级 ENA/SRA accession
"""
import os
import pickle
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "additional_files")
os.makedirs(OUT, exist_ok=True)

CASE_LABELS = {"case", "cirrhosis", "nafld", "nash", "ibd", "cancer", "adenoma"}

BAND_MAP = {
    "全跨度 0-107": "All 0-107",
    "儿童 <18": "Children <18",
    "青年 18-40": "Young 18-40",
    "中年 40-70": "Middle-aged 40-70",
    "老年 60+": "Older 60+",
    "成人 18+": "Adults 18+",
    "非中年→40-70": "Non-middle-aged -> 40-70",
    "儿童→成人": "Children -> adults",
    "中年→非中年": "Middle-aged -> non-middle-aged",
}
FAMILY_MAP = {
    "线性（L1+L2）": "Linear (L1+L2)",
    "装袋树": "Bagged trees",
    "装袋树·属级": "Bagged trees (genus level)",
    "提升树": "Boosted trees",
    "神经网络": "Neural network",
    "核方法": "Kernel method",
}
MODE_MAP = {"A_loose": "loose", "B_strict": "strict"}


def _first_accession(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return ""
    return x.split(";")[0].strip()


# ------------------------------------------------------------------ AF1
def build_af1():
    with open(os.path.join(ROOT, "data/metaphlan/_panel_cache.pkl"), "rb") as fh:
        X, meta = pickle.load(fh)
    d = meta.copy()
    d["age_v"] = pd.to_numeric(d["age"], errors="coerce")
    cond = d["study_condition"].astype(str).str.lower()
    d["is_case"] = cond.isin(CASE_LABELS)

    rows = []
    for study, g in d.groupby("study", sort=True):
        av = g["age_v"].dropna()
        rows.append(dict(
            study=study,
            n_samples_analysed=len(g),
            n_with_exact_age=int(av.size),
            age_mean_y=round(float(av.mean()), 1) if av.size else np.nan,
            age_sd_y=round(float(av.std()), 1) if av.size > 1 else np.nan,
            age_min_y=float(av.min()) if av.size else np.nan,
            age_max_y=float(av.max()) if av.size else np.nan,
            n_case=int(g["is_case"].sum()),
            n_control=int((~g["is_case"]).sum()),
            n_species_detected=int((X.loc[g.index] > 0).any(axis=0).sum()),
        ))
    af1 = pd.DataFrame(rows)

    # 队列内自训练结果
    ic = pd.read_csv(os.path.join(RES, "m1_metaphlan/intracohort_clock.csv"))
    ic = ic.rename(columns={"n": "n_self_train", "r2": "self_train_cv_r2",
                            "mae": "self_train_cv_mae_y", "r": "self_train_cv_r"})
    af1 = af1.merge(
        ic[["study", "n_self_train", "self_train_cv_r2", "self_train_cv_mae_y",
            "self_train_cv_r"]], on="study", how="left")
    af1["self_trainable"] = np.where(af1["n_self_train"].notna(), "yes", "no")

    # PMID / accession
    acc = pd.read_csv(os.path.join(ROOT, "data/metaphlan/study_accessions.csv"))
    acc = acc.fillna({"PMID": "", "NCBI_accession": ""})
    acc["NCBI_accession"] = acc["NCBI_accession"].map(_first_accession)
    agg = acc.groupby("study_name").agg(
        pmid=("PMID", "first"),
        n_sequence_accessions=("NCBI_accession", lambda s: s[s != ""].nunique()),
        accession_example=("NCBI_accession", lambda s: next((v for v in s if v), "")),
    ).reset_index().rename(columns={"study_name": "study"})
    af1 = af1.merge(agg, on="study", how="left")
    af1["pmid"] = af1["pmid"].fillna("")
    af1["n_sequence_accessions"] = af1["n_sequence_accessions"].fillna(0).astype(int)
    af1["accession_example"] = af1["accession_example"].fillna("")

    af1 = af1[[
        "study", "pmid", "n_samples_analysed", "n_with_exact_age",
        "age_mean_y", "age_sd_y", "age_min_y", "age_max_y",
        "n_case", "n_control", "n_species_detected",
        "self_trainable", "n_self_train", "self_train_cv_r2",
        "self_train_cv_mae_y", "self_train_cv_r",
        "n_sequence_accessions", "accession_example",
    ]].sort_values("study")
    af1.to_csv(os.path.join(OUT, "Additional_file_1_cohort_survey.csv"), index=False)
    print(f"AF1: {len(af1)} studies | samples={af1.n_samples_analysed.sum()} "
          f"| with age={af1.n_with_exact_age.sum()} "
          f"| self-trainable={int((af1.self_trainable == 'yes').sum())}")
    return X, d


# ------------------------------------------------------------------ AF7
def build_af7(meta_rows, panel_index):
    sa = pd.read_csv(os.path.join(ROOT, "data/metaphlan/sample_accessions.csv"))
    sa = sa.fillna({"PMID": "", "NCBI_accession": ""})
    sa["NCBI_accession"] = sa["NCBI_accession"].map(_first_accession)
    sa["key"] = sa["study"].astype(str) + "::" + sa["sample_id"].astype(str)
    sa = sa.drop_duplicates("key").set_index("key")

    out = meta_rows.loc[panel_index, ["study"]].copy()
    out["sample_id"] = [k.split("::", 1)[1] for k in panel_index]
    out["pmid"] = sa.reindex(panel_index)["PMID"].to_numpy()
    out["NCBI_accession"] = sa.reindex(panel_index)["NCBI_accession"].to_numpy()
    out["body_site"] = "stool"
    out["NCBI_accession"] = out["NCBI_accession"].fillna("")
    # curatedMetagenomicData 本身对部分样本未提供 run accession（4,760/20,283），
    # 这些样本只能回溯到原始文献（pmid），必须显式标注，不能留空让审稿人猜。
    out["accession_available"] = np.where(out["NCBI_accession"] != "", "yes", "no")
    out = out.reset_index(drop=True)
    out.to_csv(os.path.join(OUT, "Additional_file_7_sample_accessions.csv"), index=False)
    miss = int((out["accession_available"] == "no").sum())
    print(f"AF7: {len(out)} samples | no accession in cMD={miss} "
          f"({miss / len(out) * 100:.1f}%)")


# ------------------------------------------------------------------ AF2-AF5
def build_af2():
    b = pd.read_csv(os.path.join(RES, "m1_multiclock/band_performance.csv"))
    b["mode"] = b["mode"].map(MODE_MAP).fillna(b["mode"])
    b["family"] = b["family"].map(FAMILY_MAP).fillna(b["family"])
    b["band"] = b["band"].map(BAND_MAP).fillna(b["band"])
    b["tag"] = b["tag"].map(BAND_MAP).fillna(b["tag"])
    b = b.rename(columns={
        "mode": "healthy_definition", "cv": "cv_scheme", "n": "n_samples",
        "r2_lo": "r2_ci_low", "r2_hi": "r2_ci_high", "mae": "mae_y",
        "r": "pearson_r", "p": "p_value", "resid_sd": "residual_sd_y",
        "pred_sd": "prediction_sd_y", "age_sd": "age_sd_y",
    })
    b = b[["healthy_definition", "impl", "family", "cv_scheme", "band",
           "n_samples", "n_feat", "age_sd_y", "r2", "r2_ci_low", "r2_ci_high",
           "mae_y", "pearson_r", "p_value", "residual_sd_y", "prediction_sd_y"]]
    b.to_csv(os.path.join(OUT, "Additional_file_2_band_performance.csv"), index=False)
    print(f"AF2: {b.shape[0]} rows")


def build_af3():
    l = pd.read_csv(os.path.join(RES, "m1_multiclock/loso_projection.csv"))
    l["mode"] = l["mode"].map(MODE_MAP).fillna(l["mode"])
    l = l.rename(columns={"mode": "healthy_definition", "n": "n_samples",
                          "r_proj": "within_study_r", "p": "p_value",
                          "bias": "bias_y", "mae": "mae_y"})
    l.to_csv(os.path.join(OUT, "Additional_file_3_loso_projection.csv"), index=False)
    print(f"AF3: {l.shape[0]} rows")


def build_af4():
    c = pd.read_csv(os.path.join(RES, "m1_multiclock/interclock_concordance.csv"))
    c = c.rename(columns={"impl_a": "implementation_a", "impl_b": "implementation_b",
                          "r": "pearson_r_oof", "n": "n_samples"})
    c.to_csv(os.path.join(OUT, "Additional_file_4_interclock_concordance.csv"), index=False)
    print(f"AF4: {c.shape[0]} rows")


def build_af5():
    d = pd.read_csv(os.path.join(RES, "m1_metaphlan/deviation_per_sample.csv"))
    d = d.rename(columns={"sampleID": "sample_id", "D": "deviation_D",
                          "is_case": "case", "age": "age_y"})
    if "case" in d:
        d["case"] = d["case"].astype(int)
    d.to_csv(os.path.join(OUT, "Additional_file_5_deviation_per_sample.csv"), index=False)
    print(f"AF5: {d.shape[0]} rows")


def build_af6():
    src = os.path.join(RES, "m1_multiclock/figS1_concordance.pdf")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, "Additional_file_6_Figure_S1.pdf"))
        print("AF6: copied")
    else:
        print("AF6: [missing] figS1_concordance.pdf")


def build_readme():
    txt = """Additional files 1-7 — content and column definitions
Manuscript: "..." (Microbiome, submitted)
Generated by scripts/m1_export_additional_files.py

All tables are UTF-8 CSV with a header row. Sample counts refer to the
de-duplicated analysis panel (74 studies, 14,898 samples, 2,022 species);
the panel was obtained by intersecting each per-study MetaPhlAn profile with
the curatedMetagenomicData sample metadata on the composite key
study::sampleID (177 sample IDs collide across studies and were separated
this way).

AF1 Additional_file_1_cohort_survey.csv   (74 rows, one per study)
  study                     study label in curatedMetagenomicData
  pmid                      PubMed ID of the original publication
  n_samples_analysed        samples from this study in the analysis panel
  n_with_exact_age          samples with a numeric age
  age_mean_y/age_sd_y       mean and SD of age (years)
  age_min_y/age_max_y       age range (years)
  n_case/n_control          case/control counts (study_condition based)
  n_species_detected        species with non-zero abundance in this study
  self_trainable            yes if n >= 60 and age SD >= 10 y
  n_self_train/...          within-cohort 5-fold CV of a self-trained Ridge clock
  n_sequence_accessions     number of distinct ENA/SRA accessions
  accession_example         one representative accession (see AF7 for all)

AF2 Additional_file_2_band_performance.csv   (126 rows)
  healthy_definition  loose (study_condition contains healthy/control) or strict (disease == healthy)
  impl                EN, RF, GBM, DNN, SVR, RFg
  cv_scheme           random 5-fold, study-grouped 5-fold, or transfer (out-of-band)
  band                age band; "A -> B" denotes a transfer experiment
  r2/r2_ci_low/r2_ci_high   R2 with 95% bootstrap CI (400 resamples)
  mae_y/pearson_r           MAE (years) and Pearson r
  n_samples/n_feat/...      sample size, feature count, age SD, residual and prediction SD

AF3 Additional_file_3_loso_projection.csv   (264 rows = 44 studies x 6 implementations)
  within_study_r      Pearson r between predicted and chronological age inside the held-out study
  bias_y              mean(predicted - chronological) in years
  mae_y, n_samples, p_value

AF4 Additional_file_4_interclock_concordance.csv   (15 rows)
  pairwise Pearson correlation of out-of-fold age predictions

AF5 Additional_file_5_deviation_per_sample.csv   (323 rows)
  deviation_D   healthy-manifold deviation D (diagonal Mahalanobis distance in PCA space)
  case          1 = liver disease case, 0 = control
  age_y, study

AF6 Additional_file_6_Figure_S1.pdf   Figure S1 (heatmap of AF4)

AF7 Additional_file_7_sample_accessions.csv   (14,898 rows)
  study, sample_id, pmid, NCBI_accession (ENA/SRA run or project), body_site
  accession_available  "no" for 2,836 samples for which curatedMetagenomicData
                       itself provides no run accession (concentrated in 10
                       studies, e.g. LifeLinesDeep_2016, YachidaS_2019,
                       VilaAV_2018); for these the original publication (pmid)
                       is the only route to the raw reads.
"""
    with open(os.path.join(OUT, "README_additional_files.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("README written")


if __name__ == "__main__":
    X, meta = build_af1()
    build_af7(meta, np.asarray(meta.index))
    build_af2(); build_af3(); build_af4(); build_af5(); build_af6(); build_readme()
    print("done ->", OUT)
