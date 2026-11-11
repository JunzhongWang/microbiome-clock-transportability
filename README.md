# microbiome-clock-transportability

Analysis code for the manuscript *"Measurement validity of gut microbiome ageing
clocks in adult chronic liver disease: early-life-driven accuracy, cross-cohort
projection bias, and a pre-projection calibration gate"* (submitted to
**Microbiome**).

The study asks whether published gut-microbiome "ageing clocks" (microbiome
biological age gap, mBAG) can be used as outcome measures in adult chronic liver
disease. It is a re-analysis of public data plus six re-implementations of the
clock, and it reports four results: (i) the published performance is reproducible
only across the full age range; (ii) predictive power collapses inside the
40–70-year band and is higher in children than in adults; (iii) cross-cohort
projection carries a study-level bias SD of 11–14 years, larger than the effect
sizes it is meant to measure; (iv) a healthy-manifold deviation metric that uses
no age information retains a case–control signal in the one liver cohort where
mBAG can be computed.

---

## Project name and identifiers

| Field | Value |
|---|---|
| Project name | microbiome-clock-transportability |
| Project home page | https://github.com/`<account>`/microbiome-clock-transportability |
| Archived version | https://doi.org/10.5281/zenodo.`<XXXXXXX>` (v1.0.0) |
| Operating system | Platform independent (tested on macOS 15 arm64, Linux x86_64) |
| Programming language | Python 3.13 (all analyses); R 4.1 (data export only) |
| Other requirements | numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1 |
| License | MIT (see `LICENSE`) |
| Restrictions for non-academics | None |

---

## Data

No new sequencing data were generated. All abundance profiles come from
**curatedMetagenomicData 3.14** (Bioconductor 3.19), package DOI
[10.18129/B9.bioc.curatedMetagenomicData](https://doi.org/10.18129/B9.bioc.curatedMetagenomicData),
citation: Pasolli et al., *Nat Methods* 2017;14:1023–1024,
doi:10.1038/nmeth.4468.

* Analysis panel: **74 studies, 14,898 stool metagenomes, 2,022 species**.
* Samples were keyed by `study::sampleID`. 177 sample IDs collide across studies
  (LeChatelierE_2013 and NielsenHB_2014 both use `MH0001…MH0177`); the composite
  key removes this collision and takes the total from 15,075 to 14,898.
* Healthy subsets with exact age: 7,657 (loose definition,
  `study_condition` contains "healthy" or "control") and 7,534 (strict,
  `disease == "healthy"`).
* Per-study accession numbers are in
  `additional_files/Additional_file_1_cohort_survey.csv` (study level) and
  `Additional_file_7_sample_accessions.csv` (sample level).

**Known data caveat.** `data/metaphlan/per_study/` as distributed with the
original working copy contains 80 per-study tables, but 6 of them
(`BedarfJR_2017`, `FrankelAE_2017`, `LeeKA_2022`, `MetaCardis_2020_a`,
`PetersBA_2019`, `WindTT_2020`) come from a later curatedMetagenomicData release
whose sample IDs are absent from the 3.14 metadata used here. They contribute
**0 samples** to every analysis in the manuscript. Do not count them: the
analysed panel is 74 studies, not 80.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Downloading and re-exporting the raw curatedMetagenomicData objects requires R
and roughly 4 GB of disk:

```bash
python scripts/fetch_curated_resources.py     # not needed if data/metaphlan exists
Rscript scripts/m1_export_metaphlan.R         # .rda -> per-study wide tables
```

---

## Reproduce

Run in this order. Each step writes to `results/`; the manuscript figure or
table it produces is noted in brackets. All steps are deterministic (seed 42)
except the bootstrap CIs, which use 400 resamples with the same seed.

| # | Command | Output | Manuscript item |
|---|---|---|---|
| 1 | `python scripts/m1_cohort_survey.py` | `results/m1_metaphlan/cohort_survey.csv`, `intracohort_clock.csv` | Additional file 1, Results "Within-cohort self-trained clocks" |
| 2 | `python scripts/m1_age_stratified_clock.py` | `age_stratified_clock.csv`, `clock_transportability.csv` | Figure 1, Table 1–2 |
| 3 | `python scripts/m1_multiclock_run.py EN,GBM` then `RF cv`, `RF loso`, `DNN`, `SVR`, `RFg` | `results/m1_multiclock/part_*.csv` | Figure 2, Additional files 2–4 |
| 4 | `python scripts/m1_multiclock_merge.py` | `band_performance.csv`, `loso_projection.csv`, `summary.csv`, `interclock_concordance.csv` | Tables 2–3, Figure 3 |
| 5 | `python scripts/m1_deviation_metric.py` | `deviation_per_sample.csv`, `deviation_scores.csv` | Figure 5, Table 4, Additional file 5 |
| 6 | `python scripts/m1_export_additional_files.py` | `results/additional_files/` | Additional files 1–7 |
| 7 | `python scripts/m1_figures_english.py`, `m1_multiclock_figures.py` | `en_fig1–4.png/pdf`, `fig2_multiclock.*`, `figS1_concordance.pdf` | Figures 1–5, Figure S1 |

Step 3 is the only expensive one: about 40–60 min per implementation on 8 cores
(Random Forest dominates). The runner accepts one implementation at a time so the
work can be split across sessions; a feature cache (`_feats_cache.npz`) is
reused across runs. Thread count is capped by `M1_MAX_JOBS` (default 4) to keep
the machine responsive; results are independent of this value.

---

## Layout

```
scripts/
  m1_export_metaphlan.R          curatedMetagenomicData .rda -> per-study tables
  m1_cohort_survey.py            per-study sample size, age span, self-trained clock
  m1_age_stratified_clock.py     age-band CV, projection, calibration gate
  m1_multiclock_validation.py    library: loaders, six implementations, metrics
  m1_multiclock_run.py           batched runner (one implementation per call)
  m1_multiclock_merge.py         merge parts, bootstrap CIs, summary tables
  m1_deviation_metric.py         healthy-manifold deviation D
  m1_export_additional_files.py  Additional files 1-7
  m1_figures_english.py          Figures 1-4 (English labels)
  m1_multiclock_figures.py       Figure 2 panels, Figure S1
additional_files/                the seven supplementary files as submitted
```

---

## Definitions used throughout

* **Age bands** — all 0–107, children <18, young 18–40, middle-aged 40–70,
  older ≥60, adults ≥18.
* **Random CV** — 5-fold, stratified on age quintiles, comparable to the
  published protocol.
* **Study-grouped CV** — `GroupKFold` on study identity, the honest estimate for
  an unseen cohort.
* **Strictly out-of-band transfer** — training and test age bands are disjoint;
  an earlier "all → middle-aged" design was discarded because its training set
  contained the test band (in-sample evaluation inflated R² to 0.67).
* **LOSO projection** — train on all studies but one (44 studies with n ≥ 20),
  predict the held-out study, report within-study r, bias, MAE.
* **Calibration gate** — a projection is accepted only if within-study
  predicted-vs-chronological r ≥ 0.3.
* **Deviation D** — diagonal Mahalanobis distance from the healthy centroid in
  the first 30 principal components (57.3% of variance); uses no age.

---

## Citation

If you use this code, please cite the manuscript and the archived release:

```
<Authors>. Measurement validity of gut microbiome ageing clocks in adult
chronic liver disease: early-life-driven accuracy, cross-cohort projection
bias, and a pre-projection calibration gate. Microbiome (submitted).
Code: https://doi.org/10.5281/zenodo.<XXXXXXX>
```

## License

MIT. See `LICENSE`.
