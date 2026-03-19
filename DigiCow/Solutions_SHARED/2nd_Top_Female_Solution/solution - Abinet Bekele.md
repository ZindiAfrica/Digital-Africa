# Solution Documentation: DigiCow Farmer Training Adoption Challenge

**Competition:** [DigiCow Farmer Training Adoption Challenge](https://zindi.africa/competitions/digicow-farmer-training-adoption-challenge)  
**Notebook:** `FinalSub.ipynb`  
**Author:** Abinet Bekele  

This document describes the solution implemented in `FinalSub.ipynb`, following the [Zindi documentation guideline](https://zindi.africa/learn/documentation-guideline).

---

## 1. Overview and Objectives

### Purpose

The solution predicts the **probability** that a farmer will adopt a DigiCow-supported practice within **7**, **90**, and **120 days** of their first training, using only information available at the time of training. The goal is to enable DigiCow and partners to prioritise follow-ups, tailor support, and design stronger extension strategies.

### Problems Addressed

- **Low and uneven adoption rates** after agricultural training.
- **Early prediction** of adoption to target follow-up and support.
- **Multi-horizon probability estimation** for three time windows (7, 90, 120 days).

### Objectives and Expected Outcomes

- Output predicted probabilities for each target (7-, 90-, 120-day adoption) in the format required by the challenge.
- Optimise for the evaluation metric: **75% Log Loss** and **25% ROC-AUC** (weighted average).
- Produce a reproducible, open-source R pipeline that runs from raw data to submission file.

---

## 2. Architecture Overview

High-level flow of the solution:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTRACT                                            │
│  Train.csv, Test.csv, Prior.csv → data.table (R)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSFORM (Feature Engineering)                    │
│  1. Parse topics_list (JSON) → topic vectors                                 │
│  2. Trainer history (smoothed prior adoption rate by trainer/day)            │
│  3. Topic-driven: top-50 topic flags, diversity, popularity                 │
│  4. TF-IDF: word (1,2)-gram + char 3–5-gram + SVD themes                     │
│  5. Keyword categories: dairy, poultry, crop, health, nutrition, etc.        │
│  6. Farmer history: visit_no, gaps, recent counts, topic rollups, Jaccard     │
│  7. Time: month sin/cos, peak/off months, quarter                            │
│  8. Categorical encoding + geo/trainer/group aggregates                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOAD (Matrices)                                    │
│  X_train_all, X_test_all, y_all (3 targets)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL TRAINING                                     │
│  Per target: 5-fold group CV (by farmer_name)                                │
│  Models: XGBoost + LightGBM → weighted ensemble (target-specific weights)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE & POST-PROCESS                           │
│  Test predictions → probability sharpening (gamma=1.2)                     │
│  Monotonicity constraints: 7 ≤ 90 ≤ 120 for AUC and LogLoss columns           │
│  Output: finsub.csv (ID, Target_07/90/120_AUC, Target_07/90/120_LogLoss)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ETL Process

### Extract

- **Data sources:** Challenge data files `Train.csv`, `Test.csv`, and `Prior.csv`. The notebook assumes they are available under a configurable `base_dir` (e.g. `/content` when run on Google Colab; data can be provided via a zip from Google Drive).
- **Formats:** CSV; date column `training_day` is read and converted to `IDate`.
- **Volume / frequency:** Single batch load; no streaming. Extraction is one-off per run.

### Transform

Transformation is done in a single combined dataset (`all_dt`) built from Train, Test, and Prior, then split back for modelling.

1. **Topic parsing**  
   - `topics_list` is parsed from string (JSON-like) into a list of topic strings per row (`unpack_topics`). Handles escaped quotes, `None` → `null`, and empty/NA.

2. **Trainer history (prior adoption)**  
   - From **Prior** only, a smoothed historical adoption rate for the 7-day target is computed per `(trainer, training_day)` using cumulative positives and counts and a Bayesian-style smoothing constant `C = 15`. This is joined to all rows via `attach_trainer_history` (roll join on `trainer`, `training_day`) to get `trn_hist_sm07`.

3. **Topic-driven features**  
   - Top-50 most frequent topics (that also appear in test) become binary flags `tp_f_*`.  
   - `tp_div_ratio`: diversity of topics per row (unique count / length).  
   - `tp_pop_mean`: mean frequency of topics in the row.  
   - `farm_prev_seen`: whether the farmer was seen in prior events.  
   - `tp_text_blob`: concatenated topic string for TF-IDF.

4. **TF-IDF features**  
   - **Word (1,2)-gram TF-IDF** (`text2vec`): vocabulary from tokenizer `tok_word_12_vec`, max 20 terms, min_df=5; L2-normalised; selected columns exported as `w_tfx_*`.  
   - **Char 3–5-gram TF-IDF**: max 8 terms, min_df=5; selected indices as `c_tfx_*`.  
   - **SVD themes**: larger word TF-IDF (max 100 terms, min_df=1), then 5-component SVD via `irlba`; sign-flipped for stability; components as `tp_svd_0` … `tp_svd_4`.

5. **Keyword category features**  
   - Predefined keyword lists for: dairy, poultry, crop, health, nutrition, breeding, ruminant, digital, business, energy, hygiene, livestock.  
   - For each category: count of topic hits (`ct_*`) and binary “has hit” (`hx_*`).  
   - Derived: `tp_cat_main` (dominant category), `tp_cat_n` (number of categories hit), and aggregates `trn_cat_cnt`, `ward_cat_cnt`, `cty_cat_cnt`.

6. **Farmer history features**  
   - Sorted by `farmer_name`, `training_day`: `visit_no`, `gap_days` since previous event, `first_evt`, `seen_trn`, `seen_grp`.  
   - Rolling counts: `prev_in_7`, `prev_in_90`, `prev_in_120` (events in last 7/90/120 days per farmer).  
   - Per-event topic stats: `new_tp_cnt`, `seen_tp_cnt`, `evt_tp_uniq`; cumulative unique topics `farm_uniq_cum`; `new_tp_ratio`; Jaccard overlap with past topics `jacc_past`.  
   - Event size bin by `ward_day_n`; trainer–ward counts and share; `al_dairy`, `al_crop` (category count × trainer mean).  
   - Lexicographic encoding of topic blob: `enc_tp_text`.

7. **Time and aggregate stats**  
   - Month sin/cos, `is_peak_m`, `is_off_m`, `qtr`.  
   - Topic length `tp_len`; trainer/category means (`trn_tp_mean1`, `trn_tp_divm`, `cat_tp_mean`).  
   - Topic entropy per row `tp_entropy`; `trn_tp_rate`.

8. **Categorical encoding and aggregates**  
   - Categorical columns (e.g. `gender`, `age`, `registration`, `belong_to_cooperative`, `county`, `subcounty`, `ward`, `trainer`, `group_name`, `has_topic_trained_on`) are label-encoded with `lex_encode0`.  
   - Geo aggregates: `geo_*_n`, `geo_*_coopm` for county/subcounty/ward.  
   - Trainer stats: `trn_evt_n`, `trn_tp_mean2`, `trn_trained_rate`.  
   - Event counts by trainer/county/ward/group; group-level stats (`grp_n`, `grp_coop_m`, `grp_tp_mean`, `grp_trained_m`).  
   - Interaction encodings: `enc_age_sex`, `enc_coop_trn`, `enc_reg_coop`.

### Load

- **Target storage:** In-memory R objects.  
- **Outputs:**  
  - `train_dt` / `test_dt`: feature tables with targets (train) or without (test).  
  - `X_train_all`, `X_test_all`: numeric matrices; `y_all`: three target columns.  
- **Dropped for modelling:** `topics_list`, `tp_vec`, `tp_text_blob`, `ID`, `farmer_name`, `training_day`, `rid___`, and target columns from test.  
- **Feature set:** All remaining columns form `feat_cols` used for training and inference.

---

## 4. Data Modelling

### Model Types and Assumptions

- **Task:** Binary probability estimation for each of three adoption windows (7, 90, 120 days).  
- **Assumption:** Adoption probabilities are non-decreasing over time (7 ≤ 90 ≤ 120); enforced only in post-processing.  
- **Theoretical basis:** Gradient boosting (XGBoost, LightGBM) for tabular features with binary cross-entropy loss.

### Feature Selection, Engineering, and Normalisation

- **Selection:** No explicit feature selection; all engineered features in `feat_cols` are used. TF-IDF and SVD reduce dimensionality of text.  
- **Engineering:** As in the Transform section (topics, TF-IDF, SVD, keyword categories, farmer history, time, categorical and aggregate features).  
- **Normalisation:** TF-IDF vectors are L2-normalised; numeric features are used as-is by tree models (no additional scaling).

### Training Process

- **Algorithms:** XGBoost (`xgboost`) and LightGBM (`lightgbm`), both with binary logistic objective.  
- **Hyperparameters:**  
  - **LightGBM:** 1500 iterations, learning_rate 0.08, max_depth 4, feature_fraction 0.6, bagging_fraction 0.7, min_data_in_leaf 20, lambda_l1/l2 0.1.  
  - **XGBoost:** 900 rounds, eta 0.01, max_depth 8, colsample_bytree 0.7, subsample 0.8.  
- **Validation:** 5-fold **group** cross-validation by `farmer_name` (same farmer never in both train and validation in a fold). Seeds vary by fold for reproducibility.  
- **Early stopping:** LightGBM only; 50 rounds.

### Ensemble Weights (per target)

- **adopted_within_07_days:** AUC blend 0.5 XGB + 0.5 LGB; LogLoss blend 0.3 XGB + 0.7 LGB.  
- **adopted_within_90_days:** AUC 0.6 XGB + 0.4 LGB; LogLoss 0.5 XGB + 0.5 LGB.  
- **adopted_within_120_days:** AUC 0.6 XGB + 0.4 LGB; LogLoss 0.55 XGB + 0.45 LGB (approximate; see notebook for exact weights).

### Evaluation Metrics

- **Primary (in notebook):** Binary log loss on out-of-fold predictions.  
- **Challenge metric:** Weighted combination of Log Loss (75%) and ROC-AUC (25%) on the submission format.

### Model Validation

- Performance is measured via OOF log loss per target and per model (XGB vs LGB), printed in the notebook.  
- Final submission is validated by monotonicity constraints and probability sharpening.

---

## 5. Inference

### Deployment Context

- **Environment:** Single run in an R environment (e.g. Google Colab). No separate serving API.  
- **Input:** Same `Test.csv` (and same `Train.csv` / `Prior.csv` for feature computation).  
- **Output:** One submission file `finsub.csv`.

### How New Data Is Used

- Test rows are merged with Train and Prior for feature construction (e.g. trainer history, topic vocabulary, SVD).  
- The same `feat_cols` and same preprocessing functions are applied so that test feature matrix `X_test_all` is aligned with training.  
- Each of the three targets gets an ensemble of 5 XGB + 5 LGB models (one per fold); test predictions are averaged per model type, then blended per target.

### Output Interpretation

- Each row is one test `ID`.  
- Columns: `Target_07_AUC`, `Target_90_AUC`, `Target_120_AUC`, `Target_07_LogLoss`, `Target_90_LogLoss`, `Target_120_LogLoss` — all in [0, 1].  
- For the challenge, the platform uses the appropriate columns for Log Loss and ROC-AUC evaluation.

### Post-Processing

- **Probability sharpening:** `prob_sharpen(..., gamma = 1.2, eps = 1e-16)` to make probabilities more confident while staying in (0, 1).  
- **Monotonicity:**  
  - `Target_90_AUC := pmax(Target_07_AUC, Target_90_AUC)`  
  - `Target_120_AUC := pmax(Target_90_AUC, Target_120_AUC)`  
  - Same for LogLoss columns so that 7-day ≤ 90-day ≤ 120-day.

### Model Updates and Retraining

- The notebook does not implement versioning or automated retraining. To update: re-run the notebook with new data; ensure the same R package versions and seeds for reproducibility.

---

## 6. Run Time

- **Per script/notebook:** Not formally measured in the notebook. Approximate expectations:  
  - Data load and feature engineering: on the order of minutes (depending on data size and hardware).  
  - TF-IDF and SVD: moderate (vocabulary build + irlba).  
  - Model training: 3 targets × 5 folds × 2 models → 30 model fits; typically several minutes to tens of minutes in total.  
- **Recommendation:** Add explicit timing (e.g. `system.time()` or `tictoc`) around major sections and report in documentation for code review.

---

## 7. Performance Metrics

- **Challenge metrics:**  
  - **Log Loss (75% weight):** Penalises confident wrong predictions.  
  - **ROC-AUC (25% weight):** Ranking of adopters vs non-adopters.  
  - Final score = weighted average of the two.  
- **Reported in notebook:** OOF binary log loss per target for XGB and LGB (e.g. ~0.0358 for 7-day target).  
- **Public/private scores:** To be filled by the author with actual leaderboard (public and private) scores after submission and leaderboard freeze.  
- **Other metrics:** Only log loss is printed; ROC-AUC can be added on OOF predictions for alignment with the challenge metric.

---

## 8. Error Handling and Logging

- **Topic parsing:** `unpack_topics` uses `tryCatch`; on parse failure it returns `character(0)`.  
- **Missing/NA:** Character inputs to tokenizers and encoders replace NA with `"nan"` or `""` as appropriate; numeric gaps use sentinel `-99L` where needed.  
- **DTM row alignment:** `expand_dtm_rows` restores empty-document rows so TF-IDF matrix row order matches the full document vector.  
- **Logging:** No formal logging framework; progress is via `cat(sprintf(...))` for target and OOF log loss.  
- **Warnings:** The notebook may show warnings (e.g. XGBoost `watchlist` renamed to `evals`, or data.table `..has_cols`); they do not stop execution but should be addressed for a clean code review.

---

## 9. Maintenance and Monitoring

- **Dependencies:** R packages: `data.table`, `lubridate`, `Matrix`, `text2vec`, `irlba`, `lightgbm`, `xgboost`, `jsonlite`. Versions should be pinned for reproducibility.  
- **Data assumptions:** Expect columns and schema of the challenge `Train.csv`, `Test.csv`, and `Prior.csv` (including `topics_list`, `training_day`, `farmer_name`, `trainer`, etc.). If the host changes schema, topic or keyword lists and feature steps must be updated.  
- **Scaling:** Single-machine only; for much larger data, consider sampling, chunked TF-IDF, or distributed boosting.  
- **Lifecycle:** Re-run the notebook when new test data or new training data is released; keep seeds and hyperparameters fixed unless intentionally retuning.  
- **Known limitations:**  
  - Google Drive / `gdown` and paths assume a Colab-like setup; for local runs, set `base_dir` and ensure CSV files (and optional zip) are in place.  
  - Custom packages beyond CRAN are not used; all packages should be publicly available to comply with challenge rules.

---

## 10. Submission File Format

The submission file `finsub.csv` conforms to the challenge requirement:

| Column            | Description                                      |
|-------------------|--------------------------------------------------|
| ID                | Test set identifier                              |
| Target_07_AUC     | P(adopt within 7 days)  for AUC evaluation       |
| Target_07_LogLoss | P(adopt within 7 days)  for Log Loss evaluation  |
| Target_90_AUC     | P(adopt within 90 days) for AUC evaluation       |
| Target_90_LogLoss | P(adopt within 90 days) for Log Loss evaluation  |
| Target_120_AUC    | P(adopt within 120 days) for AUC evaluation      |
| Target_120_LogLoss| P(adopt within 120 days) for Log Loss evaluation  |

All probability columns are in [0, 1], with monotonicity 7 ≤ 90 ≤ 120 enforced.

---

## 11. Reproducibility

- **Seeds:** Fold construction uses `seed = 1L` in `make_group_folds`; SVD uses `set.seed(42)`; XGBoost/LightGBM use fold-dependent seeds.  
- **Data:** Same Train/Test/Prior CSVs must be used to reproduce scores.  
- **Environment:** R version and package versions should be recorded (e.g. in a `sessionInfo()` or `renv` lockfile) for code review.

---

*This documentation was written to support the Zindi code review process and to clarify the purpose, design, and usage of the solution in `FinalSub.ipynb`.*
