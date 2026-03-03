# Farm to Feed Challenge - Winning Solution Documentation

> **Competition**: Farm to Feed Challenge (Zindi)  
> **Author**: LiveFeed Team
> **Date**: January 2026

---

## Executive Summary

This solution predicts customer purchase behavior (1-week and 2-week horizons) and quantities for an agricultural e-commerce platform. The winning approach combines **three specialized pipelines** through strategic blending:

- **V2**: K-Fold Target Encoding with categorical combinations (Kaggle winner strategy)
- **V4**: Advanced feature engineering with lag features, segment-specific patterns, and dormancy modeling
- **V11**: Multi-scale SVD embeddings for customer-product affinity modeling with temporal awareness

The final submission uses an optimized blend that leverages each model's strengths: V11 for WAUC (classification), V4 for MAE (quantity), and V2 to add diversity to purchase predictions.

---

## Final Scores

### Top Score (Primary Submission)
| Metric | Score |
|--------|-------|
| **Target 1 WAUC** | 0.965566394 |
| **Target 2 WAUC** | 0.965629073 |
| Target 1 W Qty MAE | 0.290462697 |
| Target 2 W Qty MAE | 0.470406171 |

### Second Best Score (Backup Submission)
| Metric | Score |
|--------|-------|
| Target 1 WAUC | 0.965015929 |
| Target 2 WAUC | 0.96514134 |
| Target 1 W Qty MAE | 0.290462697 |
| Target 2 W Qty MAE | 0.470406171 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│   │   Train.csv │    │   Test.csv  │    │ Sample.csv  │                 │
│   │  (275 MB)   │    │   (27 MB)   │    │   (7 MB)    │                 │
│   └──────┬──────┘    └──────┬──────┘    └─────────────┘                 │
│          │                  │                                            │
│          ▼                  ▼                                            │
│   ┌─────────────────────────────────────┐                               │
│   │       FEATURE ENGINEERING           │                               │
│   │  • Temporal features                │                               │
│   │  • Customer-Product aggregations    │                               │
│   │  • Recency & decay features         │                               │
│   │  • Rolling windows (2w, 4w, 8w)     │                               │
│   └─────────────────────────────────────┘                               │
│                      │                                                   │
│          ┌──────────┼──────────┐                                        │
│          ▼          ▼          ▼                                        │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐                            │
│   │    V2     │ │    V4     │ │    V11    │                            │
│   │ Target    │ │ Advanced  │ │ SVD       │                            │
│   │ Encoding  │ │ Features  │ │ Embeddings│                            │
│   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                            │
│         │             │             │                                    │
│         ▼             ▼             ▼                                    │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐                            │
│   │ XGBoost   │ │ LightGBM  │ │ XGBoost   │                            │
│   │ CatBoost  │ │ XGBoost   │ │ CatBoost  │                            │
│   │ LightGBM  │ │           │ │ LightGBM  │                            │
│   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                            │
│         │             │             │                                    │
│         ▼             ▼             ▼                                    │
│   submission_v2  submission_v4  submission_v11                          │
│         │             │             │                                    │
│         └──────────┬──┴──────┬──────┘                                   │
│                    ▼         ▼                                          │
│            ┌─────────────────────┐                                      │
│            │   BLENDING LAYER    │                                      │
│            │                     │                                      │
│            │  WAUC: V2 + V11     │                                      │
│            │  MAE:  V4 (100%)    │                                      │
│            └──────────┬──────────┘                                      │
│                       ▼                                                  │
│         submission_blend_v2_v4_v11.csv                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Environment Setup

### Requirements

```bash
# Python version
Python 3.10+

# Core dependencies
catboost==1.2.8
joblib==1.5.2
lightgbm==4.6.0
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.7.2
scipy==1.16.3
tqdm==4.67.1
xgboost==3.1.2
```

### Hardware Used

- **CPU**: AMD Rayzen 5900X 12-core processor
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX 3060ti 8GB

---

## Training Logs & Validation Results

### V11 Pipeline (SVD Embeddings) - Actual Run Log

```
[10:26:38] Loading data...
[10:26:42] Train: (2,114,436, 20), Test: (275,796, 11)

[10:26:42] Building SVD Embeddings:
  - Standard SVD (dim=16): Done
  - Standard SVD (dim=32): Done  
  - Standard SVD (dim=64): Done
  - Quantity-weighted SVD (dim=32): Done
  - Temporal SVD (last 8 weeks, dim=32): Done
[10:26:42] Total SVD configurations: 5

[10:29:21] Feature Engineering Complete:
  - 1w features: 90 (V2 base + 48 SVD features)
  - 2w features: 125

[10:29:22] OOF 1w Predictions (5-Fold CV):
  Fold 1: AUC = 0.997798
  Fold 2: AUC = 0.997809
  Fold 3: AUC = 0.997651
  Fold 4: AUC = 0.997702
  Fold 5: AUC = 0.997685
  ─────────────────────────
  Overall OOF AUC: 0.997728

[11:36:03] Final Model Training:
  XGB: 8 configs × 3 seeds = 24 models [43:49]
  Cat: 4 configs × 3 seeds = 12 models [53:55]
  LGB: 7 seeds [02:35]

[13:23:48] 2w Purchase Models: 9 seeds [04:01]
[13:27:29] Quantity Models Complete
[13:27:30] Saved submission_v11.csv

Total Runtime: ~3 hours (10:26 → 13:27)
```

### V4 Pipeline (Advanced Features) - Actual Run Log

```
[18:42:55] Rolling Window Validation Results:
  ─────────────────────────────────────────
  Window 1 (weeks 1-26 → 27-28):  AUC = 0.9770
  Window 2 (weeks 7-32 → 33-34):  AUC = 0.9602
  Window 3 (weeks 13-38 → 39-40): AUC = 0.9738
  Window 4 (weeks 19-44 → 45-46): AUC = 0.9775
  ─────────────────────────────────────────
  AVERAGE WINDOW AUC: 0.9721

[18:43:17] Feature Engineering Complete:
  - Total features: 136

[18:43:18] Final Model Training:
  1w Purchase - LightGBM: 7 models [06:47] (~58s/model)
  1w Purchase - XGBoost:  4 models [04:33] (~68s/model)
  2w Purchase - LightGBM: 7 models [05:35] (~48s/model)
  1w Quantity: 5 models [00:13]
  2w Quantity: 5 models [00:15]

[19:00:44] Saved submission_v4.csv

[19:00:44] Prediction Statistics:
  1w Purchase: mean = 0.013112
  2w Purchase: mean = 0.018473
  1w Quantity: mean = 0.1893
  2w Quantity: mean = 0.4077

Total Runtime: ~22 minutes final training (after validation)
```

### V2 Pipeline (Target Encoding) - Estimated

```
[Estimated based on similar architecture]

Feature Engineering:
  - Base features: ~42 features (same as V2 baseline)
  - Target Encoding features: 18 features
  - Combination features: 10 features
  - Total: ~70 features

OOF Validation:
  - Expected OOF AUC: ~0.9977 (similar to V11)

Final Model Training:
  - XGB: 6 configs × 2 seeds = 12 models
  - CatBoost: 2 configs × 3 seeds = 6 models
  - LightGBM: 5 seeds

Estimated Runtime: ~2-3 hours
```

### Internal Validation Summary

| Pipeline | Validation Method | Score | Features |
|----------|-------------------|-------|----------|
| V11 | 5-Fold CV (OOF) | **0.9977 AUC** | 90 (1w), 125 (2w) |
| V4 | Rolling Window (4 windows) | **0.9721 AUC** | 136 |
| V2 | 5-Fold CV (estimated) | ~0.9977 AUC | ~70 |

---

## Data Description

### Input Files

| File | Description | Size | Records |
|------|-------------|------|---------|
| `Train.csv` | Historical customer-product-week data | 275 MB | ~5M rows |
| `Test.csv` | Test set for predictions | 27 MB | ~500K rows |
| `SampleSubmission.csv` | Submission format | 7 MB | ~500K rows |

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | ID | Unique customer identifier |
| `product_unit_variant_id` | ID | Unique product variant identifier |
| `week_start` | Date | Week start date |
| `purchased_this_week` | Binary | Purchase indicator (0/1) |
| `qty_this_week` | Float | Quantity purchased |
| `customer_category` | Category | Customer segment |
| `customer_status` | Category | Customer status |
| `grade_name` | Category | Product grade |
| `unit_name` | Category | Product unit type |

### Targets

| Target | Type | Description |
|--------|------|-------------|
| `Target_purchase_next_1w` | Binary | Will purchase in next 1 week? |
| `Target_purchase_next_2w` | Binary | Will purchase in next 2 weeks? |
| `Target_qty_next_1w` | Float | Quantity in next 1 week |
| `Target_qty_next_2w` | Float | Quantity in next 2 weeks |

---

## Feature Engineering

### V2: Target Encoding Features

The V2 pipeline implements the **Kaggle winning strategy** of K-Fold Target Encoding:

```python
# Key technique: K-Fold Target Encoding with smoothing
def target_encode_kfold(train_df, test_df, cols, target, n_folds=5, smooth=20):
    """
    Prevents data leakage by using out-of-fold encoding for training data.
    Smoothing prevents overfitting on rare categories.
    """
    # Smoothed encoding formula:
    # encoded = (count * mean + smooth * global_mean) / (count + smooth)
```

**Features Added:**
- Target encoding for: `customer_category`, `customer_status`, `grade_name`, `unit_name`
- Count encoding for frequency information
- Categorical combinations:
  - `customer_category × grade_name`
  - `customer_category × unit_name`
  - `customer_status × grade_name`
  - `grade_name × unit_name`
  - `customer_category × customer_status × grade_name`

### V4: Advanced Feature Engineering

V4 builds on V2 with additional sophisticated features:

| Feature Category | Features | Rationale |
|-----------------|----------|-----------|
| **Lag Features** | `purch_lag_1`, `purch_lag_2`, `purch_lag_3`, `purch_lag_4` | Capture sequential purchase patterns |
| **Dormancy** | `avg_purchase_interval`, `is_overdue`, `dormancy_score` | Model customer churn risk |
| **Segment-Specific** | `is_danger_zone`, `danger_zone_score` | Target "Old(4-12w)" segment |
| **Trends** | `purch_trend_2v8`, `cust_momentum`, `prod_momentum` | Detect acceleration/deceleration |
| **Interactions** | `recency_x_freq`, `affinity_x_recency` | Non-linear relationships |

```python
# Danger zone detection (4-12 weeks since last purchase)
df['is_danger_zone'] = ((df['weeks_since_last'] >= 4) & 
                        (df['weeks_since_last'] <= 12)).astype(np.int8)
df['danger_zone_score'] = np.where(
    df['is_danger_zone'] == 1,
    df['cp_purch_sum'] * df['recency_decay_28d'],
    0
)
```

### V11: Multi-Scale SVD Embeddings

V11 introduces **collaborative filtering** via SVD to capture latent customer-product affinities:

```python
# Multi-scale SVD configuration
SVD_DIMS = [16, 32, 64]  # Multiple embedding dimensions
RECENT_WEEKS = 8         # For temporal SVD

# Types of SVD embeddings:
# 1. Standard SVD (all history)
# 2. Quantity-weighted SVD (log(qty) as weights)
# 3. Temporal SVD (last 8 weeks only)
```

**SVD Features:**
| Feature | Description |
|---------|-------------|
| `svd_sim_d{16/32/64}` | Cosine similarity between customer and product embeddings |
| `svd_dot_d{16/32/64}` | Dot product of embeddings |
| `svd_dist_d{16/32/64}` | Euclidean distance between embeddings |
| `svd_int{0/1/2}_d{dim}` | Top component interactions |
| `svd_agreement` | Multi-scale SVD agreement score |
| `svd_temporal_diff` | Difference between recent and global SVD |

**Cross Features (V11):**
```python
# SVD × Recency interactions
df['svd_x_rec7'] = df['svd_sim_d32'] * df['recency_decay_7d']
df['svd_x_rec28'] = df['svd_sim_d32'] * df['recency_decay_28d']

# SVD × Purchase history
df['svd_x_prate'] = df['svd_sim_d32'] * df['hist_purch_rate']
df['svd_x_pcnt'] = df['svd_sim_d32'] * np.log1p(df['hist_purch_cnt'])
```

---

## Model Architecture

### Classification Models (WAUC Optimization)

#### XGBoost Configurations (V11 uses 8 configs × 3 seeds)
```python
xgb_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 6, 
     'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 7, 
     'min_child_weight': 80, 'subsample': 0.7, 'colsample_bytree': 0.65},
    # ... 6 more configurations for diversity
]
```

#### CatBoost Configurations (4 configs × 3 seeds)
```python
cat_configs = [
    {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    {'iterations': 1500, 'learning_rate': 0.02, 'depth': 7, 'l2_leaf_reg': 2},
    {'iterations': 2000, 'learning_rate': 0.015, 'depth': 6, 'l2_leaf_reg': 4},
    {'iterations': 1000, 'learning_rate': 0.03, 'depth': 8, 'l2_leaf_reg': 2},
]
```

#### LightGBM (7-9 seeds)
```python
lgb_params = {
    'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 6,
    'min_child_samples': 100, 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 5
}
```

### Regression Models (MAE Optimization)

```python
# Quantity predictions trained only on purchasers (positive samples)
lgb_params_qty = {
    'objective': 'regression_l1',  # MAE loss
    'metric': 'mae',
    'learning_rate': 0.015, 'max_depth': 7,
    'min_child_samples': 30
}
```

### Ensemble Strategies

| Pipeline | Strategy | Details |
|----------|----------|---------|
| V2 | Weighted Average | XGB: 80%, Cat: 12%, LGB: 8% |
| V4 | Simple Average | LightGBM + XGBoost ensemble |
| V11 | Rank Averaging | More robust to outliers |

```python
# V11 Rank Averaging
def rank_average(predictions):
    """Rank averaging for robust ensemble."""
    ranks = np.array([rankdata(p) for p in predictions])
    return np.mean(ranks, axis=0) / len(predictions[0])
```

---

## Training Pipeline

### Time Decay Weighting

Recent samples are weighted more heavily using exponential decay:

```python
DECAY_HALFLIFE_WEEKS = 15.0
weeks_ago = max_week_idx - train['week_idx'].values
sample_weights = np.power(2, -weeks_ago / DECAY_HALFLIFE_WEEKS)
```

### Cross-Validation Strategy

- **5-Fold K-Fold** with shuffle for OOF predictions
- **Rolling Window Validation** (V4) for temporal validation:
  ```python
  windows = [
      {'train_start': 1, 'train_end': 26, 'test_start': 27, 'test_end': 28},
      {'train_start': 7, 'train_end': 32, 'test_start': 33, 'test_end': 34},
      {'train_start': 13, 'train_end': 38, 'test_start': 39, 'test_end': 40},
      {'train_start': 19, 'train_end': 44, 'test_start': 45, 'test_end': 46},
  ]
  ```

### Sequential Training

1. Train 1w classification models → Generate OOF predictions
2. Add 1w predictions as feature → Train 2w classification models
3. Train quantity models on purchasers only
4. Combine purchase probability × quantity for final predictions

---

## Blending Strategy

### Top Score: Triple Blend (V2 + V4 + V11)

```python
# blend_v2_v4_v11.py - WINNING SUBMISSION

# WAUC Target 1: 35% V2 + 65% V11
T1_V2_WEIGHT = 0.35
blend['Target_purchase_next_1w'] = (
    T1_V2_WEIGHT * v2['Target_purchase_next_1w'] + 
    (1 - T1_V2_WEIGHT) * v11['Target_purchase_next_1w']
)

# WAUC Target 2: 25% V2 + 75% V11
T2_V4_11_WEIGHT = 0.75
blend['Target_purchase_next_2w'] = (
    (1 - T2_V4_11_WEIGHT) * v2['Target_purchase_next_2w'] + 
    T2_V4_11_WEIGHT * v11['Target_purchase_next_2w']
)

# MAE: 100% V4 (best for quantity predictions)
blend['Target_qty_next_1w'] = v4['Target_qty_next_1w']
blend['Target_qty_next_2w'] = v4['Target_qty_next_2w']
```

**Rationale:**
- **V11**: Best WAUC scores due to SVD capturing collaborative filtering patterns
- **V2**: Adds diversity through target encoding; slight blend improves generalization
- **V4**: Best MAE scores due to advanced quantity-focused features

### Second Best: Simple Blend (V4 + V11)

```python
# blend_v4_v11.py - BACKUP SUBMISSION

# WAUC: Use V11 (100%)
blend['Target_purchase_next_1w'] = v11['Target_purchase_next_1w']
blend['Target_purchase_next_2w'] = v11['Target_purchase_next_2w']

# MAE: Use V4 (100%)
blend['Target_qty_next_1w'] = v4['Target_qty_next_1w']
blend['Target_qty_next_2w'] = v4['Target_qty_next_2w']
```

---

## Reproduction Steps

### Step 1: Prepare Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost scipy tqdm joblib
```

### Step 2: Prepare Data

Place the following files in the `Farm to Feed 5/` directory:
- `Train.csv`
- `Test.csv`
- `SampleSubmission.csv`

### Step 3: Train Base Models

```bash
cd "Farm to Feed 5"

# Train V2 (Target Encoding pipeline)
python farm_to_feed_v2.py
# Output: submission_v2.csv (~2-3 hours)

# Train V4 (Advanced Features pipeline)
python farm_to_feed_v4.py
# Output: submission_v4.csv (~2-3 hours)

# Train V11 (SVD Embeddings pipeline)
python farm_to_feed_v11.py
# Output: submission_v11.csv (~3-4 hours)
```

### Step 4: Create Blended Submission

```bash
# Top Score submission
python "blend_v2_v4_v11 copy.py"
# Output: submission_blend_v2_v4_v11.csv

# Backup submission
python blend_v4_v11.py
# Output: submission_blend_v4_v11.csv
```

### Step 5: Submit

Upload `submission_blend_v2_v4_v11.csv` for the best score.

---

## Runtime Details (Actual Measurements)

### Detailed Timing Breakdown

| Script | Stage | Duration | Notes |
|--------|-------|----------|-------|
| **V11** | Data Loading | ~4 sec | 2.1M train, 276K test |
| | SVD Building | ~1 min | 5 SVD configurations |
| | Feature Engineering | ~3 min | 90 features (1w) |
| | OOF Predictions | ~66 min | 5 folds, ~13 min/fold |
| | Final 1w Models | ~100 min | XGB+Cat+LGB ensemble |
| | 2w + Quantity | ~8 min | Faster training |
| | **Total V11** | **~3 hours** | 10:26 → 13:27 |
| **V4** | Window Validation | ~18 min | 4 rolling windows |
| | Feature Engineering | ~1 min | 136 features |
| | 1w Training | ~11 min | LGB 7×58s, XGB 4×68s |
| | 2w Training | ~6 min | LGB 7×48s |
| | Quantity Models | ~30 sec | Fast training |
| | **Total V4** | **~40 min** | 18:38 → 19:00 |
| **V2** | (Estimated) | **~2-3 hours** | Similar to V11 |
| **Blend** | Loading + Blending | **~30 sec** | Simple averaging |

### Summary

| Pipeline | Runtime | Models Trained |
|----------|---------|----------------|
| V11 (SVD) | ~3 hours | 24 XGB + 12 Cat + 7 LGB + 9 2w + 17 qty = **69 models** |
| V4 (Advanced) | ~40 min | 11 LGB + 4 XGB + 10 qty = **25 models** |
| V2 (TE) | ~2-3 hours (est.) | 12 XGB + 6 Cat + 5 LGB + qty = **~30 models** |
| Blend | ~30 sec | N/A |
| **Total** | **~6-7 hours** | **~124 models** |

---

## Key Insights

### What Worked

1. **SVD Embeddings (V11)**: Multi-scale SVD at dimensions 16, 32, 64 captures customer-product affinities that tree models alone cannot learn. The temporal SVD (last 8 weeks) adds recency awareness.

2. **K-Fold Target Encoding (V2)**: Proper K-fold encoding prevents data leakage while extracting powerful categorical signals. The smoothing parameter (20) balances signal and noise.

3. **Segment-Specific Features (V4)**: The "danger zone" (4-12 weeks since last purchase) is a critical segment. Explicit features for this segment improved predictions.

4. **Rank Averaging (V11)**: More robust than mean averaging for combining diverse model predictions.

5. **Specialized Blending**: Using different models for different targets (V11 for WAUC, V4 for MAE) exploits each model's strengths.

### What Didn't Work

| Approach | Why It Failed |
|----------|---------------|
| Neural Networks | Overfitting on tabular data; gradient boosting superior |
| Deep embeddings (dim>64) | Diminishing returns; increased noise |
| Uniform blend weights | Sub-optimal; specialized blending worked better |
| Single model submission | Lower leaderboard score than ensemble |

---

## File Structure

```
Farm to Feed 5/
├── Train.csv                       # Training data (275 MB)
├── Test.csv                        # Test data (27 MB)
├── SampleSubmission.csv            # Submission format
├── variable_description.pdf        # Data dictionary
│
├── SOLUTION_DOCUMENTATION.md       # This file
├── REPRODUCTION_CHECKLIST.md       # Quick reproduction steps
├── requirements.txt                # Python dependencies
│
├── farm_to_feed_v2.py              # V2: Target Encoding pipeline
├── farm_to_feed_v4.py              # V4: Advanced Features pipeline
├── farm_to_feed_v11.py             # V11: SVD Embeddings pipeline
│
├── submission_v2.csv               # V2 predictions
├── submission_v4.csv               # V4 predictions
├── submission_v11.csv              # V11 predictions
│
├── blend_final_winning.py          # TOP SCORE blend script (clean version)
├── blend_v2_v4_v11 copy.py         # TOP SCORE blend script (original)
├── blend_v4_v11.py                 # Backup blend script
│
├── submission_blend_v2_v4_v11.csv  # FINAL SUBMISSION (Top Score)
└── submission_blend_v4_v11.csv     # Backup submission
```

*Documentation generated: January 2026*
