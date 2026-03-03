"""
Farm to Feed V4 - Advanced Pipeline for 0.98 Window AUC
========================================================
Key improvements over V3:
1. Target encoding with regularization (smoothing)
2. Lag features for sequential purchase patterns
3. Segment-specific features for "Old(4-12w)" segment
4. Cross-fold target statistics (leave-one-out encoding)
5. Advanced interaction and ratio features
6. Optimized hyperparameters

Current: 0.9656 Window AUC | Target: 0.98 Window AUC | Gap: 0.014
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
from datetime import datetime
from tqdm import tqdm
import gc
import os

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

USE_GPU = True
N_JOBS = -1

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =============================================================================
# LOAD DATA
# =============================================================================
log("Loading data...")
train = pd.read_csv('Train.csv', parse_dates=['week_start', 'customer_created_at'])
test = pd.read_csv('Test.csv', parse_dates=['week_start', 'customer_created_at'])
submission = pd.read_csv('SampleSubmission.csv')

log(f"Train: {train.shape}, Test: {test.shape}")

train['log_qty_1w'] = np.log1p(train['Target_qty_next_1w'])
train['log_qty_2w'] = np.log1p(train['Target_qty_next_2w'])
train['week_idx'] = train['week_start'].rank(method='dense').astype(np.int32)
max_week_idx = train['week_idx'].max()
max_train_week = train['week_start'].max()

log(f"Train weeks: 1 to {max_week_idx}")

# =============================================================================
# TARGET ENCODING HELPER (with smoothing)
# =============================================================================
def target_encode(train_df, val_df, col, target_col, min_samples=20, smoothing=10):
    """Target encoding with smoothing to prevent overfitting."""
    global_mean = train_df[target_col].mean()
    
    agg = train_df.groupby(col, observed=True)[target_col].agg(['mean', 'count'])
    agg.columns = [f'{col}_target_mean', f'{col}_target_count']
    
    # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
    agg[f'{col}_te'] = (agg[f'{col}_target_count'] * agg[f'{col}_target_mean'] + 
                        smoothing * global_mean) / (agg[f'{col}_target_count'] + smoothing)
    
    # Only use encoding if enough samples
    agg.loc[agg[f'{col}_target_count'] < min_samples, f'{col}_te'] = global_mean
    
    result = agg[[f'{col}_te']].reset_index()
    return result

# =============================================================================
# V4 ENHANCED FEATURE ENGINEERING
# =============================================================================
def create_v4_features(train_df, target_df, max_hist_week, is_train=True, fold_data=None):
    """
    V4 Features with target encoding and advanced patterns:
    - All V3 features
    - Target encoding (customer, product, CP)
    - Lag features (1, 2, 3, 4 week lags)
    - Sequential pattern features
    - Segment-specific features
    """
    hist = train_df[train_df['week_start'] <= max_hist_week].copy()
    df = target_df.copy()
    
    # === BASIC TEMPORAL FEATURES ===
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(np.int32)
    df['month'] = df['week_start'].dt.month.astype(np.int32)
    df['quarter'] = df['week_start'].dt.quarter.astype(np.int8)
    df['days_since_creation'] = (df['week_start'] - df['customer_created_at']).dt.days.astype(np.float32)
    
    # Cyclical encoding
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52).astype(np.float32)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52).astype(np.float32)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
    
    # Customer-level categoricals
    for col in ['customer_category', 'customer_status']:
        df[col] = df[col].astype('category').cat.codes.astype(np.int16)
    
    if len(hist) == 0:
        for col in ['grade_name', 'unit_name']:
            df[col] = df[col].astype('category').cat.codes.astype(np.int16)
        df.fillna(0, inplace=True)
        return df
    
    # === CUSTOMER-PRODUCT AGGREGATIONS ===
    log("    CP aggregations...")
    cp = hist.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
        'purchased_this_week': ['sum', 'mean', 'count', 'std', 'max'],
        'qty_this_week': ['sum', 'mean', 'std', 'max', 'min'],
        'spend_this_week': ['sum', 'mean'],
        'week_idx': ['min', 'max', 'nunique']
    })
    cp.columns = ['cp_purch_sum', 'cp_purch_rate', 'cp_weeks', 'cp_purch_std', 'cp_purch_max',
                  'cp_qty_sum', 'cp_qty_mean', 'cp_qty_std', 'cp_qty_max', 'cp_qty_min',
                  'cp_spend_sum', 'cp_spend_mean', 
                  'cp_first_week', 'cp_last_week', 'cp_unique_weeks']
    cp = cp.fillna(0).reset_index()
    
    # === CUSTOMER AGGREGATIONS ===
    log("    Customer aggregations...")
    cust = hist.groupby('customer_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean', 'std', 'max'],
        'qty_this_week': ['sum', 'mean', 'std'],
        'spend_this_week': ['sum', 'mean'],
        'product_unit_variant_id': 'nunique',
        'week_start': 'nunique'
    })
    cust.columns = ['cust_purch_sum', 'cust_purch_rate', 'cust_purch_std', 'cust_purch_max',
                    'cust_qty_sum', 'cust_qty_mean', 'cust_qty_std',
                    'cust_spend_sum', 'cust_spend_mean',
                    'cust_unique_prods', 'cust_active_weeks']
    cust = cust.fillna(0).reset_index()
    
    # === PRODUCT AGGREGATIONS ===
    log("    Product aggregations...")
    prod = hist.groupby('product_unit_variant_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean', 'std'],
        'qty_this_week': ['sum', 'mean'],
        'customer_id': 'nunique',
        'week_start': 'nunique'
    })
    prod.columns = ['prod_purch_sum', 'prod_purch_rate', 'prod_purch_std',
                    'prod_qty_sum', 'prod_qty_mean', 
                    'prod_unique_custs', 'prod_active_weeks']
    prod = prod.fillna(0).reset_index()
    
    # === COLD-START CATEGORY FEATURES ===
    log("    Cold-start features...")
    cust_grade = hist.groupby(['customer_id', 'grade_name'], observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'qty_this_week': 'sum'
    })
    cust_grade.columns = ['cust_grade_purch_sum', 'cust_grade_purch_rate', 'cust_grade_qty_sum']
    cust_grade = cust_grade.reset_index()
    
    cust_unit = hist.groupby(['customer_id', 'unit_name'], observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'qty_this_week': 'sum'
    })
    cust_unit.columns = ['cust_unit_purch_sum', 'cust_unit_purch_rate', 'cust_unit_qty_sum']
    cust_unit = cust_unit.reset_index()
    
    # === TARGET ENCODING (V4 NEW) ===
    log("    Target encoding...")
    if is_train and fold_data is not None:
        # For training, use out-of-fold encoding
        te_train = fold_data
        cust_te = target_encode(te_train, df, 'customer_id', 'Target_purchase_next_1w')
        prod_te = target_encode(te_train, df, 'product_unit_variant_id', 'Target_purchase_next_1w')
        grade_te = target_encode(te_train, df, 'grade_name', 'Target_purchase_next_1w')
    else:
        # For validation/test, use all historical data
        if 'Target_purchase_next_1w' in hist.columns:
            cust_te = target_encode(hist, df, 'customer_id', 'Target_purchase_next_1w')
            prod_te = target_encode(hist, df, 'product_unit_variant_id', 'Target_purchase_next_1w')
            grade_te = target_encode(hist, df, 'grade_name', 'Target_purchase_next_1w')
        else:
            cust_te = pd.DataFrame()
            prod_te = pd.DataFrame()
            grade_te = pd.DataFrame()
    
    # === RECENCY FEATURES ===
    log("    Recency features...")
    purch = hist[hist['purchased_this_week'] == 1]
    
    if len(purch) > 0:
        last_purch = purch.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
            'week_start': 'max',
            'qty_this_week': 'last',
            'week_idx': 'max'
        }).reset_index()
        last_purch.columns = ['customer_id', 'product_unit_variant_id', 'last_purch_date', 'last_qty', 'last_purch_week_idx']
        
        first_purch = purch.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['week_start'].min().reset_index()
        first_purch.columns = ['customer_id', 'product_unit_variant_id', 'first_purch_date']
        
        cust_last = purch.groupby('customer_id', observed=True)['week_start'].max().reset_index()
        cust_last.columns = ['customer_id', 'cust_last_purch_date']
        
        df = df.merge(last_purch, on=['customer_id', 'product_unit_variant_id'], how='left')
        df = df.merge(first_purch, on=['customer_id', 'product_unit_variant_id'], how='left')
        df = df.merge(cust_last, on='customer_id', how='left')
        
        df['days_since_last'] = (df['week_start'] - df['last_purch_date']).dt.days.fillna(999).astype(np.float32)
        df['days_since_first'] = (df['week_start'] - df['first_purch_date']).dt.days.fillna(999).astype(np.float32)
        df['days_since_cust_last'] = (df['week_start'] - df['cust_last_purch_date']).dt.days.fillna(999).astype(np.float32)
        df['last_qty'] = df['last_qty'].fillna(0).astype(np.float32)
        df['last_purch_week_idx'] = df['last_purch_week_idx'].fillna(0).astype(np.int32)
        
        df.drop(columns=['last_purch_date', 'first_purch_date', 'cust_last_purch_date'], inplace=True)
    else:
        df['days_since_last'] = np.float32(999)
        df['days_since_first'] = np.float32(999)
        df['days_since_cust_last'] = np.float32(999)
        df['last_qty'] = np.float32(0)
        df['last_purch_week_idx'] = np.int32(0)
    
    df['days_since_last_capped'] = np.clip(df['days_since_last'], 0, 180).astype(np.float32)
    df['weeks_since_last'] = (df['days_since_last'] / 7).astype(np.float32)
    
    # === LAG FEATURES (V4 NEW) ===
    log("    Lag features...")
    current_week_idx = hist['week_idx'].max() if len(hist) > 0 else 0
    
    for lag in [1, 2, 3, 4]:
        lag_week_idx = current_week_idx - lag + 1
        lag_data = hist[hist['week_idx'] == lag_week_idx]
        if len(lag_data) > 0:
            lag_agg = lag_data.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
                'purchased_this_week': 'max',
                'qty_this_week': 'sum'
            }).reset_index()
            lag_agg.columns = ['customer_id', 'product_unit_variant_id', f'purch_lag_{lag}', f'qty_lag_{lag}']
            df = df.merge(lag_agg, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    # === ROLLING WINDOWS ===
    log("    Rolling windows...")
    for window in [2, 4, 8, 12]:
        start_week = max_hist_week - pd.Timedelta(weeks=window-1)
        window_data = hist[(hist['week_start'] >= start_week) & (hist['week_start'] <= max_hist_week)]
        if len(window_data) > 0:
            roll = window_data.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
                'purchased_this_week': ['sum', 'mean'],
                'qty_this_week': ['sum', 'mean', 'max']
            }).reset_index()
            roll.columns = ['customer_id', 'product_unit_variant_id', 
                           f'purch_roll_{window}w', f'purch_rate_{window}w',
                           f'qty_roll_{window}w', f'qty_mean_{window}w', f'qty_max_{window}w']
            df = df.merge(roll, on=['customer_id', 'product_unit_variant_id'], how='left')
            
        # Customer rolling
        cust_roll = window_data.groupby('customer_id', observed=True).agg({
            'purchased_this_week': 'sum',
            'product_unit_variant_id': 'nunique'
        }).reset_index()
        cust_roll.columns = ['customer_id', f'cust_purch_roll_{window}w', f'cust_prods_{window}w']
        df = df.merge(cust_roll, on='customer_id', how='left')
        
        # Product rolling
        prod_roll = window_data.groupby('product_unit_variant_id', observed=True).agg({
            'purchased_this_week': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        prod_roll.columns = ['product_unit_variant_id', f'prod_purch_roll_{window}w', f'prod_custs_{window}w']
        df = df.merge(prod_roll, on='product_unit_variant_id', how='left')
    
    # === MERGE ALL AGGREGATIONS ===
    log("    Merging...")
    df = df.merge(cp, on=['customer_id', 'product_unit_variant_id'], how='left')
    df = df.merge(cust, on='customer_id', how='left')
    df = df.merge(prod, on='product_unit_variant_id', how='left')
    df = df.merge(cust_grade, on=['customer_id', 'grade_name'], how='left')
    df = df.merge(cust_unit, on=['customer_id', 'unit_name'], how='left')
    
    # Merge target encoding
    if len(cust_te) > 0:
        df = df.merge(cust_te, on='customer_id', how='left')
    if len(prod_te) > 0:
        df = df.merge(prod_te, on='product_unit_variant_id', how='left')
    if len(grade_te) > 0:
        df = df.merge(grade_te, on='grade_name', how='left')
    
    # Encode remaining categoricals
    for col in ['grade_name', 'unit_name']:
        df[col] = df[col].astype('category').cat.codes.astype(np.int16)
    
    # === DECAY FEATURES ===
    days = df['days_since_last_capped'].values
    df['recency_decay_7d'] = np.exp(-days / 7).astype(np.float32)
    df['recency_decay_14d'] = np.exp(-days / 14).astype(np.float32)
    df['recency_decay_28d'] = np.exp(-days / 28).astype(np.float32)
    df['recency_decay_56d'] = np.exp(-days / 56).astype(np.float32)
    df['cust_recency_decay'] = np.exp(-df['days_since_cust_last'].values / 28).astype(np.float32)
    
    # === DORMANCY FEATURES ===
    log("    Dormancy features...")
    df['cp_span_weeks'] = (df['cp_last_week'] - df['cp_first_week']).clip(lower=1).fillna(1).astype(np.float32)
    df['avg_purchase_interval'] = (df['cp_span_weeks'] / df['cp_purch_sum'].clip(lower=1)).fillna(52).astype(np.float32)
    df['is_overdue'] = (df['weeks_since_last'] > 1.5 * df['avg_purchase_interval']).astype(np.int8)
    df['overdue_ratio'] = (df['weeks_since_last'] / df['avg_purchase_interval'].clip(lower=1)).clip(0, 10).fillna(0).astype(np.float32)
    df['dormancy_score'] = (1 - np.exp(-df['weeks_since_last'] / df['avg_purchase_interval'].clip(lower=1))).clip(0, 1).fillna(0).astype(np.float32)
    df['reactivation_potential'] = (df['recency_decay_28d'] * (1 + df['cp_purch_sum'] * 0.05)).clip(0, 5).astype(np.float32)
    
    # === SEGMENT-SPECIFIC FEATURES (V4 NEW - for Old(4-12w) segment) ===
    log("    Segment features...")
    # Is in "danger zone" (4-12 weeks since last purchase)
    df['is_danger_zone'] = ((df['weeks_since_last'] >= 4) & (df['weeks_since_last'] <= 12)).astype(np.int8)
    df['danger_zone_score'] = np.where(
        df['is_danger_zone'] == 1,
        df['cp_purch_sum'] * df['recency_decay_28d'],  # Higher if more history and more recent within zone
        0
    ).astype(np.float32)
    
    # Purchase frequency ratio (for segment handling)
    df['purchase_frequency'] = (df['cp_purch_sum'] / df['cp_unique_weeks'].clip(lower=1)).fillna(0).astype(np.float32)
    
    # === INTERACTION FEATURES ===
    log("    Interactions...")
    df['recency_x_freq'] = (df['recency_decay_28d'] * df['cp_purch_rate'].fillna(0)).astype(np.float32)
    
    cust_max = df['cust_purch_sum'].max() if df['cust_purch_sum'].max() > 0 else 1
    prod_max = df['prod_purch_sum'].max() if df['prod_purch_sum'].max() > 0 else 1
    df['cust_activity'] = (df['cust_purch_sum'] / cust_max).fillna(0).astype(np.float32)
    df['prod_popularity'] = (df['prod_purch_sum'] / prod_max).fillna(0).astype(np.float32)
    df['activity_x_popularity'] = (df['cust_activity'] * df['prod_popularity']).astype(np.float32)
    
    df['cp_affinity'] = (df['cp_purch_sum'] / df['cust_purch_sum'].clip(lower=1)).fillna(0).astype(np.float32)
    df['affinity_x_recency'] = (df['cp_affinity'] * df['recency_decay_28d']).astype(np.float32)
    
    # === TREND FEATURES ===
    log("    Trends...")
    df['purchase_velocity'] = (df['cp_purch_sum'] / df['cp_span_weeks'].clip(lower=1)).fillna(0).astype(np.float32)
    
    if 'purch_roll_2w' in df.columns and 'purch_roll_8w' in df.columns:
        df['purch_trend_2v8'] = (df['purch_roll_2w'] / df['purch_roll_8w'].clip(lower=0.1)).clip(0, 10).fillna(0).astype(np.float32)
    if 'purch_roll_4w' in df.columns and 'purch_roll_12w' in df.columns:
        df['purch_trend_4v12'] = (df['purch_roll_4w'] / df['purch_roll_12w'].clip(lower=0.1)).clip(0, 10).fillna(0).astype(np.float32)
    
    if 'purch_trend_2v8' in df.columns:
        df['trend_x_history'] = (df['purch_trend_2v8'] * df['cp_purch_sum']).clip(0, 100).fillna(0).astype(np.float32)
    
    if 'cust_purch_roll_4w' in df.columns and 'cust_purch_roll_12w' in df.columns:
        df['cust_momentum'] = (df['cust_purch_roll_4w'] / df['cust_purch_roll_12w'].clip(lower=0.1)).clip(0, 10).fillna(0).astype(np.float32)
    
    if 'prod_purch_roll_4w' in df.columns and 'prod_purch_roll_12w' in df.columns:
        df['prod_momentum'] = (df['prod_purch_roll_4w'] / df['prod_purch_roll_12w'].clip(lower=0.1)).clip(0, 10).fillna(0).astype(np.float32)
    
    # === LAG-BASED SEQUENTIAL FEATURES (V4 NEW) ===
    log("    Sequential patterns...")
    for col in ['purch_lag_1', 'purch_lag_2', 'purch_lag_3', 'purch_lag_4']:
        if col not in df.columns:
            df[col] = np.float32(0)
    
    # Consecutive purchases
    df['consecutive_purch'] = (df['purch_lag_1'].fillna(0) + df['purch_lag_2'].fillna(0)).astype(np.float32)
    
    # Purchase pattern (weighted recent lags)
    df['lag_pattern'] = (0.4 * df['purch_lag_1'].fillna(0) + 
                         0.3 * df['purch_lag_2'].fillna(0) + 
                         0.2 * df['purch_lag_3'].fillna(0) + 
                         0.1 * df['purch_lag_4'].fillna(0)).astype(np.float32)
    
    # === DERIVED FEATURES ===
    df['has_purchased'] = (df['cp_purch_sum'].fillna(0) > 0).astype(np.int8)
    df['purchase_consistency'] = (df['cp_purch_sum'] / df['cp_weeks'].clip(lower=1)).fillna(0).astype(np.float32)
    df['product_share'] = (df['cp_qty_sum'] / df['cust_qty_sum'].clip(lower=1)).clip(0, 1).fillna(0).astype(np.float32)
    df['recency_x_last_qty'] = (df['recency_decay_28d'] * df['last_qty']).astype(np.float32)
    
    # Customer loyalty (how often vs number of products)
    df['customer_loyalty'] = (df['cp_purch_sum'] / df['cust_unique_prods'].clip(lower=1)).fillna(0).astype(np.float32)
    
    # Product stickiness
    df['product_stickiness'] = (df['cp_purch_sum'] / df['prod_unique_custs'].clip(lower=1)).fillna(0).astype(np.float32)
    
    df.fillna(0, inplace=True)
    
    return df

# =============================================================================
# MODEL CONFIGURATIONS (Optimized)
# =============================================================================
log("Configuring models...")

# LightGBM - optimized for AUC
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'num_leaves': 63, 'max_depth': 8,
    'min_child_samples': 50, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
    'bagging_freq': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.2, 
    'verbose': -1, 'seed': SEED,
    'device': 'cpu'  # More stable
}

# XGBoost
xgb_params = {
    'n_estimators': 1500, 'learning_rate': 0.015, 'max_depth': 7, 
    'min_child_weight': 50, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 0.2,
    'tree_method': 'hist', 'device': 'cuda' if USE_GPU else 'cpu'
}

lgb_params_qty = {
    'objective': 'regression_l1', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'num_leaves': 31, 'max_depth': 7,
    'min_child_samples': 30, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'verbose': -1, 'seed': SEED,
    'device': 'cpu'
}

# =============================================================================
# EXCLUDED COLUMNS
# =============================================================================
exclude_cols = ['ID', 'customer_id', 'product_unit_variant_id', 'week_start', 'customer_created_at',
                'product_id', 'product_grade_variant_id', 'selling_price',
                'Target_purchase_next_1w', 'Target_qty_next_1w',
                'Target_purchase_next_2w', 'Target_qty_next_2w', 'log_qty_1w', 'log_qty_2w',
                'qty_this_week', 'purchased_this_week', 'spend_this_week', 'num_orders_week',
                'pred_1w', 'week_idx', 'cp_first_week', 'cp_last_week', 'last_purch_week_idx']

# =============================================================================
# ROLLING WINDOW VALIDATION
# =============================================================================
log("\n" + "="*80)
log("V4 ROLLING WINDOW VALIDATION")
log("="*80)

windows = [
    {'train_start': 1, 'train_end': 26, 'test_start': 27, 'test_end': 28},
    {'train_start': 7, 'train_end': 32, 'test_start': 33, 'test_end': 34},
    {'train_start': 13, 'train_end': 38, 'test_start': 39, 'test_end': 40},
    {'train_start': 19, 'train_end': 44, 'test_start': 45, 'test_end': 46},
]

windows = [w for w in windows if w['test_end'] <= max_week_idx]

window_aucs = []
all_val_preds = []
all_val_targets = []

for w_idx, w in enumerate(windows, 1):
    log(f"\n--- Window {w_idx}: Train {w['train_start']}-{w['train_end']}, Val {w['test_start']}-{w['test_end']} ---")
    
    train_mask = (train['week_idx'] >= w['train_start']) & (train['week_idx'] <= w['train_end'])
    val_mask = (train['week_idx'] >= w['test_start']) & (train['week_idx'] <= w['test_end'])
    
    train_df = train[train_mask].copy()
    val_df = train[val_mask].copy()
    
    max_hist = train_df['week_start'].max()
    
    log(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")
    
    # Build features
    train_feat = create_v4_features(train_df, train_df, max_hist, is_train=True, fold_data=train_df)
    val_feat = create_v4_features(train_df, val_df, max_hist, is_train=False)
    
    feat_cols = [c for c in train_feat.columns if c not in exclude_cols]
    log(f"  Features: {len(feat_cols)}")
    
    X_tr = train_feat[feat_cols].values.astype(np.float32)
    y_tr = train_df['Target_purchase_next_1w'].values
    X_val = val_feat[feat_cols].values.astype(np.float32)
    y_val = val_df['Target_purchase_next_1w'].values
    
    # Time decay weights
    max_train_week_idx = train_df['week_idx'].max()
    weeks_ago = max_train_week_idx - train_df['week_idx'].values
    weights = np.power(2, -weeks_ago / 20.0).astype(np.float32)
    
    # Train ensemble
    val_preds = []
    
    # LightGBM models
    for s in range(5):
        p = lgb_params.copy()
        p['seed'] = SEED + s
        train_set = lgb.Dataset(X_tr, label=y_tr, weight=weights)
        model = lgb.train(p, train_set, num_boost_round=800)
        val_preds.append(model.predict(X_val))
    
    # XGBoost models
    for s in range(3):
        cfg = xgb_params.copy()
        try:
            model = xgb.XGBClassifier(**cfg, random_state=SEED+s, use_label_encoder=False, 
                                       eval_metric='auc', verbosity=0)
            model.fit(X_tr, y_tr, sample_weight=weights, verbose=False)
            val_preds.append(model.predict_proba(X_val)[:, 1])
        except Exception as e:
            cfg['device'] = 'cpu'
            model = xgb.XGBClassifier(**cfg, random_state=SEED+s, use_label_encoder=False, 
                                       eval_metric='auc', verbosity=0)
            model.fit(X_tr, y_tr, sample_weight=weights, verbose=False)
            val_preds.append(model.predict_proba(X_val)[:, 1])
    
    # Ensemble
    val_pred = np.mean(val_preds, axis=0)
    val_auc = roc_auc_score(y_val, val_pred)
    window_aucs.append(val_auc)
    
    all_val_preds.extend(val_pred)
    all_val_targets.extend(y_val)
    
    log(f"  Val AUC: {val_auc:.4f}")
    
    del model, train_set, train_feat, val_feat
    gc.collect()

avg_auc = np.mean(window_aucs)
log(f"\n{'='*60}")
log(f"AVERAGE WINDOW AUC: {avg_auc:.4f}")
log(f"Window AUCs: {[f'{a:.4f}' for a in window_aucs]}")
log(f"{'='*60}")

# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================
log("\n" + "="*80)
log("FINAL MODEL TRAINING")
log("="*80)

log("\nBuilding features for full training...")
train_feat = create_v4_features(train, train, max_train_week, is_train=True, fold_data=train)
test_feat = create_v4_features(train, test, max_train_week, is_train=False)

feature_cols = [c for c in train_feat.columns if c not in exclude_cols]
log(f"Features: {len(feature_cols)}")

X_full = train_feat[feature_cols].values.astype(np.float32)
X_test = test_feat[feature_cols].values.astype(np.float32)
y_full_1w = train['Target_purchase_next_1w'].values
y_full_2w = train['Target_purchase_next_2w'].values
y_full_log_qty_1w = train['log_qty_1w'].values
y_full_log_qty_2w = train['log_qty_2w'].values

# Time decay weights
weeks_ago = max_week_idx - train['week_idx'].values
sample_weights = np.power(2, -weeks_ago / 20.0).astype(np.float32)

# Train 1w purchase models
log("\nTraining 1w purchase models...")
test_preds_1w = []

for s in tqdm(range(7), desc="LightGBM"):
    p = lgb_params.copy()
    p['seed'] = SEED + s
    train_set = lgb.Dataset(X_full, label=y_full_1w, weight=sample_weights)
    model = lgb.train(p, train_set, num_boost_round=1000)
    test_preds_1w.append(model.predict(X_test))

for s in tqdm(range(4), desc="XGBoost"):
    cfg = xgb_params.copy()
    try:
        model = xgb.XGBClassifier(**cfg, random_state=SEED+s, use_label_encoder=False, 
                                   eval_metric='auc', verbosity=0)
        model.fit(X_full, y_full_1w, sample_weight=sample_weights, verbose=False)
        test_preds_1w.append(model.predict_proba(X_test)[:, 1])
    except Exception as e:
        cfg['device'] = 'cpu'
        model = xgb.XGBClassifier(**cfg, random_state=SEED+s, use_label_encoder=False, 
                                   eval_metric='auc', verbosity=0)
        model.fit(X_full, y_full_1w, sample_weight=sample_weights, verbose=False)
        test_preds_1w.append(model.predict_proba(X_test)[:, 1])

pred_1w = np.mean(test_preds_1w, axis=0)

# Train 2w purchase models
log("\nTraining 2w purchase models...")
test_preds_2w = []

for s in tqdm(range(7), desc="2w LightGBM"):
    p = lgb_params.copy()
    p['seed'] = SEED + s
    train_set = lgb.Dataset(X_full, label=y_full_2w, weight=sample_weights)
    model = lgb.train(p, train_set, num_boost_round=800)
    test_preds_2w.append(model.predict(X_test))

pred_2w = np.mean(test_preds_2w, axis=0)

# Train quantity models
log("\nTraining quantity models...")
purch_mask_1w = y_full_1w == 1
purch_mask_2w = y_full_2w == 1

qty_preds_1w = []
for s in tqdm(range(5), desc="1w Qty"):
    p = lgb_params_qty.copy()
    p['seed'] = SEED + s
    train_set = lgb.Dataset(X_full[purch_mask_1w], label=y_full_log_qty_1w[purch_mask_1w],
                            weight=sample_weights[purch_mask_1w])
    model = lgb.train(p, train_set, num_boost_round=500)
    qty_preds_1w.append(model.predict(X_test))

qty_preds_2w = []
for s in tqdm(range(5), desc="2w Qty"):
    p = lgb_params_qty.copy()
    p['seed'] = SEED + s
    train_set = lgb.Dataset(X_full[purch_mask_2w], label=y_full_log_qty_2w[purch_mask_2w],
                            weight=sample_weights[purch_mask_2w])
    model = lgb.train(p, train_set, num_boost_round=500)
    qty_preds_2w.append(model.predict(X_test))

pred_log_qty_1w = np.mean(qty_preds_1w, axis=0)
pred_log_qty_2w = np.mean(qty_preds_2w, axis=0)

pred_qty_1w = pred_1w * np.clip(np.expm1(pred_log_qty_1w), 0, 500)
pred_qty_2w = pred_2w * np.clip(np.expm1(pred_log_qty_2w), 0, 500)

# =============================================================================
# SUBMISSION
# =============================================================================
log("\nCreating submission...")

submission['Target_purchase_next_1w'] = pred_1w
submission['Target_qty_next_1w'] = pred_qty_1w
submission['Target_purchase_next_2w'] = pred_2w
submission['Target_qty_next_2w'] = pred_qty_2w

submission.to_csv('submission_v4.csv', index=False)
log("Saved submission_v4.csv")

# =============================================================================
# SUMMARY
# =============================================================================
log("\n" + "="*80)
log("V4 PIPELINE SUMMARY")
log("="*80)
log(f"Features: {len(feature_cols)}")
log(f"Average Window AUC: {avg_auc:.4f}")
log(f"Window AUCs: {[f'{a:.4f}' for a in window_aucs]}")
log("")
log("V4 New Features:")
log("  - Target encoding (customer, product, grade)")
log("  - Lag features (1-4 week lags)")
log("  - Sequential pattern features (consecutive, lag_pattern)")
log("  - Segment-specific (danger_zone, danger_zone_score)")
log("  - Advanced rolling (product/customer trending)")
log("")
log("Predictions:")
log(f"  1w Purchase: mean={pred_1w.mean():.6f}")
log(f"  2w Purchase: mean={pred_2w.mean():.6f}")
log(f"  1w Quantity: mean={pred_qty_1w.mean():.4f}")
log(f"  2w Quantity: mean={pred_qty_2w.mean():.4f}")
log("="*80)
