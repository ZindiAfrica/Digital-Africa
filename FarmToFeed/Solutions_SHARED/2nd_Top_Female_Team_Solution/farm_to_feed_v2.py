"""
Farm to Feed v2 - Enhanced with Target Encoding

Based on validation results from feature_engineering_validation.ipynb:
- Baseline (Label Only): Val AUC 0.9978, Overfit Gap 0.0003
- With Target Encoding: Val AUC 0.9977, Overfit Gap 0.0004  
- With Combinations: Val AUC 0.9977, Overfit Gap 0.0004

Key additions in v2:
1. K-Fold Target Encoding with smoothing for categorical features
2. Count Encoding for frequency information
3. Categorical Combination features (customer_category × grade_name, etc.)
4. Proper train/test encoding separation to prevent leakage 

The validation showed minimal overfitting (gap < 0.04%), confirming the Kaggle winner's
approach is safe when implemented with proper K-fold encoding.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import warnings
from datetime import datetime
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_TIME_DECAY_1W = True
USE_TIME_DECAY_2W_PURCH = False
USE_TIME_DECAY_2W_QTY = True
DECAY_HALFLIFE_WEEKS = 15.0
N_JOBS = -1

# v2 Feature Engineering Config
USE_TARGET_ENCODING = True
USE_COUNT_ENCODING = True
USE_COMBINATIONS = True
TE_SMOOTHING = 20
TE_N_FOLDS = 5

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

# Time-decay weights
weeks_ago = max_week_idx - train['week_idx'].values
sample_weights = np.power(2, -weeks_ago / DECAY_HALFLIFE_WEEKS).astype(np.float32)

# =============================================================================
# TARGET ENCODING FUNCTIONS (from Kaggle winner strategy)
# =============================================================================
def target_encode_kfold(train_df, test_df, cols, target, n_folds=5, smooth=20, agg='mean'):
    """
    K-Fold Target Encoding with smoothing to prevent leakage.
    
    This is the key technique from the Kaggle winning solution.
    Uses nested K-fold to encode training data to prevent data leakage.
    """
    if isinstance(cols, str):
        cols = [cols]
    
    col_name = '_'.join(str(c) for c in cols)
    feature_name = f'TE_{agg.upper()}_{col_name}'
    
    # Global stats for fallback
    if agg == 'mean':
        global_agg = train_df[target].mean()
    elif agg == 'median':
        global_agg = train_df[target].median()
    elif agg == 'std':
        global_agg = train_df[target].std()
    else:
        global_agg = train_df[target].sum()
    
    # Initialize output
    train_encoded = pd.Series(index=train_df.index, dtype='float32')
    
    # K-fold encoding for train
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for enc_idx, val_idx in kf.split(train_df):
        enc_df = train_df.iloc[enc_idx]
        val_df = train_df.iloc[val_idx]
        
        if agg == 'count':
            agg_stats = enc_df.groupby(cols).size().reset_index(name='count')
            agg_stats['encoded'] = agg_stats['count']
        else:
            agg_stats = enc_df.groupby(cols)[target].agg([agg, 'count']).reset_index()
            agg_stats.columns = cols + ['agg_val', 'count']
            # Smoothed encoding
            agg_stats['encoded'] = (
                (agg_stats['agg_val'] * agg_stats['count'] + global_agg * smooth) / 
                (agg_stats['count'] + smooth)
            )
        
        # Create mapping
        if len(cols) > 1:
            keys = agg_stats[cols].apply(tuple, axis=1)
        else:
            keys = agg_stats[cols[0]]
        mapping = dict(zip(keys, agg_stats['encoded']))
        
        # Apply to validation fold
        if len(cols) > 1:
            val_keys = val_df[cols].apply(tuple, axis=1)
        else:
            val_keys = val_df[cols[0]]
        train_encoded.iloc[val_idx] = val_keys.map(mapping).fillna(global_agg)
    
    # Full encoding for test
    if agg == 'count':
        agg_stats = train_df.groupby(cols).size().reset_index(name='count')
        agg_stats['encoded'] = agg_stats['count']
    else:
        agg_stats = train_df.groupby(cols)[target].agg([agg, 'count']).reset_index()
        agg_stats.columns = cols + ['agg_val', 'count']
        agg_stats['encoded'] = (
            (agg_stats['agg_val'] * agg_stats['count'] + global_agg * smooth) / 
            (agg_stats['count'] + smooth)
        )
    
    if len(cols) > 1:
        keys = agg_stats[cols].apply(tuple, axis=1)
        test_keys = test_df[cols].apply(tuple, axis=1)
    else:
        keys = agg_stats[cols[0]]
        test_keys = test_df[cols[0]]
    
    mapping = dict(zip(keys, agg_stats['encoded']))
    test_encoded = test_keys.map(mapping).fillna(global_agg).astype('float32')
    
    return train_encoded.astype('float32'), test_encoded, feature_name


def count_encode(train_df, test_df, cols):
    """Count encoding (frequency) for categorical columns."""
    if isinstance(cols, str):
        cols = [cols]
    
    col_name = '_'.join(str(c) for c in cols)
    feature_name = f'CE_{col_name}'
    
    # Combined counts for better test generalization
    combined = pd.concat([train_df[cols], test_df[cols]], axis=0, ignore_index=True)
    counts = combined.groupby(cols).size().reset_index(name='count')
    
    if len(cols) > 1:
        keys = counts[cols].apply(tuple, axis=1)
        train_keys = train_df[cols].apply(tuple, axis=1)
        test_keys = test_df[cols].apply(tuple, axis=1)
    else:
        keys = counts[cols[0]]
        train_keys = train_df[cols[0]]
        test_keys = test_df[cols[0]]
    
    mapping = dict(zip(keys, counts['count']))
    train_encoded = train_keys.map(mapping).fillna(0).astype('int32')
    test_encoded = test_keys.map(mapping).fillna(0).astype('int32')
    
    return train_encoded, test_encoded, feature_name


def add_target_encoding_features(train_df, test_df, target_col, cat_cols=None):
    """Add target encoding and count encoding for categorical columns."""
    if cat_cols is None:
        cat_cols = ['grade_name', 'unit_name', 'customer_category', 'customer_status']
    
    features_added = []
    
    for col in cat_cols:
        # TE Mean
        train_enc, test_enc, fname = target_encode_kfold(
            train_df, test_df, col, target_col, n_folds=TE_N_FOLDS, smooth=TE_SMOOTHING, agg='mean'
        )
        train_df[fname] = train_enc
        test_df[fname] = test_enc
        features_added.append(fname)
        
        # Count Encoding
        train_enc, test_enc, fname = count_encode(train_df, test_df, col)
        train_df[fname] = train_enc
        test_df[fname] = test_enc
        features_added.append(fname)
    
    return train_df, test_df, features_added


def add_combination_features(train_df, test_df, target_col):
    """Add categorical combination features (interaction effects)."""
    combinations = [
        ['customer_category', 'grade_name'],
        ['customer_category', 'unit_name'],
        ['customer_status', 'grade_name'],
        ['grade_name', 'unit_name'],
        ['customer_category', 'customer_status', 'grade_name'],
    ]
    
    features_added = []
    
    for combo in combinations:
        # TE Mean
        train_enc, test_enc, fname = target_encode_kfold(
            train_df, test_df, combo, target_col, n_folds=TE_N_FOLDS, smooth=TE_SMOOTHING, agg='mean'
        )
        train_df[fname] = train_enc
        test_df[fname] = test_enc
        features_added.append(fname)
        
        # Count Encoding
        train_enc, test_enc, fname = count_encode(train_df, test_df, combo)
        train_df[fname] = train_enc
        test_df[fname] = test_enc
        features_added.append(fname)
    
    return train_df, test_df, features_added


# =============================================================================
# v2 FEATURES (v1 baseline + Target Encoding)
# =============================================================================
def create_features_1w_v2(train_df, target_df, max_hist_week):
    """v2's 1w features: v1 baseline + Target Encoding + Combinations."""
    hist = train_df[train_df['week_start'] <= max_hist_week]
    df = target_df.copy()
    
    # Temporal features
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(np.int32)
    df['month'] = df['week_start'].dt.month.astype(np.int32)
    df['days_since_creation'] = (df['week_start'] - df['customer_created_at']).dt.days.astype(np.float32)
    
    # Label encode categoricals (keep for tree models)
    for col in ['grade_name', 'unit_name', 'customer_category', 'customer_status']:
        df[col + '_LE'] = df[col].astype('category').cat.codes.astype(np.int16)
    
    if len(hist) == 0:
        return df
    
    # Customer-Product aggregations (v1 baseline)
    cp = hist.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
        'purchased_this_week': ['sum', 'mean', 'count'],
        'qty_this_week': ['sum', 'mean']
    })
    cp.columns = ['hist_purch_cnt', 'hist_purch_rate', 'hist_weeks', 'hist_qty_sum', 'hist_qty_mean']
    cp = cp.reset_index()
    
    # Customer aggregations
    cust = hist.groupby('customer_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'product_unit_variant_id': 'nunique'
    })
    cust.columns = ['cust_purch_cnt', 'cust_purch_rate', 'cust_unique_prods']
    cust = cust.reset_index()
    
    # Product aggregations
    prod = hist.groupby('product_unit_variant_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'customer_id': 'nunique'
    })
    prod.columns = ['prod_purch_cnt', 'prod_purch_rate', 'prod_unique_custs']
    prod = prod.reset_index()
    
    # Recency features
    purch = hist[hist['purchased_this_week'] == 1]
    if len(purch) > 0:
        last = purch.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['week_start'].max().reset_index()
        last.columns = ['customer_id', 'product_unit_variant_id', 'last_purch_date']
        df = df.merge(last, on=['customer_id', 'product_unit_variant_id'], how='left')
        df['days_since_last'] = (df['week_start'] - df['last_purch_date']).dt.days.fillna(999).astype(np.float32)
        df.drop(columns=['last_purch_date'], inplace=True)
    else:
        df['days_since_last'] = np.float32(999)
    
    # Decay features
    days = df['days_since_last'].values
    df['recency_decay_7d'] = np.exp(-days / 7).astype(np.float32)
    df['recency_decay_14d'] = np.exp(-days / 14).astype(np.float32)
    df['recency_decay_28d'] = np.exp(-days / 28).astype(np.float32)
    
    # Merge aggregations
    df = df.merge(cp, on=['customer_id', 'product_unit_variant_id'], how='left')
    df = df.merge(cust, on='customer_id', how='left')
    df = df.merge(prod, on='product_unit_variant_id', how='left')
    
    # Rolling window
    start_4w = max_hist_week - pd.Timedelta(weeks=3)
    roll4 = hist[(hist['week_start'] >= start_4w) & (hist['week_start'] <= max_hist_week)]
    if len(roll4) > 0:
        r4 = roll4.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['purchased_this_week'].sum().reset_index()
        r4.columns = ['customer_id', 'product_unit_variant_id', 'purch_roll_4w']
        df = df.merge(r4, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    df['has_purchased'] = (df.get('hist_purch_cnt', 0) > 0).astype(np.int8)
    df.fillna(0, inplace=True)
    return df


def create_features_2w_v2(train_df, target_df, max_hist_week):
    """v2's 2w features: v1 baseline + Target Encoding + Combinations."""
    hist_data = train_df[train_df['week_start'] <= max_hist_week]
    df = target_df.copy()
    
    week_of_year = df['week_start'].dt.isocalendar().week.astype(np.int32)
    df['week_of_year'] = week_of_year
    df['month'] = df['week_start'].dt.month.astype(np.int32)
    df['days_since_creation'] = (df['week_start'] - df['customer_created_at']).dt.days.astype(np.float32)
    df['week_sin'] = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32)
    df['week_cos'] = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32)
    
    # Label encode categoricals
    for col in ['grade_name', 'unit_name', 'customer_category', 'customer_status']:
        df[col + '_LE'] = df[col].astype('category').cat.codes.astype(np.int16)
    
    if len(hist_data) == 0:
        return df
    
    # Customer-Product aggregations
    cp_agg = hist_data.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
        'purchased_this_week': ['sum', 'mean', 'count'],
        'qty_this_week': ['sum', 'mean', 'std', 'max'],
        'spend_this_week': ['sum', 'mean'],
        'num_orders_week': ['sum', 'mean']
    })
    cp_agg.columns = ['hist_purchase_count', 'hist_purchase_rate', 'hist_weeks_seen',
                      'hist_qty_sum', 'hist_qty_mean', 'hist_qty_std', 'hist_qty_max',
                      'hist_spend_sum', 'hist_spend_mean', 'hist_orders_sum', 'hist_orders_mean']
    cp_agg['hist_qty_std'] = cp_agg['hist_qty_std'].fillna(0)
    cp_agg = cp_agg.reset_index()
    
    # Customer aggregations
    cust_agg = hist_data.groupby('customer_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'qty_this_week': ['sum', 'mean'],
        'spend_this_week': ['sum', 'mean'],
        'product_unit_variant_id': 'nunique',
        'week_start': 'nunique'
    })
    cust_agg.columns = ['cust_total_purchases', 'cust_purchase_rate', 'cust_total_qty', 'cust_avg_qty',
                        'cust_total_spend', 'cust_avg_spend', 'cust_unique_products', 'cust_active_weeks']
    cust_agg = cust_agg.reset_index()
    
    # Product aggregations
    prod_agg = hist_data.groupby('product_unit_variant_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'qty_this_week': ['sum', 'mean'],
        'customer_id': 'nunique',
        'week_start': 'nunique'
    })
    prod_agg.columns = ['prod_total_purchases', 'prod_purchase_rate', 'prod_total_qty', 'prod_avg_qty',
                        'prod_unique_customers', 'prod_active_weeks']
    prod_agg = prod_agg.reset_index()
    
    # Recency features
    purchased = hist_data[hist_data['purchased_this_week'] == 1]
    if len(purchased) > 0:
        last_cp = purchased.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['week_start'].max().reset_index()
        last_cp.columns = ['customer_id', 'product_unit_variant_id', 'last_purchase_date']
        first_cp = purchased.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['week_start'].min().reset_index()
        first_cp.columns = ['customer_id', 'product_unit_variant_id', 'first_purchase_date']
        cust_last = purchased.groupby('customer_id', observed=True)['week_start'].max().reset_index()
        cust_last.columns = ['customer_id', 'cust_last_purchase_date']
        
        df = df.merge(last_cp, on=['customer_id', 'product_unit_variant_id'], how='left')
        df = df.merge(first_cp, on=['customer_id', 'product_unit_variant_id'], how='left')
        df = df.merge(cust_last, on='customer_id', how='left')
        
        df['days_since_last'] = (df['week_start'] - df['last_purchase_date']).dt.days.fillna(999).astype(np.float32)
        df['days_since_first'] = (df['week_start'] - df['first_purchase_date']).dt.days.fillna(999).astype(np.float32)
        df['days_since_cust_last'] = (df['week_start'] - df['cust_last_purchase_date']).dt.days.fillna(999).astype(np.float32)
        df.drop(columns=['last_purchase_date', 'first_purchase_date', 'cust_last_purchase_date'], inplace=True, errors='ignore')
    else:
        df['days_since_last'] = np.float32(999)
        df['days_since_first'] = np.float32(999)
        df['days_since_cust_last'] = np.float32(999)
    
    # Decay features
    days = df['days_since_last'].values
    df['recency_decay_7d'] = np.exp(-days / 7).astype(np.float32)
    df['recency_decay_14d'] = np.exp(-days / 14).astype(np.float32)
    df['recency_decay_28d'] = np.exp(-days / 28).astype(np.float32)
    df['recency_decay_56d'] = np.exp(-days / 56).astype(np.float32)
    df['cust_recency_decay'] = np.exp(-df['days_since_cust_last'].values / 28).astype(np.float32)
    df['recency_bucket'] = pd.cut(df['days_since_last'], bins=[-np.inf, 7, 14, 21, 28, 42, 56, 84, np.inf],
                                  labels=False).astype(np.int16)
    
    df = df.merge(cp_agg, on=['customer_id', 'product_unit_variant_id'], how='left')
    df = df.merge(cust_agg, on='customer_id', how='left')
    df = df.merge(prod_agg, on='product_unit_variant_id', how='left')
    
    # Rolling windows
    for window in [2, 4, 8]:
        start_week = max_hist_week - pd.Timedelta(weeks=window-1)
        window_data = hist_data[(hist_data['week_start'] >= start_week) & (hist_data['week_start'] <= max_hist_week)]
        if len(window_data) > 0:
            roll_agg = window_data.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
                'purchased_this_week': 'sum', 'qty_this_week': 'sum'
            }).reset_index()
            roll_agg.columns = ['customer_id', 'product_unit_variant_id', f'purch_roll_{window}w', f'qty_roll_{window}w']
            df = df.merge(roll_agg, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    # Lag features
    for lag in [1, 2]:
        lag_week = max_hist_week - pd.Timedelta(weeks=lag-1)
        lag_data = hist_data[hist_data['week_start'] == lag_week][
            ['customer_id', 'product_unit_variant_id', 'purchased_this_week', 'qty_this_week']].copy()
        if len(lag_data) > 0:
            lag_data.columns = ['customer_id', 'product_unit_variant_id', f'purch_lag_{lag}w', f'qty_lag_{lag}w']
            df = df.merge(lag_data, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    df['has_purchased'] = (df.get('hist_purchase_count', 0) > 0).astype(np.int8)
    
    # Derived features
    hist_purch = df.get('hist_purchase_count', pd.Series(0)).fillna(0)
    cust_total = df.get('cust_total_purchases', pd.Series(1)).clip(lower=1)
    cust_qty = df.get('cust_total_qty', pd.Series(1)).clip(lower=1)
    hist_weeks = df.get('hist_weeks_seen', pd.Series(1)).clip(lower=1)
    hist_qty = df.get('hist_qty_sum', pd.Series(0)).fillna(0)
    
    df['cust_prod_affinity'] = (hist_purch / cust_total).fillna(0).astype(np.float32)
    df['product_share'] = (hist_qty / cust_qty).clip(0, 1).fillna(0).astype(np.float32)
    df['purchase_consistency'] = (hist_purch / hist_weeks).fillna(0).astype(np.float32)
    df['recency_x_affinity'] = (df['recency_decay_28d'] * df['cust_prod_affinity']).astype(np.float32)
    df['recency_x_consistency'] = (df['recency_decay_28d'] * df['purchase_consistency']).astype(np.float32)
    
    if 'purch_roll_2w' in df.columns and 'purch_roll_8w' in df.columns:
        df['purch_trend'] = (df['purch_roll_2w'] / df['purch_roll_8w'].clip(lower=0.1)).clip(0, 10).fillna(0).astype(np.float32)
    
    df.fillna(0, inplace=True)
    return df


# =============================================================================
# MODEL TRAINING HELPERS
# =============================================================================
def train_xgb_model(cfg, X, y, weights, seed):
    model = xgb.XGBClassifier(**cfg, random_state=seed, use_label_encoder=False, eval_metric='auc', verbosity=0)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model

def train_cat_model(cfg, X, y, weights, seed):
    model = CatBoostClassifier(**cfg, random_seed=seed, verbose=False)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model

def train_lgb_model(params, X, y, weights, num_rounds):
    train_data = lgb.Dataset(X, label=y, weight=weights)
    model = lgb.train(params, train_data, num_boost_round=num_rounds)
    return model

def train_xgb_reg(cfg, X, y, weights, seed):
    model = xgb.XGBRegressor(**cfg, random_state=seed, verbosity=0)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model

def train_cat_reg(cfg, X, y, weights, seed):
    model = CatBoostRegressor(**cfg, random_seed=seed, verbose=False)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model


# =============================================================================
# MODEL PARAMETERS
# =============================================================================
exclude_cols = ['ID', 'customer_id', 'product_unit_variant_id', 'week_start', 'customer_created_at',
                'product_id', 'product_grade_variant_id', 'selling_price',
                'Target_purchase_next_1w', 'Target_qty_next_1w',
                'Target_purchase_next_2w', 'Target_qty_next_2w', 'log_qty_1w', 'log_qty_2w',
                'qty_this_week', 'purchased_this_week', 'spend_this_week', 'num_orders_week',
                'pred_1w', 'week_idx',
                # Raw categorical columns (we use LE versions + TE versions)
                'grade_name', 'unit_name', 'customer_category', 'customer_status']

# v2 classification configs (same as v1 - proven)
xgb_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 6, 'min_child_weight': 100,
     'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.4},
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 7, 'min_child_weight': 80,
     'subsample': 0.7, 'colsample_bytree': 0.65, 'reg_alpha': 0.25, 'reg_lambda': 0.45},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 8, 'min_child_weight': 50,
     'subsample': 0.65, 'colsample_bytree': 0.6, 'reg_alpha': 0.3, 'reg_lambda': 0.5},
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 5, 'min_child_weight': 120,
     'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_alpha': 0.15, 'reg_lambda': 0.35},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 6, 'min_child_weight': 100,
     'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.3},
    {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 100,
     'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.4},
]

cat_configs = [
    {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    {'iterations': 1500, 'learning_rate': 0.02, 'depth': 7, 'l2_leaf_reg': 2},
]

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 6,
    'min_child_samples': 100, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 0.2, 'reg_lambda': 0.4, 'verbose': -1, 'seed': SEED
}

# Quantity configs
lgb_params_qty = {
    'objective': 'regression_l1', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'num_leaves': 31, 'max_depth': 7,
    'min_child_samples': 30, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'verbose': -1, 'seed': SEED
}

xgb_qty_config = {
    'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 6, 'min_child_weight': 30,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.3,
    'objective': 'reg:absoluteerror'
}

cat_qty_config = {
    'iterations': 1000, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 2,
    'loss_function': 'MAE'
}

ENSEMBLE_WEIGHTS = {'xgb': 0.80, 'cat': 0.12, 'lgb': 0.08}

# =============================================================================
# BUILD CLASSIFICATION FEATURES (v2 with Target Encoding)
# =============================================================================
log("")
log("=" * 80)
log("V2: Enhanced with K-Fold Target Encoding (Kaggle Winner Strategy)")
log("=" * 80)

log("\nBuilding 1w classification features (v2)...")
train_1w_clf = create_features_1w_v2(train, train, max_train_week)
test_1w_clf = create_features_1w_v2(train, test, max_train_week)

# Add Target Encoding features
if USE_TARGET_ENCODING:
    log("  Adding Target Encoding features...")
    train_1w_clf, test_1w_clf, te_features_1w = add_target_encoding_features(
        train_1w_clf, test_1w_clf, 'Target_purchase_next_1w'
    )
    log(f"  Added {len(te_features_1w)} TE features")

# Add Combination features
if USE_COMBINATIONS:
    log("  Adding Combination features...")
    train_1w_clf, test_1w_clf, combo_features_1w = add_combination_features(
        train_1w_clf, test_1w_clf, 'Target_purchase_next_1w'
    )
    log(f"  Added {len(combo_features_1w)} combination features")

feature_cols_1w_clf = [c for c in train_1w_clf.columns if c not in exclude_cols]
log(f"1w classification features: {len(feature_cols_1w_clf)}")

X_full_1w_clf = train_1w_clf[feature_cols_1w_clf].values.astype(np.float32)
X_test_1w_clf = test_1w_clf[feature_cols_1w_clf].values.astype(np.float32)
y_full_1w = train_1w_clf['Target_purchase_next_1w'].values

del train_1w_clf, test_1w_clf
gc.collect()

# =============================================================================
# STEP 1: OOF 1w predictions (v2 features)
# =============================================================================
log("")
log("Step 1: OOF 1w predictions (v2 features with TE)...")

oof_1w_preds = np.zeros(len(X_full_1w_clf), dtype=np.float32)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_1w_clf)):
    X_tr, X_val = X_full_1w_clf[train_idx], X_full_1w_clf[val_idx]
    y_tr, y_val = y_full_1w[train_idx], y_full_1w[val_idx]
    
    fold_preds = []
    fold_weights = []
    
    xgb_models = Parallel(n_jobs=N_JOBS)(
        delayed(train_xgb_model)(cfg, X_tr, y_tr, None, SEED) for cfg in xgb_configs
    )
    for m in xgb_models:
        fold_preds.append(m.predict_proba(X_val)[:, 1])
        fold_weights.append(ENSEMBLE_WEIGHTS['xgb'] / len(xgb_configs))
    
    cat_models = Parallel(n_jobs=N_JOBS)(
        delayed(train_cat_model)(cfg, X_tr, y_tr, None, SEED) for cfg in cat_configs
    )
    for m in cat_models:
        fold_preds.append(m.predict_proba(X_val)[:, 1])
        fold_weights.append(ENSEMBLE_WEIGHTS['cat'] / len(cat_configs))
    
    lgb_model = train_lgb_model(lgb_params, X_tr, y_tr, None, 500)
    fold_preds.append(lgb_model.predict(X_val))
    fold_weights.append(ENSEMBLE_WEIGHTS['lgb'])
    
    oof_1w_preds[val_idx] = np.average(fold_preds, axis=0, weights=fold_weights)
    log(f"  Fold {fold+1}: AUC = {roc_auc_score(y_val, oof_1w_preds[val_idx]):.6f}")
    gc.collect()

log(f"  Overall OOF AUC: {roc_auc_score(y_full_1w, oof_1w_preds):.6f}")

# =============================================================================
# STEP 2: Final 1w classification models
# =============================================================================
log("")
log("Step 2: Final 1w classification models...")

weights_1w = sample_weights if USE_TIME_DECAY_1W else None

xgb_tasks = [(cfg, SEED + s) for cfg in xgb_configs for s in range(2)]
xgb_models = Parallel(n_jobs=N_JOBS)(
    delayed(train_xgb_model)(cfg, X_full_1w_clf, y_full_1w, weights_1w, seed) 
    for cfg, seed in tqdm(xgb_tasks, desc="  XGB")
)
xgb_preds = [m.predict_proba(X_test_1w_clf)[:, 1] for m in xgb_models]

cat_tasks = [(cfg, SEED + s) for cfg in cat_configs for s in range(3)]
cat_models = Parallel(n_jobs=N_JOBS)(
    delayed(train_cat_model)(cfg, X_full_1w_clf, y_full_1w, weights_1w, seed) 
    for cfg, seed in tqdm(cat_tasks, desc="  CatBoost")
)
cat_preds = [m.predict_proba(X_test_1w_clf)[:, 1] for m in cat_models]

lgb_preds = []
for s in tqdm(range(5), desc="  LightGBM"):
    p = lgb_params.copy()
    p['seed'] = SEED + s
    m = train_lgb_model(p, X_full_1w_clf, y_full_1w, weights_1w, 500)
    lgb_preds.append(m.predict(X_test_1w_clf))

all_1w_preds = xgb_preds + cat_preds + lgb_preds
all_1w_weights = ([ENSEMBLE_WEIGHTS['xgb'] / len(xgb_preds)] * len(xgb_preds) +
                  [ENSEMBLE_WEIGHTS['cat'] / len(cat_preds)] * len(cat_preds) +
                  [ENSEMBLE_WEIGHTS['lgb'] / len(lgb_preds)] * len(lgb_preds))

pred_1w_test = np.average(all_1w_preds, axis=0, weights=all_1w_weights)

del xgb_models, cat_models
gc.collect()

# =============================================================================
# STEP 3: 2w classification features (v2)
# =============================================================================
log("")
log("Step 3: Building 2w classification features (v2 with TE)...")

train_2w_clf = create_features_2w_v2(train, train, max_train_week)
test_2w_clf = create_features_2w_v2(train, test, max_train_week)

# Add Target Encoding for 2w target
if USE_TARGET_ENCODING:
    log("  Adding Target Encoding features for 2w...")
    train_2w_clf, test_2w_clf, te_features_2w = add_target_encoding_features(
        train_2w_clf, test_2w_clf, 'Target_purchase_next_2w'
    )
    log(f"  Added {len(te_features_2w)} TE features")

if USE_COMBINATIONS:
    log("  Adding Combination features for 2w...")
    train_2w_clf, test_2w_clf, combo_features_2w = add_combination_features(
        train_2w_clf, test_2w_clf, 'Target_purchase_next_2w'
    )
    log(f"  Added {len(combo_features_2w)} combination features")

train_2w_clf['pred_1w'] = oof_1w_preds
test_2w_clf['pred_1w'] = pred_1w_test

exclude_cols_2w = [c for c in exclude_cols if c != 'pred_1w']
feature_cols_2w_clf = [c for c in train_2w_clf.columns if c not in exclude_cols_2w]
log(f"2w classification features: {len(feature_cols_2w_clf)}")

X_full_2w_clf = train_2w_clf[feature_cols_2w_clf].values.astype(np.float32)
X_test_2w_clf = test_2w_clf[feature_cols_2w_clf].values.astype(np.float32)
y_full_2w = train_2w_clf['Target_purchase_next_2w'].values

del train_2w_clf, test_2w_clf
gc.collect()

# =============================================================================
# STEP 4: 2w PURCHASE model
# =============================================================================
log("")
log("Step 4: 2w purchase model (v2 with TE)...")

lgb_models_2w = Parallel(n_jobs=N_JOBS)(
    delayed(train_lgb_model)({**lgb_params, 'seed': SEED + s}, X_full_2w_clf, y_full_2w, None, 500)
    for s in tqdm(range(7), desc="  2w LightGBM")
)
pred_2w = np.mean([m.predict(X_test_2w_clf) for m in lgb_models_2w], axis=0)

del lgb_models_2w
gc.collect()

# =============================================================================
# STEP 5: Quantity models (using v1 enhanced features - proven for MAE)
# =============================================================================
log("")
log("Step 5: Quantity models (v1 enhanced features)...")

# Build quantity features (v1 style - proven for MAE)
log("  Building 1w quantity features...")
train_1w_qty = create_features_1w_v2(train, train, max_train_week)
test_1w_qty = create_features_1w_v2(train, test, max_train_week)

feature_cols_1w_qty = [c for c in train_1w_qty.columns if c not in exclude_cols]

X_full_1w_qty = train_1w_qty[feature_cols_1w_qty].values.astype(np.float32)
X_test_1w_qty = test_1w_qty[feature_cols_1w_qty].values.astype(np.float32)
y_full_log_qty_1w = train['log_qty_1w'].values

purch_mask_1w = y_full_1w == 1

del train_1w_qty, test_1w_qty
gc.collect()

# 1w quantity
weights_qty_1w = sample_weights[purch_mask_1w] if USE_TIME_DECAY_1W else None

log("  1w Quantity ensemble...")
lgb_qty_1w = Parallel(n_jobs=N_JOBS)(
    delayed(train_lgb_model)({**lgb_params_qty, 'seed': SEED + s}, 
                              X_full_1w_qty[purch_mask_1w], y_full_log_qty_1w[purch_mask_1w], 
                              weights_qty_1w, 400)
    for s in range(5)
)
lgb_qty_1w_preds = np.mean([m.predict(X_test_1w_qty) for m in lgb_qty_1w], axis=0)

xgb_qty_1w = Parallel(n_jobs=N_JOBS)(
    delayed(train_xgb_reg)(xgb_qty_config, 
                           X_full_1w_qty[purch_mask_1w], y_full_log_qty_1w[purch_mask_1w],
                           weights_qty_1w, SEED + s)
    for s in range(3)
)
xgb_qty_1w_preds = np.mean([m.predict(X_test_1w_qty) for m in xgb_qty_1w], axis=0)

cat_qty_1w = Parallel(n_jobs=N_JOBS)(
    delayed(train_cat_reg)(cat_qty_config,
                           X_full_1w_qty[purch_mask_1w], y_full_log_qty_1w[purch_mask_1w],
                           weights_qty_1w, SEED + s)
    for s in range(3)
)
cat_qty_1w_preds = np.mean([m.predict(X_test_1w_qty) for m in cat_qty_1w], axis=0)

pred_log_qty_1w = 0.5 * lgb_qty_1w_preds + 0.3 * xgb_qty_1w_preds + 0.2 * cat_qty_1w_preds

# 2w quantity
log("  Building 2w quantity features...")
train_2w_qty = create_features_2w_v2(train, train, max_train_week)
test_2w_qty = create_features_2w_v2(train, test, max_train_week)

feature_cols_2w_qty = [c for c in train_2w_qty.columns if c not in exclude_cols]

X_full_2w_qty = train_2w_qty[feature_cols_2w_qty].values.astype(np.float32)
X_test_2w_qty = test_2w_qty[feature_cols_2w_qty].values.astype(np.float32)
y_full_log_qty_2w = train['log_qty_2w'].values

purch_mask_2w = y_full_2w == 1

del train_2w_qty, test_2w_qty
gc.collect()

weights_qty_2w = sample_weights[purch_mask_2w] if USE_TIME_DECAY_2W_QTY else None

log("  2w Quantity ensemble...")
lgb_qty_2w = Parallel(n_jobs=N_JOBS)(
    delayed(train_lgb_model)({**lgb_params_qty, 'seed': SEED + s}, 
                              X_full_2w_qty[purch_mask_2w], y_full_log_qty_2w[purch_mask_2w], 
                              weights_qty_2w, 400)
    for s in range(5)
)
lgb_qty_2w_preds = np.mean([m.predict(X_test_2w_qty) for m in lgb_qty_2w], axis=0)

xgb_qty_2w = Parallel(n_jobs=N_JOBS)(
    delayed(train_xgb_reg)(xgb_qty_config,
                           X_full_2w_qty[purch_mask_2w], y_full_log_qty_2w[purch_mask_2w],
                           weights_qty_2w, SEED + s)
    for s in range(3)
)
xgb_qty_2w_preds = np.mean([m.predict(X_test_2w_qty) for m in xgb_qty_2w], axis=0)

cat_qty_2w = Parallel(n_jobs=N_JOBS)(
    delayed(train_cat_reg)(cat_qty_config,
                           X_full_2w_qty[purch_mask_2w], y_full_log_qty_2w[purch_mask_2w],
                           weights_qty_2w, SEED + s)
    for s in range(3)
)
cat_qty_2w_preds = np.mean([m.predict(X_test_2w_qty) for m in cat_qty_2w], axis=0)

pred_log_qty_2w = 0.5 * lgb_qty_2w_preds + 0.3 * xgb_qty_2w_preds + 0.2 * cat_qty_2w_preds

# Final quantity
pred_qty_1w = pred_1w_test * np.clip(np.expm1(pred_log_qty_1w), 0, 500)
pred_qty_2w = pred_2w * np.clip(np.expm1(pred_log_qty_2w), 0, 500)

# =============================================================================
# SUBMISSION
# =============================================================================
log("")
log("Creating submission...")

submission['Target_purchase_next_1w'] = pred_1w_test
submission['Target_qty_next_1w'] = pred_qty_1w
submission['Target_purchase_next_2w'] = pred_2w
submission['Target_qty_next_2w'] = pred_qty_2w

submission.to_csv('submission_v2.csv', index=False)
log("Saved submission_v2.csv")

# =============================================================================
# SUMMARY
# =============================================================================
log("")
log("=" * 80)
log("V2 SUMMARY - Enhanced with K-Fold Target Encoding")
log("=" * 80)
log("")
log("New in v2 (from Kaggle Winner Strategy):")
log(f"   - Target Encoding: {USE_TARGET_ENCODING}")
log(f"   - Count Encoding: {USE_COUNT_ENCODING}")
log(f"   - Categorical Combinations: {USE_COMBINATIONS}")
log(f"   - TE Smoothing: {TE_SMOOTHING}")
log(f"   - TE K-Folds: {TE_N_FOLDS}")
log("")
log("Classification features:")
log(f"   - 1w clf features: {len(feature_cols_1w_clf)}")
log(f"   - 2w clf features: {len(feature_cols_2w_clf)}")
log("")
log("Predictions:")
log(f"   - 1w Purchase: mean={pred_1w_test.mean():.6f}")
log(f"   - 2w Purchase: mean={pred_2w.mean():.6f}")
log(f"   - 1w Quantity: mean={pred_qty_1w.mean():.4f}")
log(f"   - 2w Quantity: mean={pred_qty_2w.mean():.4f}")
log("")
log("Expected (based on validation):")
log("   - Val AUC: ~0.9977 (no significant degradation)")
log("   - Overfit Gap: < 0.04% (well controlled)")
log("   - TE Features may help on unseen test distributions")
log("=" * 80)
