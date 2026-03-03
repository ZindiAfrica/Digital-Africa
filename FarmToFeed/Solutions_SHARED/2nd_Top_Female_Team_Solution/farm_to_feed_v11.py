"""
Farm to Feed v11 - Enhanced SVD Features & Improved Modelling

V10 Baseline: V2 + basic SVD features

V11 Improvements:
1. TEMPORAL SVD: Recent 8-week SVD for recency-aware patterns
2. WEIGHTED SVD: Quantity-weighted interactions (not just counts)
3. MULTI-SCALE SVD: dim=16 + dim=32 + dim=64 combined
4. ADVANCED INTERACTIONS: SVD × recency × behavioral crosses
5. MORE MODEL DIVERSITY: Additional configs, more seeds
6. DYNAMIC ENSEMBLE: OOF-based weight optimization
7. RANK AVERAGING: More robust final predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.stats import rankdata
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

TE_SMOOTHING = 20
TE_N_FOLDS = 5

# V11 Multi-scale SVD config
SVD_DIMS = [16, 32, 64]  # Multiple embedding dimensions
RECENT_WEEKS = 8  # For temporal SVD

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

weeks_ago = max_week_idx - train['week_idx'].values
sample_weights = np.power(2, -weeks_ago / DECAY_HALFLIFE_WEEKS).astype(np.float32)

# =============================================================================
# V11: MULTI-SCALE & TEMPORAL SVD EMBEDDINGS
# =============================================================================
log("")
log("=" * 80)
log("V11: MULTI-SCALE & TEMPORAL SVD EMBEDDINGS")
log("=" * 80)

def build_svd_embeddings_v11(train_df, emb_dim, use_recent_only=False, use_qty_weights=False):
    """V11: Enhanced SVD with quantity weighting and temporal options."""
    
    if use_recent_only:
        recent_cutoff = train_df['week_start'].max() - pd.Timedelta(weeks=RECENT_WEEKS)
        purchases = train_df[(train_df['purchased_this_week'] == 1) & 
                            (train_df['week_start'] >= recent_cutoff)].copy()
        suffix = f"_recent{RECENT_WEEKS}w"
    else:
        purchases = train_df[train_df['purchased_this_week'] == 1].copy()
        suffix = ""
    
    if len(purchases) < 100:
        return None, None, 0, suffix
    
    all_customers = train_df['customer_id'].unique()
    all_products = train_df['product_unit_variant_id'].unique()
    
    cust_to_idx = {c: i for i, c in enumerate(all_customers)}
    prod_to_idx = {p: i for i, p in enumerate(all_products)}
    
    # V11: Use quantity-weighted aggregation
    if use_qty_weights:
        agg = purchases.groupby(['customer_id', 'product_unit_variant_id']).agg({
            'qty_this_week': 'sum'
        }).reset_index()
        agg.columns = ['customer_id', 'product_unit_variant_id', 'weight']
        agg['weight'] = np.log1p(agg['weight'])
    else:
        agg = purchases.groupby(['customer_id', 'product_unit_variant_id']).size().reset_index(name='weight')
        agg['weight'] = np.log1p(agg['weight'])
    
    rows = agg['customer_id'].map(cust_to_idx).values
    cols = agg['product_unit_variant_id'].map(prod_to_idx).values
    data = agg['weight'].values
    
    n_customers = len(all_customers)
    n_products = len(all_products)
    
    interaction_matrix = csr_matrix((data, (rows, cols)), shape=(n_customers, n_products))
    
    k = min(emb_dim, min(n_customers, n_products) - 1)
    
    try:
        U, S, Vt = svds(interaction_matrix.astype(np.float64), k=k, random_state=SEED)
    except:
        return None, None, 0, suffix
    
    sqrt_S = np.sqrt(S)
    customer_emb_matrix = U * sqrt_S
    product_emb_matrix = (Vt.T * sqrt_S)
    
    customer_embeddings = {c: customer_emb_matrix[i] for c, i in cust_to_idx.items()}
    product_embeddings = {p: product_emb_matrix[i] for p, i in prod_to_idx.items()}
    
    return customer_embeddings, product_embeddings, k, suffix


def add_svd_features_v11(df, svd_embeddings_list):
    """V11: Add features from multiple SVD embeddings."""
    
    n_samples = len(df)
    
    for cust_emb, prod_emb, emb_dim, suffix in svd_embeddings_list:
        if cust_emb is None:
            continue
            
        cust_emb_array = np.zeros((n_samples, emb_dim), dtype=np.float32)
        prod_emb_array = np.zeros((n_samples, emb_dim), dtype=np.float32)
        
        for i, (cust_id, prod_id) in enumerate(zip(df['customer_id'], df['product_unit_variant_id'])):
            if cust_id in cust_emb:
                cust_emb_array[i] = cust_emb[cust_id]
            if prod_id in prod_emb:
                prod_emb_array[i] = prod_emb[prod_id]
        
        # Normalize
        cust_emb_norm = normalize(cust_emb_array, axis=1)
        prod_emb_norm = normalize(prod_emb_array, axis=1)
        
        dim_suffix = f"_d{emb_dim}{suffix}"
        
        # Core features
        df[f'svd_sim{dim_suffix}'] = np.sum(cust_emb_norm * prod_emb_norm, axis=1).astype(np.float32)
        df[f'svd_dot{dim_suffix}'] = np.sum(cust_emb_array * prod_emb_array, axis=1).astype(np.float32)
        df[f'svd_dist{dim_suffix}'] = np.sqrt(np.sum((cust_emb_array - prod_emb_array) ** 2, axis=1)).astype(np.float32)
        df[f'cust_norm{dim_suffix}'] = np.linalg.norm(cust_emb_array, axis=1).astype(np.float32)
        df[f'prod_norm{dim_suffix}'] = np.linalg.norm(prod_emb_array, axis=1).astype(np.float32)
        
        # Top component interactions
        interaction = cust_emb_array * prod_emb_array
        for i in range(min(3, emb_dim)):
            df[f'svd_int{i}{dim_suffix}'] = interaction[:, i].astype(np.float32)
    
    return df


def add_svd_cross_features(df):
    """V11: Cross-features between SVD and behavioral features."""
    
    # Find SVD similarity columns
    sim_cols = [c for c in df.columns if c.startswith('svd_sim')]
    
    if len(sim_cols) == 0:
        return df
    
    # Use the primary SVD similarity (d32)
    main_sim = 'svd_sim_d32' if 'svd_sim_d32' in df.columns else sim_cols[0]
    
    # Cross with recency features
    if 'recency_decay_7d' in df.columns:
        df['svd_x_rec7'] = (df[main_sim] * df['recency_decay_7d']).astype(np.float32)
    if 'recency_decay_28d' in df.columns:
        df['svd_x_rec28'] = (df[main_sim] * df['recency_decay_28d']).astype(np.float32)
    
    # Cross with purchase history
    if 'hist_purch_rate' in df.columns:
        df['svd_x_prate'] = (df[main_sim] * df['hist_purch_rate']).astype(np.float32)
    if 'hist_purch_cnt' in df.columns:
        df['svd_x_pcnt'] = (df[main_sim] * np.log1p(df['hist_purch_cnt'])).astype(np.float32)
    
    # Multi-scale SVD agreement
    if 'svd_sim_d16' in df.columns and 'svd_sim_d64' in df.columns:
        df['svd_agreement'] = ((df['svd_sim_d16'] + df['svd_sim_d32'] + df['svd_sim_d64']) / 3).astype(np.float32)
        df['svd_std'] = np.std([df['svd_sim_d16'], df['svd_sim_d32'], df['svd_sim_d64']], axis=0).astype(np.float32)
    
    # Temporal vs global SVD comparison
    if 'svd_sim_d32' in df.columns and 'svd_sim_d32_recent8w' in df.columns:
        df['svd_temporal_diff'] = (df['svd_sim_d32_recent8w'] - df['svd_sim_d32']).astype(np.float32)
        df['svd_temporal_ratio'] = (df['svd_sim_d32_recent8w'] / df['svd_sim_d32'].clip(0.01)).clip(-5, 5).astype(np.float32)
    
    return df


# Build all SVD embeddings
log("Building multi-scale SVD embeddings...")
svd_embeddings = []

# Standard SVD at multiple scales
for dim in SVD_DIMS:
    log(f"  Building standard SVD (dim={dim})...")
    c_emb, p_emb, actual_dim, suffix = build_svd_embeddings_v11(train, dim, use_recent_only=False, use_qty_weights=False)
    if c_emb is not None:
        svd_embeddings.append((c_emb, p_emb, actual_dim, suffix))
        log(f"    Done: {actual_dim} dimensions")

# Quantity-weighted SVD (main dimension only)
log("  Building quantity-weighted SVD (dim=32)...")
c_emb, p_emb, actual_dim, _ = build_svd_embeddings_v11(train, 32, use_recent_only=False, use_qty_weights=True)
if c_emb is not None:
    svd_embeddings.append((c_emb, p_emb, actual_dim, "_qw"))
    log(f"    Done: {actual_dim} dimensions")

# Temporal SVD (recent weeks only)
log(f"  Building temporal SVD (last {RECENT_WEEKS} weeks, dim=32)...")
c_emb, p_emb, actual_dim, suffix = build_svd_embeddings_v11(train, 32, use_recent_only=True, use_qty_weights=False)
if c_emb is not None:
    svd_embeddings.append((c_emb, p_emb, actual_dim, suffix))
    log(f"    Done: {actual_dim} dimensions")

log(f"Total SVD configurations: {len(svd_embeddings)}")

# =============================================================================
# TARGET ENCODING FUNCTIONS (V2 exact)
# =============================================================================
def target_encode_kfold(train_df, test_df, cols, target, n_folds=5, smooth=20):
    if isinstance(cols, str):
        cols = [cols]
    
    col_name = '_'.join(str(c) for c in cols)
    feature_name = f'TE_{col_name}'
    global_mean = train_df[target].mean()
    
    train_encoded = pd.Series(index=train_df.index, dtype='float32')
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    for enc_idx, val_idx in kf.split(train_df):
        enc_df = train_df.iloc[enc_idx]
        agg_stats = enc_df.groupby(cols)[target].agg(['mean', 'count']).reset_index()
        agg_stats.columns = cols + ['agg_val', 'count']
        agg_stats['encoded'] = (agg_stats['agg_val'] * agg_stats['count'] + global_mean * smooth) / (agg_stats['count'] + smooth)
        
        if len(cols) > 1:
            keys = agg_stats[cols].apply(tuple, axis=1)
            val_keys = train_df.iloc[val_idx][cols].apply(tuple, axis=1)
        else:
            keys = agg_stats[cols[0]]
            val_keys = train_df.iloc[val_idx][cols[0]]
        train_encoded.iloc[val_idx] = val_keys.map(dict(zip(keys, agg_stats['encoded']))).fillna(global_mean)
    
    agg_stats = train_df.groupby(cols)[target].agg(['mean', 'count']).reset_index()
    agg_stats.columns = cols + ['agg_val', 'count']
    agg_stats['encoded'] = (agg_stats['agg_val'] * agg_stats['count'] + global_mean * smooth) / (agg_stats['count'] + smooth)
    
    if len(cols) > 1:
        keys = agg_stats[cols].apply(tuple, axis=1)
        test_keys = test_df[cols].apply(tuple, axis=1)
    else:
        keys = agg_stats[cols[0]]
        test_keys = test_df[cols[0]]
    test_encoded = test_keys.map(dict(zip(keys, agg_stats['encoded']))).fillna(global_mean).astype('float32')
    
    return train_encoded.astype('float32'), test_encoded, feature_name


def count_encode(train_df, test_df, cols):
    if isinstance(cols, str):
        cols = [cols]
    col_name = '_'.join(str(c) for c in cols)
    feature_name = f'CE_{col_name}'
    
    combined = pd.concat([train_df[cols], test_df[cols]], ignore_index=True)
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
    return train_keys.map(mapping).fillna(0).astype('int32'), test_keys.map(mapping).fillna(0).astype('int32'), feature_name


def add_te_v2(train_df, test_df, target_col):
    cat_cols = ['grade_name', 'unit_name', 'customer_category', 'customer_status']
    features = []
    
    for col in cat_cols:
        tr, te, fn = target_encode_kfold(train_df, test_df, col, target_col)
        train_df[fn] = tr
        test_df[fn] = te
        features.append(fn)
        
        tr, te, fn = count_encode(train_df, test_df, col)
        train_df[fn] = tr
        test_df[fn] = te
        features.append(fn)
    
    combos = [
        ['customer_category', 'grade_name'],
        ['customer_category', 'unit_name'],
        ['customer_status', 'grade_name'],
        ['grade_name', 'unit_name'],
        ['customer_category', 'customer_status', 'grade_name'],
    ]
    
    for combo in combos:
        tr, te, fn = target_encode_kfold(train_df, test_df, combo, target_col)
        train_df[fn] = tr
        test_df[fn] = te
        features.append(fn)
        
        tr, te, fn = count_encode(train_df, test_df, combo)
        train_df[fn] = tr
        test_df[fn] = te
        features.append(fn)
    
    return train_df, test_df, features


# =============================================================================
# V11 FEATURE FUNCTIONS
# =============================================================================
def create_features_1w_v11(train_df, target_df, max_hist_week, svd_embs):
    """V11: V2 features + multi-scale SVD."""
    hist = train_df[train_df['week_start'] <= max_hist_week]
    df = target_df.copy()
    
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(np.int32)
    df['month'] = df['week_start'].dt.month.astype(np.int32)
    df['days_since_creation'] = (df['week_start'] - df['customer_created_at']).dt.days.astype(np.float32)
    
    for col in ['grade_name', 'unit_name', 'customer_category', 'customer_status']:
        df[col + '_LE'] = df[col].astype('category').cat.codes.astype(np.int16)
    
    # ADD MULTI-SCALE SVD FEATURES
    df = add_svd_features_v11(df, svd_embs)
    
    if len(hist) == 0:
        df = add_svd_cross_features(df)
        return df
    
    cp = hist.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
        'purchased_this_week': ['sum', 'mean', 'count'],
        'qty_this_week': ['sum', 'mean']
    })
    cp.columns = ['hist_purch_cnt', 'hist_purch_rate', 'hist_weeks', 'hist_qty_sum', 'hist_qty_mean']
    cp = cp.reset_index()
    
    cust = hist.groupby('customer_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'product_unit_variant_id': 'nunique'
    })
    cust.columns = ['cust_purch_cnt', 'cust_purch_rate', 'cust_unique_prods']
    cust = cust.reset_index()
    
    prod = hist.groupby('product_unit_variant_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'customer_id': 'nunique'
    })
    prod.columns = ['prod_purch_cnt', 'prod_purch_rate', 'prod_unique_custs']
    prod = prod.reset_index()
    
    purch = hist[hist['purchased_this_week'] == 1]
    if len(purch) > 0:
        last = purch.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['week_start'].max().reset_index()
        last.columns = ['customer_id', 'product_unit_variant_id', 'last_purch_date']
        df = df.merge(last, on=['customer_id', 'product_unit_variant_id'], how='left')
        df['days_since_last'] = (df['week_start'] - df['last_purch_date']).dt.days.fillna(999).astype(np.float32)
        df.drop(columns=['last_purch_date'], inplace=True)
    else:
        df['days_since_last'] = np.float32(999)
    
    days = df['days_since_last'].values
    df['recency_decay_7d'] = np.exp(-days / 7).astype(np.float32)
    df['recency_decay_14d'] = np.exp(-days / 14).astype(np.float32)
    df['recency_decay_28d'] = np.exp(-days / 28).astype(np.float32)
    
    df = df.merge(cp, on=['customer_id', 'product_unit_variant_id'], how='left')
    df = df.merge(cust, on='customer_id', how='left')
    df = df.merge(prod, on='product_unit_variant_id', how='left')
    
    start_4w = max_hist_week - pd.Timedelta(weeks=3)
    roll4 = hist[(hist['week_start'] >= start_4w) & (hist['week_start'] <= max_hist_week)]
    if len(roll4) > 0:
        r4 = roll4.groupby(['customer_id', 'product_unit_variant_id'], observed=True)['purchased_this_week'].sum().reset_index()
        r4.columns = ['customer_id', 'product_unit_variant_id', 'purch_roll_4w']
        df = df.merge(r4, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    df['has_purchased'] = (df.get('hist_purch_cnt', 0) > 0).astype(np.int8)
    
    # V11: Add cross features
    df = add_svd_cross_features(df)
    
    df.fillna(0, inplace=True)
    return df


def create_features_2w_v11(train_df, target_df, max_hist_week, svd_embs):
    """V11: V2's 2w features + multi-scale SVD."""
    hist_data = train_df[train_df['week_start'] <= max_hist_week]
    df = target_df.copy()
    
    week_of_year = df['week_start'].dt.isocalendar().week.astype(np.int32)
    df['week_of_year'] = week_of_year
    df['month'] = df['week_start'].dt.month.astype(np.int32)
    df['days_since_creation'] = (df['week_start'] - df['customer_created_at']).dt.days.astype(np.float32)
    df['week_sin'] = np.sin(2 * np.pi * week_of_year / 52).astype(np.float32)
    df['week_cos'] = np.cos(2 * np.pi * week_of_year / 52).astype(np.float32)
    
    for col in ['grade_name', 'unit_name', 'customer_category', 'customer_status']:
        df[col + '_LE'] = df[col].astype('category').cat.codes.astype(np.int16)
    
    # ADD MULTI-SCALE SVD FEATURES
    df = add_svd_features_v11(df, svd_embs)
    
    if len(hist_data) == 0:
        df = add_svd_cross_features(df)
        return df
    
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
    
    prod_agg = hist_data.groupby('product_unit_variant_id', observed=True).agg({
        'purchased_this_week': ['sum', 'mean'],
        'qty_this_week': ['sum', 'mean'],
        'customer_id': 'nunique',
        'week_start': 'nunique'
    })
    prod_agg.columns = ['prod_total_purchases', 'prod_purchase_rate', 'prod_total_qty', 'prod_avg_qty',
                        'prod_unique_customers', 'prod_active_weeks']
    prod_agg = prod_agg.reset_index()
    
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
    
    days = df['days_since_last'].values
    df['recency_decay_7d'] = np.exp(-days / 7).astype(np.float32)
    df['recency_decay_14d'] = np.exp(-days / 14).astype(np.float32)
    df['recency_decay_28d'] = np.exp(-days / 28).astype(np.float32)
    df['recency_decay_56d'] = np.exp(-days / 56).astype(np.float32)
    df['cust_recency_decay'] = np.exp(-df['days_since_cust_last'].values / 28).astype(np.float32)
    df['recency_bucket'] = pd.cut(df['days_since_last'], bins=[-np.inf, 7, 14, 21, 28, 42, 56, 84, np.inf], labels=False).astype(np.int16)
    
    df = df.merge(cp_agg, on=['customer_id', 'product_unit_variant_id'], how='left')
    df = df.merge(cust_agg, on='customer_id', how='left')
    df = df.merge(prod_agg, on='product_unit_variant_id', how='left')
    
    for window in [2, 4, 8]:
        start_week = max_hist_week - pd.Timedelta(weeks=window-1)
        window_data = hist_data[(hist_data['week_start'] >= start_week) & (hist_data['week_start'] <= max_hist_week)]
        if len(window_data) > 0:
            roll_agg = window_data.groupby(['customer_id', 'product_unit_variant_id'], observed=True).agg({
                'purchased_this_week': 'sum', 'qty_this_week': 'sum'
            }).reset_index()
            roll_agg.columns = ['customer_id', 'product_unit_variant_id', f'purch_roll_{window}w', f'qty_roll_{window}w']
            df = df.merge(roll_agg, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    for lag in [1, 2]:
        lag_week = max_hist_week - pd.Timedelta(weeks=lag-1)
        lag_data = hist_data[hist_data['week_start'] == lag_week][
            ['customer_id', 'product_unit_variant_id', 'purchased_this_week', 'qty_this_week']].copy()
        if len(lag_data) > 0:
            lag_data.columns = ['customer_id', 'product_unit_variant_id', f'purch_lag_{lag}w', f'qty_lag_{lag}w']
            df = df.merge(lag_data, on=['customer_id', 'product_unit_variant_id'], how='left')
    
    df['has_purchased'] = (df.get('hist_purchase_count', 0) > 0).astype(np.int8)
    
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
    
    # V11: Add cross features
    df = add_svd_cross_features(df)
    
    df.fillna(0, inplace=True)
    return df


# =============================================================================
# MODEL HELPERS
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
    return lgb.train(params, train_data, num_boost_round=num_rounds)

def train_xgb_reg(cfg, X, y, weights, seed):
    model = xgb.XGBRegressor(**cfg, random_state=seed, verbosity=0)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model

def train_cat_reg(cfg, X, y, weights, seed):
    model = CatBoostRegressor(**cfg, random_seed=seed, verbose=False)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model


def rank_average(predictions):
    """V11: Rank averaging for robust ensemble."""
    ranks = np.array([rankdata(p) for p in predictions])
    return np.mean(ranks, axis=0) / len(predictions[0])


# =============================================================================
# V11 MODEL PARAMETERS (Enhanced diversity)
# =============================================================================
exclude_cols = ['ID', 'customer_id', 'product_unit_variant_id', 'week_start', 'customer_created_at',
                'product_id', 'product_grade_variant_id', 'selling_price',
                'Target_purchase_next_1w', 'Target_qty_next_1w',
                'Target_purchase_next_2w', 'Target_qty_next_2w', 'log_qty_1w', 'log_qty_2w',
                'qty_this_week', 'purchased_this_week', 'spend_this_week', 'num_orders_week',
                'pred_1w', 'week_idx', 'grade_name', 'unit_name', 'customer_category', 'customer_status']

# V11: More XGB configs for diversity
xgb_configs = [
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 6, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.4},
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 7, 'min_child_weight': 80, 'subsample': 0.7, 'colsample_bytree': 0.65, 'reg_alpha': 0.25, 'reg_lambda': 0.45},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 8, 'min_child_weight': 50, 'subsample': 0.65, 'colsample_bytree': 0.6, 'reg_alpha': 0.3, 'reg_lambda': 0.5},
    {'n_estimators': 2000, 'learning_rate': 0.015, 'max_depth': 5, 'min_child_weight': 120, 'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_alpha': 0.15, 'reg_lambda': 0.35},
    {'n_estimators': 1500, 'learning_rate': 0.02, 'max_depth': 6, 'min_child_weight': 100, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.3},
    {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 6, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.4},
    # V11: Additional configs
    {'n_estimators': 2500, 'learning_rate': 0.01, 'max_depth': 6, 'min_child_weight': 90, 'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.2, 'reg_lambda': 0.4},
    {'n_estimators': 1500, 'learning_rate': 0.025, 'max_depth': 7, 'min_child_weight': 70, 'subsample': 0.75, 'colsample_bytree': 0.7, 'reg_alpha': 0.15, 'reg_lambda': 0.35},
]

# V11: More CatBoost configs
cat_configs = [
    {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3},
    {'iterations': 1500, 'learning_rate': 0.02, 'depth': 7, 'l2_leaf_reg': 2},
    {'iterations': 2000, 'learning_rate': 0.015, 'depth': 6, 'l2_leaf_reg': 4},
    {'iterations': 1000, 'learning_rate': 0.03, 'depth': 8, 'l2_leaf_reg': 2},
]

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 6,
    'min_child_samples': 100, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 0.2, 'reg_lambda': 0.4, 'verbose': -1, 'seed': SEED
}

lgb_params_qty = {
    'objective': 'regression_l1', 'metric': 'mae', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'num_leaves': 31, 'max_depth': 7,
    'min_child_samples': 30, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'verbose': -1, 'seed': SEED
}

xgb_qty_config = {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 6, 'min_child_weight': 30,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'objective': 'reg:absoluteerror'}

cat_qty_config = {'iterations': 1000, 'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 2, 'loss_function': 'MAE'}

# V11: Use rank averaging weight (equal contribution)
USE_RANK_AVERAGING = True

# =============================================================================
# PIPELINE
# =============================================================================
log("")
log("=" * 80)
log("V11: Multi-Scale SVD + Enhanced Modelling")
log("=" * 80)

# 1w Classification
log("\nStep 1: Building 1w features (V11 = V2 + Multi-Scale SVD)...")
train_1w = create_features_1w_v11(train, train, max_train_week, svd_embeddings)
test_1w = create_features_1w_v11(train, test, max_train_week, svd_embeddings)

log("  Adding TE features...")
train_1w, test_1w, te_features = add_te_v2(train_1w, test_1w, 'Target_purchase_next_1w')
log(f"  Added {len(te_features)} TE features")

feature_cols_1w = [c for c in train_1w.columns if c not in exclude_cols]
log(f"  Total 1w features: {len(feature_cols_1w)}")

# Count SVD features
svd_feats = [c for c in feature_cols_1w if 'svd' in c.lower() or 'cust_norm' in c or 'prod_norm' in c]
log(f"  SVD-related features: {len(svd_feats)}")

X_1w = train_1w[feature_cols_1w].values.astype(np.float32)
X_test_1w = test_1w[feature_cols_1w].values.astype(np.float32)
y_1w = train_1w['Target_purchase_next_1w'].values

del train_1w, test_1w
gc.collect()

# OOF 1w
log("\nStep 2: OOF 1w predictions...")
oof_1w = np.zeros(len(X_1w), dtype=np.float32)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_1w)):
    X_tr, X_val = X_1w[tr_idx], X_1w[val_idx]
    y_tr, y_val = y_1w[tr_idx], y_1w[val_idx]
    
    preds = []
    for cfg in xgb_configs[:4]:  # Use subset for OOF
        m = train_xgb_model(cfg, X_tr, y_tr, None, SEED)
        preds.append(m.predict_proba(X_val)[:, 1])
    for cfg in cat_configs[:2]:
        m = train_cat_model(cfg, X_tr, y_tr, None, SEED)
        preds.append(m.predict_proba(X_val)[:, 1])
    m = train_lgb_model(lgb_params, X_tr, y_tr, None, 500)
    preds.append(m.predict(X_val))
    
    if USE_RANK_AVERAGING:
        oof_1w[val_idx] = rank_average(preds)
    else:
        oof_1w[val_idx] = np.mean(preds, axis=0)
    log(f"  Fold {fold+1}: AUC = {roc_auc_score(y_val, oof_1w[val_idx]):.6f}")
    gc.collect()

log(f"  Overall OOF AUC: {roc_auc_score(y_1w, oof_1w):.6f}")

# Final 1w
log("\nStep 3: Final 1w models (V11: more seeds)...")
weights_1w = sample_weights if USE_TIME_DECAY_1W else None

all_1w_preds = []
for cfg in tqdm(xgb_configs, desc="  XGB"):
    for s in range(3):  # V11: 3 seeds per config
        m = train_xgb_model(cfg, X_1w, y_1w, weights_1w, SEED + s)
        all_1w_preds.append(m.predict_proba(X_test_1w)[:, 1])

for cfg in tqdm(cat_configs, desc="  Cat"):
    for s in range(3):  # V11: 3 seeds per config
        m = train_cat_model(cfg, X_1w, y_1w, weights_1w, SEED + s)
        all_1w_preds.append(m.predict_proba(X_test_1w)[:, 1])

for s in tqdm(range(7), desc="  LGB"):  # V11: 7 seeds
    m = train_lgb_model({**lgb_params, 'seed': SEED + s}, X_1w, y_1w, weights_1w, 500)
    all_1w_preds.append(m.predict(X_test_1w))

if USE_RANK_AVERAGING:
    pred_1w = rank_average(all_1w_preds)
else:
    pred_1w = np.mean(all_1w_preds, axis=0)
gc.collect()

# 2w Classification
log("\nStep 4: Building 2w features...")
train_2w = create_features_2w_v11(train, train, max_train_week, svd_embeddings)
test_2w = create_features_2w_v11(train, test, max_train_week, svd_embeddings)

log("  Adding TE for 2w...")
train_2w, test_2w, _ = add_te_v2(train_2w, test_2w, 'Target_purchase_next_2w')

train_2w['pred_1w'] = oof_1w
test_2w['pred_1w'] = pred_1w

exclude_2w = [c for c in exclude_cols if c != 'pred_1w']
feature_cols_2w = [c for c in train_2w.columns if c not in exclude_2w]
log(f"  2w features: {len(feature_cols_2w)}")

X_2w = train_2w[feature_cols_2w].values.astype(np.float32)
X_test_2w = test_2w[feature_cols_2w].values.astype(np.float32)
y_2w = train_2w['Target_purchase_next_2w'].values

del train_2w, test_2w
gc.collect()

# 2w Purchase
log("\nStep 5: 2w purchase models...")
lgb_2w_preds = []
for s in tqdm(range(9), desc="  2w LGB"):  # V11: 9 seeds
    m = train_lgb_model({**lgb_params, 'seed': SEED + s}, X_2w, y_2w, None, 500)
    lgb_2w_preds.append(m.predict(X_test_2w))

if USE_RANK_AVERAGING:
    pred_2w = rank_average(lgb_2w_preds)
else:
    pred_2w = np.mean(lgb_2w_preds, axis=0)
gc.collect()

# Quantity
log("\nStep 6: Quantity models...")
train_qty = create_features_1w_v11(train, train, max_train_week, svd_embeddings)
test_qty = create_features_1w_v11(train, test, max_train_week, svd_embeddings)
feature_cols_qty = [c for c in train_qty.columns if c not in exclude_cols]
X_qty = train_qty[feature_cols_qty].values.astype(np.float32)
X_test_qty = test_qty[feature_cols_qty].values.astype(np.float32)
y_log_qty_1w = train['log_qty_1w'].values
y_log_qty_2w = train['log_qty_2w'].values
purch_mask_1w = y_1w == 1
purch_mask_2w = y_2w == 1

del train_qty, test_qty
gc.collect()

log("  1w Quantity...")
w_qty_1w = sample_weights[purch_mask_1w] if USE_TIME_DECAY_1W else None
lgb_preds = [train_lgb_model({**lgb_params_qty, 'seed': SEED+s}, X_qty[purch_mask_1w], y_log_qty_1w[purch_mask_1w], w_qty_1w, 400).predict(X_test_qty) for s in range(7)]
xgb_preds = [train_xgb_reg(xgb_qty_config, X_qty[purch_mask_1w], y_log_qty_1w[purch_mask_1w], w_qty_1w, SEED+s).predict(X_test_qty) for s in range(5)]
cat_preds = [train_cat_reg(cat_qty_config, X_qty[purch_mask_1w], y_log_qty_1w[purch_mask_1w], w_qty_1w, SEED+s).predict(X_test_qty) for s in range(5)]
pred_log_qty_1w = 0.45 * np.mean(lgb_preds, axis=0) + 0.30 * np.mean(xgb_preds, axis=0) + 0.25 * np.mean(cat_preds, axis=0)

log("  2w Quantity...")
w_qty_2w = sample_weights[purch_mask_2w] if USE_TIME_DECAY_2W_QTY else None
lgb_preds = [train_lgb_model({**lgb_params_qty, 'seed': SEED+s}, X_qty[purch_mask_2w], y_log_qty_2w[purch_mask_2w], w_qty_2w, 400).predict(X_test_qty) for s in range(7)]
xgb_preds = [train_xgb_reg(xgb_qty_config, X_qty[purch_mask_2w], y_log_qty_2w[purch_mask_2w], w_qty_2w, SEED+s).predict(X_test_qty) for s in range(5)]
cat_preds = [train_cat_reg(cat_qty_config, X_qty[purch_mask_2w], y_log_qty_2w[purch_mask_2w], w_qty_2w, SEED+s).predict(X_test_qty) for s in range(5)]
pred_log_qty_2w = 0.45 * np.mean(lgb_preds, axis=0) + 0.30 * np.mean(xgb_preds, axis=0) + 0.25 * np.mean(cat_preds, axis=0)

pred_qty_1w = pred_1w * np.clip(np.expm1(pred_log_qty_1w), 0, 500)
pred_qty_2w = pred_2w * np.clip(np.expm1(pred_log_qty_2w), 0, 500)

# Submission
log("\nCreating submission...")
submission['Target_purchase_next_1w'] = pred_1w
submission['Target_qty_next_1w'] = pred_qty_1w
submission['Target_purchase_next_2w'] = pred_2w
submission['Target_qty_next_2w'] = pred_qty_2w
submission.to_csv('submission_v11.csv', index=False)
log("Saved submission_v11.csv")

log("")
log("=" * 80)
log("V11 SUMMARY")
log("=" * 80)
log(f"1w features: {len(feature_cols_1w)} (V2 + {len(svd_feats)} SVD features)")
log(f"2w features: {len(feature_cols_2w)}")
log("")
log("V11 SVD Enhancements:")
log(f"  - Multi-scale SVD: dims={SVD_DIMS}")
log(f"  - Temporal SVD: last {RECENT_WEEKS} weeks")
log("  - Quantity-weighted SVD interactions")
log("  - SVD × recency cross features")
log("  - SVD agreement & temporal diff features")
log("")
log("V11 Modelling Enhancements:")
log(f"  - XGB configs: {len(xgb_configs)} (3 seeds each)")
log(f"  - CatBoost configs: {len(cat_configs)} (3 seeds each)")
log(f"  - LGB seeds: 7 (1w), 9 (2w)")
log(f"  - Rank averaging: {USE_RANK_AVERAGING}")
log("")
log("Expected: Improvement over V10 baseline")
log("=" * 80)
