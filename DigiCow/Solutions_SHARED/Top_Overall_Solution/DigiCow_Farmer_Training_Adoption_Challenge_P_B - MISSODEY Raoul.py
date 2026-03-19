# ============================================================
# DigiCow Adoption Prediction 
# ============================================================
import re, ast, warnings, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.special import expit, logit, logsumexp
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL CONFIG
# ============================================================
RANDOM_STATE = 42
N_SPLITS = 5
EPS = 1e-15 # to prevent log error
SEEDS = [2000, 2013, 2026]
TOPK_TOPICS = 200 # Number of top topics to create binary flags
TE_ALPHA = 90.0 # Smoothing parameter for target encoding
WINDOWS_DAYS = [7, 30, 90, 180] # mooving window sizes in days
TARGET_N_FEATURES = 500 # Feature selection: keep top N features
USE_GPU = True

SHRINK_LL = {"07": 0.0, "90": 0.0, "120": 0.0}   # Calibration shrinkage factors for LogLoss optimization
#here 0.0 beacuse I tried several parameters which does'nt give accurate result. 
SHRINK_AUC = {"07": 0.003, "90": 0.004, "120": 0.004} ## Calibration shrinkage factors for AUC optimization


# Hierarchical keys for historical rate features
HIST_KEYS = [
    "trainer_first",
    "county", "subcounty", "ward", 
    "group_name",
    "county_trainer", "subcounty_trainer",
    "county_subcounty", "subcounty_ward"
]

# ============================================================
# ALL FEATURE ENGINEERING FUNCTIONS
# ============================================================

def safe_literal_eval(x):
    #Safely parse string representations of lists/dicts
    if pd.isna(x): return None
    if isinstance(x, (list, dict)): return x
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}: return None
    try: return ast.literal_eval(s)
    except: return None

def parse_trainer_first(x):
    #Extract first trainer ID from potentially nested list/string format
    if pd.isna(x): return "UNK"
    s = str(x).strip()
    if s.startswith("TRA_") and "[" not in s: return s
    obj = safe_literal_eval(s)
    if isinstance(obj, list) and len(obj) > 0: return str(obj[0]).strip()
    if isinstance(obj, str) and obj.strip(): return obj.strip()
    return "UNK"

def flatten_topics(x):
#Convert nested topic lists to flat list of strings
    obj = safe_literal_eval(x)
    if obj is None: return []
    if isinstance(obj, list):
        out = []
        for el in obj:
            if isinstance(el, list): out.extend([str(t).strip() for t in el if str(t).strip()])
            else: out.append(str(el).strip())
        return [t for t in out if t]
    if isinstance(obj, str) and obj.strip(): return [obj.strip()]
    return []

def norm_topic(t: str) -> str:
#Normalize topic text: lowercase, remove trailing dots, collapse whitespace
    t = t.lower().strip()
    t = re.sub(r"\.+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def topics_to_text(topics):
    #Convert topic list to space-separated text (underscores replace spaces in each topic)
    return " ".join([re.sub(r"\s+", "_", norm_topic(t)) for t in topics if t])

def topics_to_token_str(topics_norm):
    #Convert normalized topics to pipe-separated sorted string for deduplication
    return "|".join(sorted(set(topics_norm)))


# Keyword lists for topic categorization

KW_POULTRY = ["poultry", "chicken", "layers", "broiler", "kienyeji"]
KW_DAIRY   = ["dairy", "cow", "calf", "milking", "maziwa", "silage", "milk"]
KW_HEALTH  = ["health", "disease", "vaccin", "biosecurity", "hygiene", "ppe",
              "resistance", "parasite", "worm", "deworming", "aflatoxin",
              "antimicrobial"]
KW_FEED    = ["feed", "feeding", "nutrition", "silage", "mineral", "supplement",
              "fodder", "tyari"]
KW_MGMT    = ["management", "record", "housing", "breeding", "practices",
              "practice", "transition", "mating", "heat", "structure",
              "insemination", "ai fails"]
KW_CROPS   = ["fertilizer", "harvest", "seed", "crop", "plant", "soil",
              "agronomy", "maize", "beans", "pest"]
KW_SHOATS  = ["sheep", "goat", "shoat", "ewe", "ram"]
KW_BIOGAS  = ["biogas", "energy", "manure"]
KW_FINANCE = ["finance", "money", "loan", "kcb", "credit", "price", "market",
              "profit", "app", "digital", "bank", "shilling", "cost", "sell",
              "sale", "buyer", "ndume"]

# Grouped keyword sets with category names
ALL_KW_GROUPS = [
    ("poultry", KW_POULTRY), ("dairy", KW_DAIRY), ("health", KW_HEALTH),
    ("feed", KW_FEED), ("management", KW_MGMT), ("crops", KW_CROPS),
    ("shoats", KW_SHOATS), ("biogas", KW_BIOGAS), ("finance", KW_FINANCE)
]

def count_kw(topics_norm, keywords):
# Count how many topics match any keyword in the provided list
    c = 0
    for t in topics_norm:
        for kw in keywords:
            if kw in t: c += 1; break # Count once per topic
    return c

def get_topic_category(topic_norm):
#Assign a single category to a topic based on keyword matching
    for name, kws in ALL_KW_GROUPS:
        for kw in kws:
            if kw in topic_norm: return name
    return "other"

def add_topic_features(df):
    """
    This function creates topic-related features:
    - parse and normalize topics
    - count topics per category (poultry, dairy, health, etc.)
    - calculate diversity metrics
    - generate text representations for TF-IDF/SVD
    """
    df = df.copy()
    df["topics_flat"] = df["topics_list"].apply(flatten_topics)
    df["topics_norm"] = df["topics_flat"].apply(lambda xs: [norm_topic(str(x)) for x in xs])
    df["num_topics_raw"] = df["topics_norm"].apply(len).astype(int)
    df["num_topics"] = df["num_topics_raw"]
    for name, kws in ALL_KW_GROUPS:
        df[f"topics_{name}"] = df["topics_norm"].apply(lambda xs: count_kw(xs, kws)).astype(int)
        df[f"has_{name}_topics"] = (df[f"topics_{name}"] > 0).astype(int)
    df["topics_diversity"] = sum(
        (df[f"topics_{n}"] > 0).astype(int) for n, _ in ALL_KW_GROUPS
    ).astype(int)
    # Text representations
    df["topics_text"] = df["topics_flat"].apply(topics_to_text).fillna("")
    df["topics_token_str"] = df["topics_norm"].apply(topics_to_token_str)
    df["topic_categories"] = df["topics_norm"].apply(
        lambda xs: list(set(get_topic_category(t) for t in xs)))
    df["topic_cat_str"] = df["topic_categories"].apply(lambda xs: "|".join(sorted(xs)))
    #Determine primary category by most common occurrence
    def primary_cat(topics_norm):
        cats = [get_topic_category(t) for t in topics_norm]
        if not cats: return "none"
        return Counter(cats).most_common(1)[0][0]
    df["primary_topic_cat"] = df["topics_norm"].apply(primary_cat)
    df["num_unique_topics"] = df["topics_norm"].apply(lambda xs: len(set(xs))).astype(int)
    df["topic_redundancy"] = df["num_topics"] - df["num_unique_topics"]
    return df

def build_topic_svd_embeddings(train_texts, test_texts, prior_texts, n_components=20):
    """
    This function generates dense topic embeddings via TF-IDF + SVD:
    1. Combine all text (train + test + prior)
    2. Fit TF-IDF vectorizer
    3. Apply TruncatedSVD for dimensionality reduction
    and returns: (train_embeddings, test_embeddings, prior_embeddings)
    """
    print("  Building Topic SVD embeddings :")
    all_texts = pd.concat([train_texts, test_texts, prior_texts], ignore_index=True).fillna("")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                            min_df=5, max_df=0.95, sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    n_train, n_test = len(train_texts), len(test_texts)
    # see capture variance in order to increase n_component if needed
    print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    return svd_matrix[:n_train], svd_matrix[n_train:n_train+n_test], svd_matrix[n_train+n_test:]

def add_date_features(df):

    """
    Extract temporal features from training_day:
    - Calendar features (month, quarter, day of week, etc.)
    - Cyclical encodings (sin/cos for periodicity)
    - Time-based flags (weekend, Q4, year 2025, etc.)
    """

    df = df.copy()
    df["training_day"] = pd.to_datetime(df["training_day"], errors="coerce")
    df = df[df["training_day"].notna()].copy()

    # Calendar components
    df["training_month"] = df["training_day"].dt.month.astype(int)
    df["training_quarter"] = df["training_day"].dt.quarter.astype(int)
    df["training_day_of_week"] = df["training_day"].dt.dayofweek.astype(int)
    df["training_day_of_month"] = df["training_day"].dt.day.astype(int)
    df["training_year"] = df["training_day"].dt.year.astype(int)
    df["training_week_of_year"] = df["training_day"].dt.isocalendar().week.astype(int)
    df["training_day_of_year"] = df["training_day"].dt.dayofyear.astype(int)

    # Days in month metrics
    df["training_days_in_month"] = df["training_day"].dt.days_in_month.astype(int)
    df["training_days_to_month_end"] = df["training_days_in_month"] - df["training_day_of_month"]
    m = df["training_month"].astype(float)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    dow = df["training_day_of_week"].astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["is_weekend"] = (df["training_day_of_week"] >= 5).astype(int)
    ref_date = pd.Timestamp("2024-01-01")
    df["days_since_ref"] = (df["training_day"] - ref_date).dt.days.astype(int)
    df["weeks_since_ref"] = df["days_since_ref"] // 7
    df["is_q4"] = (df["training_quarter"] == 4).astype(int)
    df["is_q1"] = (df["training_quarter"] == 1).astype(int)
    df["is_year_2025"] = (df["training_year"] == 2025).astype(int)
    df["month_year_idx"] = (df["training_year"] - 2024) * 12 + df["training_month"]
    doy = df["training_day_of_year"].astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    return df

def add_training_dt(df):

    """
    Create training datetime by adding intra-day rank:
    """

    df = df.copy()
    if "ID" not in df.columns:
        df["ID"] = np.arange(len(df)).astype(str)
    df = df.sort_values(["farmer_name", "training_day", "ID"]).reset_index(drop=True)
    df["intra_day_rank"] = df.groupby(["farmer_name", "training_day"]).cumcount().astype(int)
    df["training_dt"] = df["training_day"] + pd.to_timedelta(df["intra_day_rank"], unit="s")
    return df

def add_hier_keys(df):

    """
    Create hierarchical composite keys for target encoding:
    - Geographic : county_subcounty, subcounty_ward
    - Trainer: county_trainer, subcounty_trainer, ward_trainer
    - Group : group_trainer
    """
    df = df.copy()
    # Geographic hierarchies
    for col1, col2 in [("county","subcounty"),("subcounty","ward"),
                         ("county","trainer_first"),("subcounty","trainer_first")]:
        new_name = f"{col1}_{col2}".replace("trainer_first","trainer")
        df[new_name] = (df[col1].fillna("unk").astype(str).str.lower().str.strip()
                        + "_" + df[col2].fillna("unk").astype(str).str.lower().str.strip())
    df["group_trainer"] = (df["group_name"].fillna("unk").astype(str)
                           + "_" + df["trainer_first"].fillna("unk").astype(str))
    df["ward_trainer"] = (df["ward"].fillna("unk").astype(str).str.lower().str.strip()
                          + "_" + df["trainer_first"].fillna("unk").astype(str))
    return df

def get_top_topics(train_df, prior_df, k=120):
    """
    Identify top-k most frequent topics across train + prior data.
    """
    all_t = []
    for df in [train_df, prior_df]:
        tmp = df["topics_list"].apply(flatten_topics).apply(lambda xs: [norm_topic(x) for x in xs])
        all_t.extend([t for xs in tmp for t in xs])
    return list(pd.Series(all_t).value_counts().head(k).index)

def add_topk_topic_flags(df, top_topics):
    """
    Create binary flags for top-k most common topics:
    - top_topic_000, top_topic_001, ... (one-hot encoding)
    - topk_topics_sum: total count of top topics present
    """

    df = df.copy()
    topic_sets = df["topics_norm"].apply(set)
    for i, t in enumerate(top_topics):
        df[f"top_topic_{i:03d}"] = topic_sets.apply(lambda s: 1 if t in s else 0).astype(np.int8)
    df["topk_topics_sum"] = df[[c for c in df.columns if c.startswith("top_topic_")]].sum(axis=1).astype(int)
    return df

def compute_adversarial_weights(train_df, test_df, feature_cols, cat_cols):
    """
    Adversarial validation: train classifier to distinguish train vs test.
    Higher probabilities ---> train samples are less representative of test.
    Returns: sample weights for reweighting train to match test distribution.
    """
    import lightgbm as lgb
    print("Computing adversarial validation weights :")
    # Select features
    text_skip = {"topics_text"}
    use_cols = [c for c in feature_cols if c not in text_skip and c in train_df.columns and c in test_df.columns]
    cat_use = [c for c in cat_cols if c in use_cols]

    X_tr = train_df[use_cols].copy()
    X_te = test_df[use_cols].copy()
    X = pd.concat([X_tr, X_te], ignore_index=True)
    for c in X.columns:
        if X[c].dtype == "object": X[c] = X[c].astype("category")
    y = np.array([0]*len(X_tr) + [1]*len(X_te))

    # LightGBM params for adversarial validation
    params = {
        "objective": "binary", "metric": "auc", "boosting": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": 5,
        "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.5,
        "reg_lambda": 10.0, "verbose": -1, "seed": RANDOM_STATE, "n_jobs": -1,
    }
    # 3-fold CV to get OOF predictions
    oof_preds = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, va_idx in skf.split(X, y):
        dtrain = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_use)
        dval = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_use)
        m = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds[va_idx] = m.predict(X.iloc[va_idx])
    train_probs = oof_preds[:len(X_tr)]
    adv_auc = roc_auc_score(y, oof_preds)
    print(f"  Adversarial AUC: {adv_auc:.4f}")
    weights = train_probs / (1 - train_probs + 1e-7)
    weights = np.clip(weights, 0.2, 5.0)
    weights = weights / weights.mean()
    print(f"  Adversarial weights: min={weights.min():.3f}, max={weights.max():.3f}")
    return weights

# ============================================================
# PRIOR ASOF FEATURES 
# ============================================================
def build_prior_asof_features(prior_df, main_df, time_col="training_dt"):
    """
    Build expanding window features from prior history using asof merge:
    - Cumulative counts/sums/rates for adoption outcomes
    - Time gaps between sessions
    - Adoption trends 
    - Session streaks and patterns
    # avoid leakage
    """
    p = prior_df.copy()
    p = p.sort_values(["farmer_name", time_col]).reset_index(drop=True)
    grp = p.groupby("farmer_name", sort=False)
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    p["prior_training_count"] = grp.cumcount() + 1

    # Cumulative adoption metrics (sum and rate for each window)
    for w in ["07","90","120"]:
        col = f"adopted_within_{w}_days"
        p[f"prior_adopted_{w}_sum"] = grp[col].cumsum()
        p[f"prior_adopted_{w}_rate"] = p[f"prior_adopted_{w}_sum"] / p["prior_training_count"]
    p["has_topic_trained_on"] = pd.to_numeric(p.get("has_topic_trained_on", 0), errors="coerce").fillna(0).astype(int)
    p["belong_to_cooperative"] = pd.to_numeric(p.get("belong_to_cooperative", 0), errors="coerce").fillna(0).astype(int)

    # Cumulative rates for binary features
    p["prior_has_topic_rate"] = grp["has_topic_trained_on"].cumsum() / p["prior_training_count"]
    p["prior_coop_rate"] = grp["belong_to_cooperative"].cumsum() / p["prior_training_count"]

    # Topic count aggregates
    p["prior_topics_total"] = grp["num_topics"].cumsum()
    p["prior_topics_avg"] = p["prior_topics_total"] / p["prior_training_count"]
    p["prior_topics_max"] = grp["num_topics"].cummax()

    # Time gaps between sessions
    p["prev_dt"] = grp[time_col].shift(1)
    p["secs_since_prev"] = (p[time_col] - p["prev_dt"]).dt.total_seconds().fillna(0)
    p["prior_avg_gap_secs"] = grp["secs_since_prev"].transform(lambda x: x.expanding().mean())
    p["prior_std_gap_secs"] = grp["secs_since_prev"].transform(lambda x: x.expanding().std()).fillna(0)

    # Recent adoption trends 
    for w in ["07","90","120"]:
        col = f"adopted_within_{w}_days"
        p[f"recent_adopted_{w}_rate"] = grp[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
        p[f"adoption_trend_{w}"] = p[f"recent_adopted_{w}_rate"] - p[f"prior_adopted_{w}_rate"]
        p[f"prior_ever_adopted_{w}"] = (p[f"prior_adopted_{w}_sum"] > 0).astype(int)
        p[f"ewm_adopted_{w}"] = grp[col].transform(lambda s: s.ewm(halflife=5, min_periods=1).mean())
    for w in ["90","120"]:
        col = f"adopted_within_{w}_days"
        for n in [3, 5, 10]:
            p[f"adopt_{w}_rate_last{n}"] = grp[col].transform(
                lambda x: x.rolling(n, min_periods=1).mean())
        p[f"adopt_accel_{w}"] = grp[f"recent_adopted_{w}_rate"].transform(
            lambda x: x.diff().rolling(3, min_periods=1).mean()
        ).fillna(0)
    for w in ["07","90","120"]:
        col = f"adopted_within_{w}_days"
        def sessions_since_last_adopt(s):
            out = np.full(len(s), -1, dtype=float)
            last = -1
            for i, v in enumerate(s.values):
                if last >= 0: out[i] = i - last
                if v == 1: last = i
            return pd.Series(out, index=s.index)
        p[f"sessions_since_last_adopt_{w}"] = grp[col].transform(sessions_since_last_adopt)

    #add cumulative unique trainers count
    def cum_unique_trainers(s):
        seen = set()
        out = []
        for v in s.values:
            seen.add(str(v))
            out.append(len(seen))
        return pd.Series(out, index=s.index)
    p["prior_unique_trainers"] = grp["trainer_first"].transform(cum_unique_trainers)
    p["first_dt"] = grp[time_col].transform("first")
    p["months_active"] = ((p[time_col] - p["first_dt"]).dt.total_seconds() / (30*24*3600)).clip(lower=0.5)
    p["sessions_per_month"] = p["prior_training_count"] / p["months_active"]
    p["gap_trend"] = grp["secs_since_prev"].transform(
        lambda x: x.rolling(5, min_periods=2).apply(
            lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) >= 2 else 0, raw=False)
    ).fillna(0)
    for w in ["120"]:
        col = f"adopted_within_{w}_days"
        def non_adopt_streak(s):
            out = np.zeros(len(s), dtype=float)
            streak = 0
            for i, v in enumerate(s.values):
                out[i] = streak
                if v == 0: streak += 1
                else: streak = 0
            return pd.Series(out, index=s.index)
        p[f"non_adopt_streak_{w}"] = grp[col].transform(non_adopt_streak)

# Cumulative unique topics
    def cum_unique_topics(topics_series):
        seen = set()
        out = []
        for tlist in topics_series.values:
            if isinstance(tlist, list): seen.update(tlist)
            out.append(len(seen))
        return pd.Series(out, index=topics_series.index)
    p["cum_unique_topics"] = grp["topics_norm"].transform(cum_unique_topics)
    p["cum_diversity"] = grp["topics_diversity"].transform(lambda x: x.expanding().mean())

     # Keep last session metadata for comparison with current session
    p["prior_last_dt"] = p[time_col]
    p["prior_last_topics_token_str"] = p["topics_token_str"]
    p["prior_last_trainer_first"] = p["trainer_first"]
    keep_cols = ["farmer_name", time_col, "prior_training_count"]
    for w in ["07","90","120"]:
        keep_cols += [
            f"prior_adopted_{w}_sum", f"prior_adopted_{w}_rate",
            f"recent_adopted_{w}_rate", f"adoption_trend_{w}",
            f"prior_ever_adopted_{w}", f"sessions_since_last_adopt_{w}",
            f"ewm_adopted_{w}",
        ]
    for w in ["90","120"]:
        for n in [3,5,10]:
            keep_cols.append(f"adopt_{w}_rate_last{n}")
        keep_cols.append(f"adopt_accel_{w}")
    keep_cols += [
        "prior_has_topic_rate", "prior_coop_rate",
        "prior_topics_total", "prior_topics_avg", "prior_topics_max",
        "prior_avg_gap_secs", "prior_std_gap_secs",
        "prior_last_dt", "prior_last_topics_token_str", "prior_last_trainer_first",
        "prior_unique_trainers", "sessions_per_month",
        "gap_trend", "non_adopt_streak_120",
        "cum_unique_topics", "cum_diversity",
    ]
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in p.columns))
    p_feat = p[keep_cols].copy()
    m = main_df.copy()
    m = m.sort_values([time_col, "farmer_name"]).reset_index(drop=True)
    p_feat = p_feat.sort_values([time_col, "farmer_name"]).reset_index(drop=True)
    m = pd.merge_asof(m, p_feat, on=time_col, by="farmer_name",
                       direction="backward", allow_exact_matches=False)
  
    # Handle metadata columns
    if "prior_last_dt" in m.columns:
        m["prior_last_dt"] = pd.to_datetime(m["prior_last_dt"], errors="coerce")
    if "prior_last_topics_token_str" in m.columns:
        m["prior_last_topics_token_str"] = m["prior_last_topics_token_str"].fillna("")
    if "prior_last_trainer_first" in m.columns:
        m["prior_last_trainer_first"] = m["prior_last_trainer_first"].fillna("UNK")

    # Fill missing values
    exclude = {"prior_last_dt", "prior_last_topics_token_str", "prior_last_trainer_first"}
    fill_cols = [c for c in m.columns if
                 (c.startswith("prior_") or c.startswith("recent_") or
                  c.startswith("adoption_trend") or c.startswith("sessions_since") or
                  c.startswith("adopt_") or c.startswith("ewm_") or
                  c in ["prior_unique_trainers","sessions_per_month",
                         "gap_trend","non_adopt_streak_120",
                         "cum_unique_topics","cum_diversity"])
                 and c not in exclude]
    for c in fill_cols:
        if c in m.columns:
            m[c] = m[c].fillna(-1 if "sessions_since" in c else 0)
    m["secs_since_prior"] = (m[time_col] - m["prior_last_dt"]).dt.total_seconds()
    m["secs_since_prior"] = m["secs_since_prior"].fillna(1e12).clip(lower=0)

    #Calculate overlap metrics between current and last session topics
    def overlap_stats(curr_str, last_str):
        if pd.isna(last_str) or last_str == "": return 0, 0.0, 0, 0.0
        curr = set(curr_str.split("|")) if curr_str else set()
        last = set(last_str.split("|")) if last_str else set()
        inter = len(curr & last)
        union = len(curr | last) if (curr or last) else 0
        jacc = (inter / union) if union else 0.0
        new_cnt = max(len(curr) - inter, 0)
        rep_ratio = inter / max(len(curr), 1)
        return inter, jacc, new_cnt, rep_ratio
    tmp = m.apply(lambda r: overlap_stats(
        r["topics_token_str"], r["prior_last_topics_token_str"]), axis=1)
    m["overlap_cnt_last"]  = [t[0] for t in tmp]
    m["jaccard_last"]      = [t[1] for t in tmp]
    m["new_topics_cnt"]    = [t[2] for t in tmp]
    m["repeat_ratio_last"] = [t[3] for t in tmp]
    m["same_trainer_as_last"] = (
        (m["trainer_first"].astype(str) == m["prior_last_trainer_first"].astype(str)) &
        (m["prior_training_count"] > 0)
    ).astype(int)
    m.drop(columns=["prior_last_dt"], inplace=True, errors="ignore")
    return m.sort_values("ID").reset_index(drop=True)

def build_topic_lift_scores(prior_df):

    """
    This function calculates lift scores for each topic (how much does a topic increase adoption?):
    - Base rates: overall adoption rates for each window
    - Topic rates: adoption rates when topic is present (with Bayesian smoothing)
    - Lift: topic_rate / base_rate (>1 means topic boosts adoption)
    Returns: (lift_dict, base_rates)
    """

    p = prior_df.copy()
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0

    # calculate global base rates
    base_rates = {
        "07": p["adopted_within_07_days"].mean(),
        "90": p["adopted_within_90_days"].mean(),
        "120": p["adopted_within_120_days"].mean(),
    }
    p_exp = p[["topics_norm","adopted_within_07_days","adopted_within_90_days",
               "adopted_within_120_days"]].copy()
    p_exp = p_exp.explode("topics_norm").rename(columns={"topics_norm":"topic"})
    p_exp["topic"] = p_exp["topic"].astype(str)
    p_exp = p_exp[p_exp["topic"] != ""].copy()

    # Calculate Bayesian smoothed rates and lift for each topic
    alpha = 50.0
    lift_dict = {}
    for topic, tdf in p_exp.groupby("topic"):
        n = len(tdf)
        if n < 5: continue # skip rare topics
        for w in ["07","90","120"]:
            col = f"adopted_within_{w}_days"
            rate = (tdf[col].sum() + alpha * base_rates[w]) / (n + alpha)
            lift = rate / max(base_rates[w], 1e-10)
            lift_dict[(topic, w)] = {"rate": rate, "lift": lift, "count": n}
    return lift_dict, base_rates

def add_topic_lift_features(df, lift_dict, base_rates):
    """
    Add topic lift features to dataframe:
    - Average/max lift across all topics in session
    - Average/max adoption rate for topics
    - Lift standard deviation 
    """
    df = df.copy()
    def compute_lifts(topics_norm, window):
    #Aggregate lift scores across all topics in a session
        lifts = []; rates = []
        for t in topics_norm:
            key = (t, window)
            if key in lift_dict:
                lifts.append(lift_dict[key]["lift"])
                rates.append(lift_dict[key]["rate"])
        if not lifts:
            return base_rates[window], 1.0, base_rates[window], 1.0, 0.0
        return (np.mean(rates), np.mean(lifts), np.max(rates), np.max(lifts), np.std(lifts))
    # calculate lift features for each window
    for w in ["07","90","120"]:
        results = df["topics_norm"].apply(lambda xs: compute_lifts(xs, w))
        df[f"topic_avg_rate_{w}"] = [r[0] for r in results]
        df[f"topic_avg_lift_{w}"] = [r[1] for r in results]
        df[f"topic_max_rate_{w}"] = [r[2] for r in results]
        df[f"topic_max_lift_{w}"] = [r[3] for r in results]
        df[f"topic_lift_std_{w}"] = [r[4] for r in results]
    return df

def add_hist_rates_from_prior(prior_df, main_df, key, alpha=60.0, time_col="training_dt"):

    """
    calculate historical adoption rates for a grouping key (county, trainer,...):
    - Expanding window: each row sees only prior events for that key
    - Bayesian smoothing
    - Asof merge: join based on timestamp to prevent leakage
    
    Args:
        key: Grouping column (e.g., "county", "trainer_first")
        alpha: Smoothing parameter for Bayesian average
    """

    p = prior_df.copy()
    if key not in p.columns: return main_df
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    g07 = p["adopted_within_07_days"].mean()
    g90 = p["adopted_within_90_days"].mean()
    g120 = p["adopted_within_120_days"].mean()

    # Sort by key and time for expanding window
    p = p.sort_values([key, time_col]).reset_index(drop=True)
    grp = p.groupby(key, sort=False)

     # Cumulative counts and sums
    p[f"{key}_hist_cnt"] = grp.cumcount() + 1
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        p[f"{key}_hist_sum{w}"] = grp[f"adopted_within_{w}_days"].cumsum()
        p[f"{key}_hist_rate{w}"] = (p[f"{key}_hist_sum{w}"] + alpha*gm) / (p[f"{key}_hist_cnt"] + alpha)
    p[f"{key}_hist_last_dt"] = p[time_col]
    keep = [key, time_col, f"{key}_hist_cnt"] + \
           [f"{key}_hist_rate{w}" for w in ["07","90","120"]] + [f"{key}_hist_last_dt"]
    p_feat = p[keep].copy()
    m = main_df.copy()
    m = m.sort_values([time_col, key]).reset_index(drop=True)
    p_feat = p_feat.sort_values([time_col, key]).reset_index(drop=True)
    m = pd.merge_asof(m, p_feat, on=time_col, by=key, direction="backward", allow_exact_matches=False)
    m[f"{key}_hist_cnt"] = m[f"{key}_hist_cnt"].fillna(0)
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        m[f"{key}_hist_rate{w}"] = m[f"{key}_hist_rate{w}"].fillna(gm)

    # Time since last event for this key
    m[f"{key}_hist_secs_since_last"] = (m[time_col] - m[f"{key}_hist_last_dt"]).dt.total_seconds()
    m[f"{key}_hist_secs_since_last"] = m[f"{key}_hist_secs_since_last"].fillna(1e12).clip(lower=0)
    m.drop(columns=[f"{key}_hist_last_dt"], inplace=True)
    return m.sort_values("ID").reset_index(drop=True)

def add_topic_hist_features(prior_df, main_df, top_topics, alpha=60.0, time_col="training_dt"):

    """
    calculate historical rates at the individual topic level:
    - Explode topics: one row per topic occurrence
    - Expanding rates: each topic gets its own historical adoption rate
    - Aggregate back: mean/max/std of topic rates per session
    """

    p = prior_df.copy()
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    g07=p["adopted_within_07_days"].mean(); g90=p["adopted_within_90_days"].mean(); g120=p["adopted_within_120_days"].mean()
    top_set = set(top_topics)
    p_exp = p[[time_col,"topics_norm","adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]].copy()
    p_exp = p_exp.explode("topics_norm").rename(columns={"topics_norm":"topic"})
    p_exp["topic"] = p_exp["topic"].astype(str)
    p_exp = p_exp[p_exp["topic"].isin(top_set)].copy()

    # Sort and calculate expanding rates per topic
    p_exp = p_exp.sort_values(["topic", time_col]).reset_index(drop=True)
    grp = p_exp.groupby("topic", sort=False)
    p_exp["topic_cnt"] = grp.cumcount() + 1
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        col = f"adopted_within_{w}_days"
        p_exp[f"topic_sum{w}"] = grp[col].cumsum()
        p_exp[f"topic_rate{w}"] = (p_exp[f"topic_sum{w}"] + alpha*gm) / (p_exp["topic_cnt"] + alpha)
    p_exp["topic_last_dt"] = p_exp[time_col]
    p_feat = p_exp[["topic",time_col,"topic_cnt","topic_last_dt","topic_rate07","topic_rate90","topic_rate120"]].copy()
    m = main_df.copy()
    m_exp = m[["ID",time_col,"topics_norm"]].copy().explode("topics_norm").rename(columns={"topics_norm":"topic"})
    m_exp["topic"] = m_exp["topic"].astype(str)
    m_exp = m_exp[m_exp["topic"].isin(top_set)].copy()
    m_exp = m_exp.sort_values([time_col,"topic"]).reset_index(drop=True)
    p_feat = p_feat.sort_values([time_col,"topic"]).reset_index(drop=True)
    m_exp = pd.merge_asof(m_exp, p_feat, on=time_col, by="topic", direction="backward", allow_exact_matches=False)
    m_exp["topic_cnt"] = m_exp["topic_cnt"].fillna(0)
    for w,gm in [("07",g07),("90",g90),("120",g120)]:
        m_exp[f"topic_rate{w}"] = m_exp[f"topic_rate{w}"].fillna(gm)
    m_exp["topic_secs_since_last"] = (m_exp[time_col] - m_exp["topic_last_dt"]).dt.total_seconds()
    m_exp["topic_secs_since_last"] = m_exp["topic_secs_since_last"].fillna(1e12).clip(lower=0)

    # Aggregate topic-level features back to session level
    agg_rules = {
        "topic_cnt": ["sum","max","mean"],
        "topic_secs_since_last": ["min","mean"],
        "topic_rate07": ["mean","max"],
        "topic_rate90": ["mean","max"],
        "topic_rate120": ["mean","max","std"],
    }
    agg = m_exp.groupby("ID").agg(agg_rules)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    m = m.merge(agg, on="ID", how="left")
    # Fill missing (sessions with no top topics)
    for c in agg.columns:
        if c == "ID": continue
        if "std" in c: m[c] = m[c].fillna(0)
        elif "rate07" in c: m[c] = m[c].fillna(g07)
        elif "rate90" in c: m[c] = m[c].fillna(g90)
        elif "rate120" in c: m[c] = m[c].fillna(g120)
        elif "secs_since" in c: m[c] = m[c].fillna(1e12)
        else: m[c] = m[c].fillna(0)
    return m

def add_window_features_from_prior(prior_df, main_df, key, windows_days, time_col="training_dt"):

    """
    Calculate rolling window statistics for a grouping key:
    - Count of events in last N days
    - Sum of adoptions in each window (7d, 90d, 120d)
    - Adoption rates in each window
    
    """

    p = prior_df.copy()
    m = main_df.copy()
    if key not in p.columns or key not in m.columns: return m
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    p = p.sort_values([key, time_col]).reset_index(drop=True)
    m = m.sort_values([key, time_col]).reset_index(drop=True)
    day_ns = 24*3600*1_000_000_000 # Nanosecond in a day
    out_cols = []
    for w in windows_days:
        for suf in ["cnt","sum07","sum90","sum120","rate07","rate90","rate120"]:
            c = f"{key}_{suf}_last_{w}d"
            m[c] = 0 if "cnt" in suf or "sum" in suf else 0.0
            out_cols.append(c)

    # Precompute cumulative sums for efficient range queries
    p_groups = {g: df for g, df in p.groupby(key, sort=False)}
    for g, m_g in m.groupby(key, sort=False):
        if g not in p_groups: continue
        p_g = p_groups[g]
    
        # Convert timestamps to int64 for binary search
        p_ts = p_g[time_col].astype("datetime64[ns]").values.astype(np.int64)
        p_c07 = p_g["adopted_within_07_days"].values.astype(np.int64)
        p_c90 = p_g["adopted_within_90_days"].values.astype(np.int64)
        p_c120 = p_g["adopted_within_120_days"].values.astype(np.int64)

        # Prefix sums
        pref_cnt = np.concatenate([[0], np.cumsum(np.ones(len(p_g), dtype=np.int64))])
        pref_07 = np.concatenate([[0], np.cumsum(p_c07)])
        pref_90 = np.concatenate([[0], np.cumsum(p_c90)])
        pref_120 = np.concatenate([[0], np.cumsum(p_c120)])
        m_ts = m_g[time_col].astype("datetime64[ns]").values.astype(np.int64)

        # Binary search: find events in [current_time - window, current_time)
        right = np.searchsorted(p_ts, m_ts, side="left")
        for w in windows_days:
            left = np.searchsorted(p_ts, m_ts - w*day_ns, side="left")
            cnt = right - left
            s07 = pref_07[right] - pref_07[left]
            s90 = pref_90[right] - pref_90[left]
            s120 = pref_120[right] - pref_120[left]
            idxs = m_g.index.values
            m.loc[idxs, f"{key}_cnt_last_{w}d"] = cnt
            m.loc[idxs, f"{key}_sum07_last_{w}d"] = s07
            m.loc[idxs, f"{key}_sum90_last_{w}d"] = s90
            m.loc[idxs, f"{key}_sum120_last_{w}d"] = s120
            denom = np.maximum(cnt, 1)
            m.loc[idxs, f"{key}_rate07_last_{w}d"] = s07 / denom
            m.loc[idxs, f"{key}_rate90_last_{w}d"] = s90 / denom
            m.loc[idxs, f"{key}_rate120_last_{w}d"] = s120 / denom
    for c in out_cols: m[c] = m[c].fillna(0)
    return m.sort_values("ID").reset_index(drop=True)

def add_days_since_last_adopt(prior_df, main_df, key="farmer_name", time_col="training_dt"):

    """
    Calculate time elapsed since last adoption event:
    - For each adoption window (7d, 90d, 120d)
    - Find last positive adoption event in prior
    - Asof merge to get recency in seconds
    
    High values ---> farmer hasn't adopted recently .
    """

    p = prior_df.copy()
    m = main_df.copy()
    if key not in p.columns or key not in m.columns: return m
    p = p.sort_values([time_col, key]).reset_index(drop=True)
    m = m.sort_values([time_col, key]).reset_index(drop=True)
    for w in ["07","90","120"]:
        col = f"adopted_within_{w}_days"
        if col not in p.columns: p[col] = 0
        else: p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0).astype(int)
        p_pos = p.loc[p[col]==1, [key, time_col]].copy()
        p_pos = p_pos.rename(columns={time_col: f"last_adopt_{w}_dt"})
        p_pos = p_pos.sort_values([f"last_adopt_{w}_dt", key]).reset_index(drop=True)
        m = pd.merge_asof(m, p_pos, left_on=time_col, right_on=f"last_adopt_{w}_dt",
                           by=key, direction="backward", allow_exact_matches=False)
        m[f"secs_since_last_adopt_{w}"] = (m[time_col] - m[f"last_adopt_{w}_dt"]).dt.total_seconds()
        m[f"secs_since_last_adopt_{w}"] = m[f"secs_since_last_adopt_{w}"].fillna(1e12).clip(lower=0)
        m.drop(columns=[f"last_adopt_{w}_dt"], inplace=True)
    return m.sort_values("ID").reset_index(drop=True)

def add_geo_temporal_trends(prior_df, main_df, time_col="training_dt"):

    """
    Calculate geographic-level temporal trends:
    - Cumulative rates: overall adoption rate for county/subcounty/ward
    - Recent rates: rolling average (last 20 events)
    - Trend: recent - cumulative (momentum indicator)
    - EWM rates: exponential weighted moving average
    
    Captures whether adoption is accelerating/decelerating in each region.
    """

    p = prior_df.copy()
    m = main_df.copy()
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    for geo_col in ["county", "subcounty", "ward"]:
        p_sorted = p.sort_values([geo_col, time_col]).reset_index(drop=True)
        grp = p_sorted.groupby(geo_col, sort=False)
        p_sorted[f"{geo_col}_cum_cnt"] = grp.cumcount() + 1
        for w in ["90","120"]:
            col = f"adopted_within_{w}_days"
            p_sorted[f"{geo_col}_cum_adopt_{w}"] = grp[col].cumsum()
            p_sorted[f"{geo_col}_cum_rate_{w}"] = (
                p_sorted[f"{geo_col}_cum_adopt_{w}"] / p_sorted[f"{geo_col}_cum_cnt"])
            p_sorted[f"{geo_col}_recent_rate_{w}"] = grp[col].transform(
                lambda x: x.rolling(20, min_periods=3).mean()
            ).fillna(p[col].mean())
            p_sorted[f"{geo_col}_trend_{w}"] = (
                p_sorted[f"{geo_col}_recent_rate_{w}"] - p_sorted[f"{geo_col}_cum_rate_{w}"])
            p_sorted[f"{geo_col}_ewm_rate_{w}"] = grp[col].transform(
                lambda x: x.ewm(halflife=15, min_periods=3).mean()
            ).fillna(p[col].mean())
        keep = [geo_col, time_col, f"{geo_col}_cum_cnt"]
        for w in ["90","120"]:
            keep += [f"{geo_col}_cum_rate_{w}", f"{geo_col}_recent_rate_{w}",
                     f"{geo_col}_trend_{w}", f"{geo_col}_ewm_rate_{w}"]
        p_feat = p_sorted[keep].copy()
        p_feat = p_feat.sort_values([time_col, geo_col]).reset_index(drop=True)
        m = m.sort_values([time_col, geo_col]).reset_index(drop=True)
        m = pd.merge_asof(m, p_feat, on=time_col, by=geo_col,
                           direction="backward", allow_exact_matches=False)
        for c in keep:
            if c in m.columns and c not in [geo_col, time_col]:
                m[c] = m[c].fillna(0)
    return m.sort_values("ID").reset_index(drop=True)

def add_trainer_effectiveness(prior_df, main_df, time_col="training_dt"):

    """
    Calculate trainer-level effectiveness metrics:
    - Session count: how many trainings has this trainer conducted
    - Adoption rates: trainer's historical success rate (Bayesian smoothed)
    - Recent rates: rolling average (are they improving/declining?)
    - Recency: time since trainer's last session
    
    High-quality trainers should have higher adoption rates.
    """

    p = prior_df.copy()
    m = main_df.copy()
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    p = p.sort_values(["trainer_first", time_col]).reset_index(drop=True)
    grp = p.groupby("trainer_first", sort=False)
    p["trainer_session_cnt"] = grp.cumcount() + 1
    for w in ["07","90","120"]:
        col = f"adopted_within_{w}_days"
        p[f"trainer_adopt_{w}_sum"] = grp[col].cumsum()
        p[f"trainer_adopt_{w}_rate"] = p[f"trainer_adopt_{w}_sum"] / p["trainer_session_cnt"]
    for w in ["90","120"]:
        col = f"adopted_within_{w}_days"
        p[f"trainer_recent_{w}"] = grp[col].transform(
            lambda x: x.rolling(30, min_periods=5).mean())
        p[f"trainer_recent_{w}"] = p[f"trainer_recent_{w}"].fillna(p[f"trainer_adopt_{w}_rate"])
    p["trainer_last_dt"] = p[time_col]
    keep = ["trainer_first", time_col, "trainer_session_cnt", "trainer_last_dt"]
    for w in ["07","90","120"]:
        keep += [f"trainer_adopt_{w}_rate"]
    for w in ["90","120"]:
        keep += [f"trainer_recent_{w}"]
    p_feat = p[keep].copy()
    p_feat = p_feat.sort_values([time_col, "trainer_first"]).reset_index(drop=True)
    m = m.sort_values([time_col, "trainer_first"]).reset_index(drop=True)
    m = pd.merge_asof(m, p_feat, on=time_col, by="trainer_first",
                       direction="backward", allow_exact_matches=False)
    g07=p["adopted_within_07_days"].mean(); g90=p["adopted_within_90_days"].mean(); g120=p["adopted_within_120_days"].mean()
    m["trainer_session_cnt"] = m["trainer_session_cnt"].fillna(0)
    m["trainer_adopt_07_rate"] = m["trainer_adopt_07_rate"].fillna(g07)
    m["trainer_adopt_90_rate"] = m["trainer_adopt_90_rate"].fillna(g90)
    m["trainer_adopt_120_rate"] = m["trainer_adopt_120_rate"].fillna(g120)
    for w in ["90","120"]:
        gm = g90 if w == "90" else g120
        m[f"trainer_recent_{w}"] = m[f"trainer_recent_{w}"].fillna(gm)
    m["trainer_secs_since"] = (m[time_col] - m["trainer_last_dt"]).dt.total_seconds()
    m["trainer_secs_since"] = m["trainer_secs_since"].fillna(1e12).clip(lower=0)
    m.drop(columns=["trainer_last_dt"], inplace=True, errors="ignore")
    return m.sort_values("ID").reset_index(drop=True)

def add_peer_pressure_features(prior_df, main_df, time_col="training_dt"):
    """
    calculate peer influence features within farmer groups:
    - Peer adopters: how many OTHER farmers in group have adopted
    - Peer count: total other farmers in group
    - Peer adopt ratio: fraction of peers who adopted
    - Group intensity: total adoptions / group size
    
    Hypothesis: farmers are more likely to adopt when peers have adopted.
    """
    p = prior_df.copy()
    m = main_df.copy()
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    p = p.sort_values(["group_name", time_col]).reset_index(drop=True)
    group_stats = []
    for g, gdf in p.groupby("group_name", sort=False):
        gdf = gdf.sort_values(time_col).reset_index(drop=True)
        farmer_adopted_120 = defaultdict(int)
        all_farmers = set()
        n_a120 = []; n_f = []; cum120 = 0
        for _, row in gdf.iterrows():
            n_a120.append(len([f for f in farmer_adopted_120 if farmer_adopted_120[f] > 0 and f != row["farmer_name"]]))
            n_f.append(len(all_farmers - {row["farmer_name"]}) if row["farmer_name"] in all_farmers else len(all_farmers))
            all_farmers.add(row["farmer_name"])
            if row["adopted_within_120_days"] == 1:
                farmer_adopted_120[row["farmer_name"]] += 1; cum120 += 1
        gdf["peer_adopters_120"] = n_a120
        gdf["peer_farmers_count"] = n_f
        gdf["group_total_adopt_120"] = [0] + list(np.cumsum([row["adopted_within_120_days"] for _, row in gdf.iterrows()])[:-1])
        group_stats.append(gdf[["group_name", time_col, "peer_adopters_120",
                                 "peer_farmers_count", "group_total_adopt_120"]])
    p_feat = pd.concat(group_stats, ignore_index=True)
    p_feat["peer_adopt_ratio_120"] = p_feat["peer_adopters_120"] / p_feat["peer_farmers_count"].clip(lower=1)
    p_feat["group_adopt_intensity"] = p_feat["group_total_adopt_120"] / p_feat["peer_farmers_count"].clip(lower=1)
    p_feat = p_feat.sort_values([time_col, "group_name"]).reset_index(drop=True)
    m = m.sort_values([time_col, "group_name"]).reset_index(drop=True)
    m = pd.merge_asof(m, p_feat, on=time_col, by="group_name",
                       direction="backward", allow_exact_matches=False)
    for c in ["peer_adopters_120","peer_farmers_count","peer_adopt_ratio_120","group_adopt_intensity","group_total_adopt_120"]:
        if c in m.columns: m[c] = m[c].fillna(0)
    return m.sort_values("ID").reset_index(drop=True)

def add_group_ward_recency(prior_df, main_df, time_col="training_dt"):
    """
    Calculate time since last adoption in group/ward:
    - Find most recent adoption event in group
    - Find most recent adoption event in ward
    - Calculate seconds elapsed (recency)
    - Log-transform for better scale    
    """
    p = prior_df.copy()
    m = main_df.copy()
    for t in ["adopted_within_120_days","adopted_within_90_days"]:
        if t in p.columns: p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else: p[t] = 0
    for key_col in ["group_name", "ward"]:
        for w in ["90","120"]:
            col = f"adopted_within_{w}_days"
            p_pos = p.loc[p[col]==1, [key_col, time_col]].copy()
            p_pos = p_pos.rename(columns={time_col: f"last_{key_col}_adopt_{w}_dt"})
            p_pos = p_pos.sort_values([f"last_{key_col}_adopt_{w}_dt", key_col]).reset_index(drop=True)

            # Asof merge: find most recent adoption in group/ward
            m = m.sort_values([time_col, key_col]).reset_index(drop=True)
            m = pd.merge_asof(m, p_pos, left_on=time_col,
                               right_on=f"last_{key_col}_adopt_{w}_dt",
                               by=key_col, direction="backward", allow_exact_matches=False)
   
            # Calculate seconds since last adoption
            m[f"secs_since_{key_col}_adopt_{w}"] = (
                m[time_col] - m[f"last_{key_col}_adopt_{w}_dt"]
            ).dt.total_seconds()
            m[f"secs_since_{key_col}_adopt_{w}"] = m[f"secs_since_{key_col}_adopt_{w}"].fillna(1e12).clip(lower=0)
            m[f"log_secs_since_{key_col}_adopt_{w}"] = np.log1p(m[f"secs_since_{key_col}_adopt_{w}"])
            m.drop(columns=[f"last_{key_col}_adopt_{w}_dt"], inplace=True, errors="ignore")
    return m.sort_values("ID").reset_index(drop=True)

def add_trainer_workload(prior_df, main_df, time_col="training_dt"):
    """
    Calculate trainer workload (sessions conducted in recent window):
    - Count sessions in last 7 days
    - Count sessions in last 30 days
    """

    p = prior_df.copy()
    m = main_df.copy()
    p = p.sort_values(["trainer_first", time_col]).reset_index(drop=True)
    day_ns = 24*3600*1_000_000_000
    trainer_groups = {g: df for g, df in p.groupby("trainer_first", sort=False)}
    for window_days, suffix in [(7, "7d"), (30, "30d")]:
        col_name = f"trainer_workload_{suffix}"
        m[col_name] = 0
        m = m.sort_values([time_col, "trainer_first"]).reset_index(drop=True)
        for g, m_g in m.groupby("trainer_first", sort=False):
            if g not in trainer_groups: continue
            p_g = trainer_groups[g]
            # Binary search for events in window
            p_ts = p_g[time_col].astype("datetime64[ns]").values.astype(np.int64)
            m_ts = m_g[time_col].astype("datetime64[ns]").values.astype(np.int64)
            right = np.searchsorted(p_ts, m_ts, side="left")
            left = np.searchsorted(p_ts, m_ts - window_days * day_ns, side="left")
            m.loc[m_g.index, col_name] = right - left
    return m.sort_values("ID").reset_index(drop=True)

def stability_feature_selection(X_train, y, cat_cols, target_n=180, n_rounds=5):
    """
    Stability-based feature selection using LightGBM:
    - Train multiple models with different seeds
    - Average feature importance across models
    - Select top-N features + keep all categorical features
    """
    import lightgbm as lgb
    print(f"\n  STABILITY FEATURE SELECTION: {X_train.shape[1]} ---> {target_n}")

    # Remove text columns ( because can't be used in LightGBM importance)
    text_skip = {"topics_text"}
    use_cols = [c for c in X_train.columns if c not in text_skip]
    cat_lgb = [c for c in cat_cols if c in use_cols]
    X = X_train[use_cols].copy()
    for c in cat_lgb: X[c] = X[c].astype("category")
    importance_sum = np.zeros(len(use_cols))
    for seed in [42, 123, 777, 2024, 9999]:
        params = {
            "objective": "binary", "metric": "auc", "boosting": "gbdt",
            "learning_rate": 0.08, "num_leaves": 63, "max_depth": 6,
            "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.5,
            "reg_lambda": 10.0, "verbose": -1, "seed": seed, "n_jobs": -1,
        }
        dtrain = lgb.Dataset(X, y, categorical_feature=cat_lgb)
        m = lgb.train(params, dtrain, num_boost_round=300)
        importance_sum += m.feature_importance("gain")
    imp = pd.DataFrame({
        "feature": use_cols,
        "importance": importance_sum / n_rounds
    }).sort_values("importance", ascending=False)
    keep = set(imp.head(target_n)["feature"].tolist())
    keep.update([c for c in cat_cols if c in use_cols])
    keep = [c for c in X_train.columns if c in keep]
    print(f"  Selected {len(keep)} features (top {target_n} + categoricals)")
    print(f"  Top 20: {imp.head(20)['feature'].tolist()}")
    return keep


def add_test_era_features(prior, train_feat, test_feat, time_col="training_dt"):
    """
    Extract features from test-era prior data (2025-05-01+):
    - Calculate base rates from test period 
    - County/trainer/ward/subcounty rates from test era
    - Test-weighted base rate (adjust for county distribution in test set)
    """
    TEST_ERA_START = pd.Timestamp("2025-05-01")
    
    prior_era = prior[prior["training_day"] >= TEST_ERA_START].copy()
    print(f"Prior events in test era: {len(prior_era)} | "
          f"dates: {prior_era['training_day'].min()} → {prior_era['training_day'].max()}")
    
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        prior_era[t] = pd.to_numeric(prior_era[t], errors="coerce").fillna(0).astype(int)

    # Calculate test-era base rates
    era_base = {
        "07": prior_era["adopted_within_07_days"].mean(),
        "90": prior_era["adopted_within_90_days"].mean(),
        "120": prior_era["adopted_within_120_days"].mean(),
    }
    print(f"\nPRIOR TEST-ERA BASE RATES (ground truth for test period):")
    for w, r in era_base.items(): print(f"  {w}d: {r:.6f}")
    
    # County-level rates from test era
    era_county_rates = prior_era.groupby("county")[
        ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]
    ].agg(["mean","count"]).reset_index()
    era_county_rates.columns = ["county"] + [
        f"era_county_{w}_{s}" 
        for w in ["07","90","120"] for s in ["rate","n"]
    ]
    
    # Trainer-level rates from test era
    era_trainer_rates = prior_era.groupby("trainer_first")[
        ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]
    ].agg(["mean","count"]).reset_index()
    era_trainer_rates.columns = ["trainer_first"] + [
        f"era_trainer_{w}_{s}" 
        for w in ["07","90","120"] for s in ["rate","n"]
    ]
    
    # Ward-level rates from test era
    era_ward_rates = prior_era.groupby("ward")[
        ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]
    ].agg(["mean","count"]).reset_index()
    era_ward_rates.columns = ["ward"] + [
        f"era_ward_{w}_{s}" 
        for w in ["07","90","120"] for s in ["rate","n"]
    ]
    
    # Subcounty-level rates
    era_sub_rates = prior_era.groupby("subcounty")[
        ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]
    ].agg(["mean","count"]).reset_index()
    era_sub_rates.columns = ["subcounty"] + [
        f"era_sub_{w}_{s}" 
        for w in ["07","90","120"] for s in ["rate","n"]
    ]
    # Merge era features to train and test
    train_feat = train_feat.merge(era_county_rates, on="county", how="left")
    test_feat  = test_feat.merge(era_county_rates,  on="county", how="left")

    train_feat = train_feat.merge(era_trainer_rates, on="trainer_first", how="left")
    test_feat  = test_feat.merge(era_trainer_rates,  on="trainer_first", how="left")

    train_feat = train_feat.merge(era_ward_rates, on="ward", how="left")
    test_feat  = test_feat.merge(era_ward_rates,  on="ward", how="left")

    train_feat = train_feat.merge(era_sub_rates, on="subcounty", how="left")
    test_feat  = test_feat.merge(era_sub_rates,  on="subcounty", how="left")
    
    # Fill missing with global era means
    for df in [train_feat, test_feat]:
        for w in ["07","90","120"]:
            for grp in ["county","trainer","ward","sub"]:
                col = f"era_{grp}_{w}_rate"
                if col in df.columns:
                    df[col] = df[col].fillna(era_base[w])
                    df[f"era_{grp}_{w}_n"] = df.get(f"era_{grp}_{w}_n", 0).fillna(0)
    
    # Test-era base rate weighted by TEST county distribution
    test_county_dist = test_feat["county"].value_counts(normalize=True)
    test_weighted_rate = {}
    for w in ["07","90","120"]:
        r = 0.0
        for c, share in test_county_dist.items():
            row = era_county_rates[era_county_rates["county"] == c]
            if len(row) > 0:
                r += share * float(row[f"era_county_{w}_rate"].iloc[0])
            else:
                r += share * era_base[w]
        test_weighted_rate[w] = r
    
    print(f"\nTEST-WEIGHTED BASE RATES (county-dist-adjusted):")
    for w, r in test_weighted_rate.items(): print(f"  {w}d: {r:.6f}")
    
    return train_feat, test_feat, era_base, test_weighted_rate

def add_interaction_features(df, prior_df):
    """
    Create interaction features (target encoding on feature combinations):
    - Registration × County: adoption rate by registration type in each county
    - Cooperative × County: coop members' adoption rate in each county
    - Has_topic × County: sessions with topics, by county (very powerful per EDA)
    - Age × County: adoption rates by age group in each county
    
    Captures local
    """

    df = df.copy()
    
    # Registration × county interaction rate from prior
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        prior_df[t] = pd.to_numeric(prior_df[t], errors="coerce").fillna(0).astype(int)
    
    g07 = prior_df["adopted_within_07_days"].mean()
    g90 = prior_df["adopted_within_90_days"].mean()
    g120 = prior_df["adopted_within_120_days"].mean()
    alpha = 50.0
    
    # Registration × county TE
    reg_county = prior_df.groupby(["registration","county"]).agg(
        n=("adopted_within_07_days","count"),
        sum07=("adopted_within_07_days","sum"),
        sum90=("adopted_within_90_days","sum"),
        sum120=("adopted_within_120_days","sum"),
    ).reset_index()
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        reg_county[f"regcounty_rate{w}"] = (
            (reg_county[f"sum{w}"] + alpha*gm) / (reg_county["n"] + alpha)
        )
    df = df.merge(
        reg_county[["registration","county","regcounty_rate07","regcounty_rate90","regcounty_rate120"]],
        on=["registration","county"], how="left"
    )
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        df[f"regcounty_rate{w}"] = df[f"regcounty_rate{w}"].fillna(gm)
    
    # Cooperative × county TE
    coop_county = prior_df.groupby(["belong_to_cooperative","county"]).agg(
        n=("adopted_within_07_days","count"),
        sum07=("adopted_within_07_days","sum"),
        sum90=("adopted_within_90_days","sum"),
        sum120=("adopted_within_120_days","sum"),
    ).reset_index()
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        coop_county[f"coopcounty_rate{w}"] = (
            (coop_county[f"sum{w}"] + alpha*gm) / (coop_county["n"] + alpha)
        )
    df = df.merge(
        coop_county[["belong_to_cooperative","county",
                     "coopcounty_rate07","coopcounty_rate90","coopcounty_rate120"]],
        on=["belong_to_cooperative","county"], how="left"
    )
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        df[f"coopcounty_rate{w}"] = df[f"coopcounty_rate{w}"].fillna(gm)
    
    # has_topic × county TE (very powerful per EDA)
    topic_county = prior_df.groupby(["has_topic_trained_on","county"]).agg(
        n=("adopted_within_07_days","count"),
        sum07=("adopted_within_07_days","sum"),
        sum90=("adopted_within_90_days","sum"),
        sum120=("adopted_within_120_days","sum"),
    ).reset_index()
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        topic_county[f"topiccounty_rate{w}"] = (
            (topic_county[f"sum{w}"] + alpha*gm) / (topic_county["n"] + alpha)
        )
    df = df.merge(
        topic_county[["has_topic_trained_on","county",
                      "topiccounty_rate07","topiccounty_rate90","topiccounty_rate120"]],
        on=["has_topic_trained_on","county"], how="left"
    )
    for w, gm in [("07",g07),("90",g90),("120",g120)]:
        df[f"topiccounty_rate{w}"] = df[f"topiccounty_rate{w}"].fillna(gm)
    
    # Age × county TE (Below 35 has 2x adoption for Aug-Nov 2024 period)
    age_county = prior_df.groupby(["age","county"]).agg(
        n=("adopted_within_07_days","count"),
        sum07=("adopted_within_07_days","sum"),
        sum120=("adopted_within_120_days","sum"),
    ).reset_index()
    for w, gm in [("07",g07),("120",g120)]:
        age_county[f"agecounty_rate{w}"] = (
            (age_county[f"sum{w}"] + alpha*gm) / (age_county["n"] + alpha)
        )
    df = df.merge(
        age_county[["age","county","agecounty_rate07","agecounty_rate120"]],
        on=["age","county"], how="left"
    )
    for w, gm in [("07",g07),("120",g120)]:
        df[f"agecounty_rate{w}"] = df[f"agecounty_rate{w}"].fillna(gm)
    
    return df
def add_recency_weighted_hist_rates(prior_df, main_df, key, 
                                     time_col="training_dt", halflife_days=180):

    """
    Recency-weighted historical rates (exponential decay):
    - Weight recent events more heavily than old events
    - Bayesian smoothing applied to weighted sum
    
    Captures that recent patterns matter more than distant past...
    """

    p = prior_df.copy()
    for t in ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]:
        if t in p.columns:
            p[t] = pd.to_numeric(p[t], errors="coerce").fillna(0).astype(int)
        else:
            p[t] = 0
    
    if key not in p.columns: 
        return main_df
    
    g07 = p["adopted_within_07_days"].mean()
    g90 = p["adopted_within_90_days"].mean()
    g120 = p["adopted_within_120_days"].mean()
    alpha = 60.0
    
    p = p.sort_values([key, time_col]).reset_index(drop=True)
    
    max_date = p[time_col].max()
    decay_factor = np.exp(-np.log(2) / (halflife_days * 86400))
    
    results = []
    for group_key, gdf in p.groupby(key, sort=False):
        gdf = gdf.sort_values(time_col).reset_index(drop=True)
        ts = gdf[time_col].values.astype("datetime64[ns]").astype(np.int64)
        
        for i in range(len(gdf)):
            t_i = ts[i]
            # Exponential weight: w_j = decay^((t_i - t_j) / 1e9 / 86400)
            deltas_sec = (t_i - ts[:i]) / 1e9  # seconds
            weights = decay_factor ** (deltas_sec / 86400.0)
            
            wsum = weights.sum() if len(weights) > 0 else 0
            for w, gm in [("07",g07),("90",g90),("120",g120)]:
                col = f"adopted_within_{w}_days"
                vals = gdf[col].values[:i]
                w_adopt = (vals * weights).sum() if len(vals) > 0 else 0
                results.append({
                    key: group_key,
                    time_col: gdf[time_col].iloc[i],
                    "ID": gdf["ID"].iloc[i] if "ID" in gdf.columns else i,
                    f"{key}_rw_rate{w}": (w_adopt + alpha * gm) / (wsum + alpha),
                    f"{key}_rw_cnt": wsum,
                })
    
    result_df = pd.DataFrame(results)

    feat_cols = [c for c in result_df.columns if "rw_" in c]
    p_feat = result_df[[key, time_col] + feat_cols].copy()
    p_feat = p_feat.sort_values([time_col, key]).reset_index(drop=True)
    
    m = main_df.copy().sort_values([time_col, key]).reset_index(drop=True)
    m = pd.merge_asof(m, p_feat, on=time_col, by=key, 
                       direction="backward", allow_exact_matches=False)
    for c in feat_cols:
        if c in m.columns:
            gm_fill = {"07": g07, "90": g90, "120": g120}
            w = c.split("rate")[-1] if "rate" in c else None
            m[c] = m[c].fillna(gm_fill.get(w, 0) if w else 0)
    
    return m.sort_values("ID").reset_index(drop=True)


# ============================================================
# CALIBRATION 
# ============================================================
def temperature_scale_fit(p, y):
    p = np.clip(p, EPS, 1-EPS)
    def nll(temp): #Negative log likelihood with temperature scaling
        logits = np.log(p/(1-p))
        scaled = 1/(1+np.exp(-logits/temp[0]))
        scaled = np.clip(scaled, EPS, 1-EPS)
        return log_loss(y, scaled)
    result = minimize(nll, [1.5], method='Nelder-Mead', options={'maxiter': 5000})
    return float(result.x[0])

def temperature_scale_apply(temp, p):
    #Apply fitted temperature to probabilities.
    p = np.clip(p, EPS, 1-EPS)
    logits = np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-logits/temp)), EPS, 1-EPS)

def logit_shift_to_mean(preds, target_mean):
    """Shift predictions in logit space to match a target mean"""
    p = np.clip(preds, EPS, 1-EPS)
    logits = np.log(p / (1 - p))

    def obj(shift):
        adjusted = expit(logits + shift[0])
        return (adjusted.mean() - target_mean) ** 2

    res = minimize(obj, [0.0], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-12})
    adjusted = expit(logits + float(res.x[0]))
    return np.clip(adjusted, EPS, 1-EPS)


def cv_platt_scaling(oof_p, y, test_p, n_splits=5, seed=42):
    """Cross-validated Platt scaling """
    oof_c = np.clip(oof_p, EPS, 1-EPS)
    test_c = np.clip(test_p, EPS, 1-EPS)

    oof_logits = np.log(oof_c / (1 - oof_c)).reshape(-1, 1)
    test_logits = np.log(test_c / (1 - test_c)).reshape(-1, 1)

    oof_cal = np.zeros_like(oof_p)
    test_cal_sum = np.zeros_like(test_p)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(oof_logits, y):
        lr = LogisticRegression(solver="lbfgs", max_iter=20000, C=1.0)
        lr.fit(oof_logits[tr_idx], y[tr_idx])
        oof_cal[va_idx] = lr.predict_proba(oof_logits[va_idx])[:, 1]
        test_cal_sum += lr.predict_proba(test_logits)[:, 1] / n_splits

    return np.clip(oof_cal, EPS, 1-EPS), np.clip(test_cal_sum, EPS, 1-EPS)


def cv_beta_scaling(oof_p, y, test_p, n_splits=5, seed=42):
    """Cross-validated Beta calibration."""
    oof_c = np.clip(oof_p, EPS, 1-EPS)
    test_c = np.clip(test_p, EPS, 1-EPS)

    oof_feats = np.column_stack([np.log(oof_c), np.log(1-oof_c)])
    test_feats = np.column_stack([np.log(test_c), np.log(1-test_c)])

    oof_cal = np.zeros_like(oof_p)
    test_cal_sum = np.zeros_like(test_p)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(oof_feats, y):
        lr = LogisticRegression(solver="lbfgs", max_iter=20000, C=1.0)
        lr.fit(oof_feats[tr_idx], y[tr_idx])
        oof_cal[va_idx] = lr.predict_proba(oof_feats[va_idx])[:, 1]
        test_cal_sum += lr.predict_proba(test_feats)[:, 1] / n_splits

    return np.clip(oof_cal, EPS, 1-EPS), np.clip(test_cal_sum, EPS, 1-EPS)


def calibrate(oof_p, y, test_p, name="",
                  train_mask=None, target_base_rate=None):
 
    # Try multiple calibration methods

    print(f"\n  === Calibration for {name} ===")

    if train_mask is not None:
        oof_train = oof_p[train_mask]
        y_train = y[train_mask]
        print(f"  Train-only: {train_mask.sum()} samples, "
              f"base_rate={y_train.mean():.6f}, n_pos={y_train.sum()}")
    else:
        oof_train = oof_p
        y_train = y

    candidates = []

    # 1) No calibration baseline
    oof_c = np.clip(oof_p, EPS, 1-EPS)
    test_c = np.clip(test_p, EPS, 1-EPS)
    ll_base = log_loss(y_train, np.clip(oof_c[train_mask] if train_mask is not None else oof_c, EPS, 1-EPS))
    candidates.append(("none", ll_base, oof_c, test_c))

    # 2) Temperature scaling (fit on train-only)
    try:
        oof_train_c = np.clip(oof_train, EPS, 1-EPS)
        T = temperature_scale_fit(oof_train_c, y_train)
        oof_t = temperature_scale_apply(T, oof_c)
        te_t = temperature_scale_apply(T, test_c)
        check = oof_t[train_mask] if train_mask is not None else oof_t
        ll_t = log_loss(y_train, np.clip(check, EPS, 1-EPS))
        candidates.append((f"temp(T={T:.3f})", ll_t, oof_t, te_t))
    except Exception as e:
        print(f"  temp failed: {e}")

    # 3) CV Platt scaling (train-only, cross-validated)
    try:
        oof_train_c = np.clip(oof_train, EPS, 1-EPS)
        # Fit CV Platt on train-only portion
        oof_platt_train, _ = cv_platt_scaling(oof_train_c, y_train, test_c, n_splits=5, seed=42)

        # Also fit global Platt on all train-only for test prediction
        oof_train_logits = logit(oof_train_c).reshape(-1, 1)
        test_logits = logit(test_c).reshape(-1, 1)
        lr_global = LogisticRegression(solver="lbfgs", max_iter=20000, C=1.0)
        lr_global.fit(oof_train_logits, y_train)
        te_platt = np.clip(lr_global.predict_proba(test_logits)[:, 1], EPS, 1-EPS)

        # For OOF: use CV predictions for train, global for prior
        oof_platt_full = np.clip(lr_global.predict_proba(logit(oof_c).reshape(-1, 1))[:, 1], EPS, 1-EPS)
        if train_mask is not None:
            oof_platt_full[train_mask] = oof_platt_train

        ll_pl = log_loss(y_train, np.clip(oof_platt_train, EPS, 1-EPS))
        candidates.append(("cv_platt", ll_pl, oof_platt_full, te_platt))
    except Exception as e:
        print(f"  cv_platt failed: {e}")

    # 4) CV Beta scaling (train-only, cross-validated)
    try:
        oof_train_c = np.clip(oof_train, EPS, 1-EPS)
        oof_beta_train, _ = cv_beta_scaling(oof_train_c, y_train, test_c, n_splits=5, seed=42)

        oof_train_feats = np.column_stack([np.log(oof_train_c), np.log(1-oof_train_c)])
        all_feats = np.column_stack([np.log(oof_c), np.log(1-oof_c)])
        test_feats = np.column_stack([np.log(test_c), np.log(1-test_c)])
        lr_beta = LogisticRegression(solver="lbfgs", max_iter=20000, C=1.0)
        lr_beta.fit(oof_train_feats, y_train)
        te_beta = np.clip(lr_beta.predict_proba(test_feats)[:, 1], EPS, 1-EPS)

        oof_beta_full = np.clip(lr_beta.predict_proba(all_feats)[:, 1], EPS, 1-EPS)
        if train_mask is not None:
            oof_beta_full[train_mask] = oof_beta_train

        ll_beta = log_loss(y_train, np.clip(oof_beta_train, EPS, 1-EPS))
        candidates.append(("cv_beta", ll_beta, oof_beta_full, te_beta))
    except Exception as e:
        print(f"  cv_beta failed: {e}")

    # 5) Ensemble of smooth calibrators (avg of available)
    try:
        smooth_cands = [(n, ll, o, t) for n, ll, o, t in candidates if n != "none"]
        if len(smooth_cands) >= 2:
            te_ens = np.mean([t for _, _, _, t in smooth_cands], axis=0)
            oof_ens = np.mean([o for _, _, o, _ in smooth_cands], axis=0)
            check = oof_ens[train_mask] if train_mask is not None else oof_ens
            ll_ens = log_loss(y_train, np.clip(check, EPS, 1-EPS))
            candidates.append(("smooth_ensemble", ll_ens,
                              np.clip(oof_ens, EPS, 1-EPS),
                              np.clip(te_ens, EPS, 1-EPS)))
    except Exception as e:
        print(f"  smooth_ensemble failed: {e}")

    # Pick best
    best = min(candidates, key=lambda x: x[1])
    method, ll, oof_best, te_best = best
    print(f"  Best calibration: {method}, LL={ll:.6f}")
    for nm, ll_c, _, _ in candidates:
        print(f"    {nm:25s}: LL={ll_c:.6f}")

    # Logit-space base rate alignment
    if target_base_rate is not None:
        before_mean = te_best.mean()
        te_best = logit_shift_to_mean(te_best, target_base_rate)
        print(f"  Base rate alignment: {before_mean:.6f} → {te_best.mean():.6f} (target={target_base_rate:.6f})")

    return np.clip(te_best, EPS, 1-EPS), oof_best


# ============================================================
# MONOTONE PROJECTION
# ============================================================
def pav_3(p7, p90, p120):
    """
    Pool Adjacent Violators (PAV) for 3 values:
    - Enforce monotonicity: p7 <= p90 <= p120
    - Average adjacent violators iteratively
    
    Ensures logical consistency (longer window = higher probability).
    """
    a, b, c = float(p7), float(p90), float(p120)
    if a > b: ab = 0.5*(a+b); a, b = ab, ab
    if b > c: bc = 0.5*(b+c); b, c = bc, bc
    if a > b: abc = (a+b+c)/3.0; a, b, c = abc, abc, abc
    return a, b, c

def enforce_monotone_arrays(p7, p90, p120):
    p7, p90, p120 = np.clip(p7,0,1), np.clip(p90,0,1), np.clip(p120,0,1)
    out7, out90, out120 = np.empty_like(p7), np.empty_like(p90), np.empty_like(p120)
    for i in range(len(p7)):
        a, b, c = pav_3(p7[i], p90[i], p120[i])
        out7[i], out90[i], out120[i] = a, b, c
    return out7, out90, out120

# ============================================================
# MODEL TRAINING
# ============================================================
def get_sample_weights(training_days, decay=0.3):
    #Calculate temporal sample weights with exponential decay
    max_day = training_days.max()
    days_ago = (max_day - training_days).dt.total_seconds() / (24*3600)
    weights = np.exp(-decay * days_ago / 365.0)
    weights = weights / weights.mean()
    return weights.values

#Train CatBoost with K-fold CV:
def train_catboost_binary_cv(X_train, y, X_test, cat_cols, text_cols,
                              sample_weight=None, seed=42, use_gpu=False):
    from catboost import CatBoostClassifier, Pool
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    text_idx = [X_train.columns.get_loc(c) for c in text_cols] if text_cols else []
    oof = np.zeros(len(X_train))
    te = np.zeros(len(X_test))
    pos, neg = float(y.sum()), float(len(y)-y.sum())
    spw = neg / max(pos, 1.0)
    params = dict(
        loss_function="Logloss", eval_metric="Logloss",
        iterations=10000, learning_rate=0.03, depth=7, l2_leaf_reg=15.0,
        bootstrap_type="MVS", subsample=0.75, random_strength=4.0,
        min_data_in_leaf=40, scale_pos_weight=spw,
        verbose=500, od_type="Iter", od_wait=500,
    )
    if use_gpu and not text_idx:
        params["task_type"] = "GPU"
        params["devices"] = "0"
        params["bootstrap_type"] = "Poisson"
        del params["subsample"]
    for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
        sw = sample_weight[tr] if sample_weight is not None else None
        m = CatBoostClassifier(**params, random_seed=seed+fold)
        pool_tr = Pool(X_train.iloc[tr], y[tr], cat_features=cat_idx,
                       text_features=text_idx if text_idx else None, weight=sw)
        pool_va = Pool(X_train.iloc[va], y[va], cat_features=cat_idx,
                       text_features=text_idx if text_idx else None)
        m.fit(pool_tr, eval_set=pool_va, use_best_model=True)
        oof[va] = m.predict_proba(X_train.iloc[va])[:,1]
        te += m.predict_proba(X_test)[:,1] / N_SPLITS
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, EPS, 1-EPS))
    print(f"  CB OOF: AUC={auc:.6f}, LL={ll:.6f}")
    return np.clip(oof, EPS, 1-EPS), np.clip(te, EPS, 1-EPS)

def train_lgbm_binary_cv(X_train, y, X_test, cat_cols, sample_weight=None, seed=42,
                          num_leaves=63, lr=0.03, min_child=40):
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))
    te = np.zeros(len(X_test))
    pos, neg = float(y.sum()), float(len(y)-y.sum())
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting": "gbdt",
        "learning_rate": lr, "num_leaves": num_leaves, "max_depth": 7,
        "min_child_samples": min_child, "subsample": 0.75, "colsample_bytree": 0.7,
        "reg_alpha": 0.5, "reg_lambda": 15.0,
        "scale_pos_weight": neg/max(pos,1),
        "verbose": -1, "seed": seed, "n_jobs": -1,
    }
    text_cols_set = {"topics_text"}
    use_cols = [c for c in X_train.columns if c not in text_cols_set]
    cat_lgb = [c for c in cat_cols if c in use_cols]
    X_tr_lgb = X_train[use_cols].copy()
    X_te_lgb = X_test[use_cols].copy()
    for c in cat_lgb:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_te_lgb[c] = X_te_lgb[c].astype("category")
    for fold, (tr, va) in enumerate(skf.split(X_tr_lgb, y), 1):
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        sw_va = sample_weight[va] if sample_weight is not None else None
        dtrain = lgb.Dataset(X_tr_lgb.iloc[tr], y[tr], weight=sw_tr, categorical_feature=cat_lgb)
        dval = lgb.Dataset(X_tr_lgb.iloc[va], y[va], weight=sw_va, categorical_feature=cat_lgb, reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(500), lgb.log_evaluation(500)])
        oof[va] = m.predict(X_tr_lgb.iloc[va])
        te += m.predict(X_te_lgb) / N_SPLITS
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, EPS, 1-EPS))
    print(f"  LGB OOF: AUC={auc:.6f}, LL={ll:.6f}")
    return np.clip(oof, EPS, 1-EPS), np.clip(te, EPS, 1-EPS)


#Train LightGBM with DART (Dropouts meet Multiple Additive Regression Trees)
def train_lgbm_dart_cv(X_train, y, X_test, cat_cols, sample_weight=None, seed=42):
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))
    te = np.zeros(len(X_test))
    pos, neg = float(y.sum()), float(len(y)-y.sum())
    params = {
        "objective": "binary", "metric": "binary_logloss", "boosting": "dart",
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 7,
        "min_child_samples": 50, "subsample": 0.75, "colsample_bytree": 0.7,
        "reg_alpha": 0.5, "reg_lambda": 15.0, "scale_pos_weight": neg/max(pos,1),
        "drop_rate": 0.1, "skip_drop": 0.5, "max_drop": 50,
        "verbose": -1, "seed": seed, "n_jobs": -1,
    }
    text_cols_set = {"topics_text"}
    use_cols = [c for c in X_train.columns if c not in text_cols_set]
    cat_lgb = [c for c in cat_cols if c in use_cols]
    X_tr = X_train[use_cols].copy()
    X_te = X_test[use_cols].copy()
    for c in cat_lgb:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = X_te[c].astype("category")
    for fold, (tr, va) in enumerate(skf.split(X_tr, y), 1):
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        dtrain = lgb.Dataset(X_tr.iloc[tr], y[tr], weight=sw_tr, categorical_feature=cat_lgb)
        dval = lgb.Dataset(X_tr.iloc[va], y[va], categorical_feature=cat_lgb, reference=dtrain)
        m = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(500), lgb.log_evaluation(500)])
        oof[va] = m.predict(X_tr.iloc[va])
        te += m.predict(X_te) / N_SPLITS
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, EPS, 1-EPS))
    print(f"  LGB-DART OOF: AUC={auc:.6f}, LL={ll:.6f}")
    return np.clip(oof, EPS, 1-EPS), np.clip(te, EPS, 1-EPS)


def train_xgb_binary_cv(X_train, y, X_test, cat_cols, sample_weight=None,seed=42, use_gpu=False):

    """
    Train XGBoost with K-fold CV:
    - Categorical encoding via LabelEncoder
    - Fallback to CPU if GPU fails
    """
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train))
    te = np.zeros(len(X_test))
    pos, neg = float(y.sum()), float(len(y)-y.sum())
    text_cols_set = {"topics_text"}
    use_cols = [c for c in X_train.columns if c not in text_cols_set]
    cat_xgb = [c for c in cat_cols if c in use_cols]
    X_tr_xgb = X_train[use_cols].copy()
    X_te_xgb = X_test[use_cols].copy()
    for c in cat_xgb:
        le = LabelEncoder()
        combined = pd.concat([X_tr_xgb[c].astype(str), X_te_xgb[c].astype(str)], ignore_index=True)
        le.fit(combined)
        X_tr_xgb[c] = le.transform(X_tr_xgb[c].astype(str))
        X_te_xgb[c] = le.transform(X_te_xgb[c].astype(str))
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "learning_rate": 0.03, "max_depth": 7,
        "min_child_weight": 40, "subsample": 0.75,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5, "reg_lambda": 15.0,
        "scale_pos_weight": neg / max(pos, 1),
        "seed": seed, "verbosity": 0, "nthread": -1,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    for fold, (tr, va) in enumerate(skf.split(X_tr_xgb, y), 1):
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        dtrain = xgb.DMatrix(X_tr_xgb.iloc[tr], y[tr], weight=sw_tr)
        dval = xgb.DMatrix(X_tr_xgb.iloc[va], y[va])
        try:
            m = xgb.train(params, dtrain, num_boost_round=10000,
                          evals=[(dval, "val")], early_stopping_rounds=500,
                          verbose_eval=500)
        except xgb.core.XGBoostError:
            params["tree_method"] = "hist"
            m = xgb.train(params, dtrain, num_boost_round=10000,
                          evals=[(dval, "val")], early_stopping_rounds=500,
                          verbose_eval=500)
        oof[va] = m.predict(dval)
        te += m.predict(xgb.DMatrix(X_te_xgb)) / N_SPLITS
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, EPS, 1-EPS))
    print(f"  XGB OOF: AUC={auc:.6f}, LL={ll:.6f}")
    return np.clip(oof, EPS, 1-EPS), np.clip(te, EPS, 1-EPS)

#Train meta-model for stacking (logistic regression on base model predictions)
def train_stacking_meta(oof_models_dict, y, test_models_dict, seed=42):
    oof_stack = np.column_stack(list(oof_models_dict.values()))
    te_stack = np.column_stack(list(test_models_dict.values()))
    for j in range(oof_stack.shape[1]):
        col_mean = np.nanmean(oof_stack[:, j])
        oof_stack[np.isnan(oof_stack[:, j]), j] = col_mean
        te_stack[np.isnan(te_stack[:, j]), j] = col_mean
    oof_meta = np.zeros(len(y))
    te_meta = np.zeros(te_stack.shape[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(oof_stack, y):
        lr = LogisticRegression(solver="lbfgs", max_iter=20000, C=0.5,
                                class_weight="balanced")
        lr.fit(oof_stack[tr_idx], y[tr_idx])
        oof_meta[va_idx] = lr.predict_proba(oof_stack[va_idx])[:,1]
        te_meta += lr.predict_proba(te_stack)[:,1] / 5
    print(f"  META OOF: AUC={roc_auc_score(y,oof_meta):.6f}, LL={log_loss(y,np.clip(oof_meta,EPS,1-EPS)):.6f}")
    return np.clip(oof_meta, EPS, 1-EPS), np.clip(te_meta, EPS, 1-EPS)

# ============================================================
#  ENSEMBLE - include ALL models for LL
# ============================================================
def optimize_logloss_weights_logit(oof_list, y, test_list, n_restarts=50, seed=42):
    #Blend in logit space with stable softmax.
    rng = np.random.default_rng(seed)
    n = len(oof_list)
    if n == 1:
        return np.clip(oof_list[0], EPS, 1-EPS), np.clip(test_list[0], EPS, 1-EPS)

    y = np.asarray(y)
    y_mean = float(np.mean(y))

    def clean_probs(arr):
        #Sanitize probabilities: handle NaN, clip to valid range
        a = np.asarray(arr, dtype=np.float64)
        a = np.where(np.isfinite(a), a, y_mean)
        a = np.clip(a, EPS, 1 - EPS)
        return a

    oof_clean = [clean_probs(o) for o in oof_list]
    te_clean  = [clean_probs(t) for t in test_list]

    oof_logits = [logit(o) for o in oof_clean]
    te_logits  = [logit(t) for t in te_clean]

    oof_logits_mat = np.vstack(oof_logits)
    te_logits_mat  = np.vstack(te_logits)

    def softmax_stable(w):
        #Numerically stable softmax: prevents overflow
        w = np.asarray(w, dtype=np.float64)
        m = np.nanmax(w)
        if not np.isfinite(m):
            return np.ones_like(w) / len(w)
        w = w - m
        lse = logsumexp(w)
        if not np.isfinite(lse):
            return np.ones_like(w) / len(w)
        out = np.exp(w - lse)
        out = np.where(np.isfinite(out), out, 0.0)
        s = out.sum()
        return out / s if s > 0 else (np.ones_like(w) / len(w))

    def neg_ll(w):
        #Negative log likelihood for optimization
        w_norm = softmax_stable(w)
        blend_logit = (w_norm[:, None] * oof_logits_mat).sum(axis=0)
        blend_prob = np.clip(expit(blend_logit), EPS, 1 - EPS)
        if not np.isfinite(blend_prob).all():
            return 1e9
        return log_loss(y, blend_prob)

    best_ll = np.inf
    best_w = np.ones(n) / n

    for _ in range(n_restarts):
        w0 = rng.normal(0, 0.3, size=n)
        res = minimize(neg_ll, w0, method="Nelder-Mead", options={"maxiter": 15000})
        if np.isfinite(res.fun) and res.fun < best_ll:
            best_ll = float(res.fun)
            best_w = softmax_stable(res.x)

    print(f"  LL weights (logit-space): {np.round(best_w, 4)}, OOF LL={best_ll:.6f}")

    blend_logit_oof = (best_w[:, None] * oof_logits_mat).sum(axis=0)
    blend_logit_te  = (best_w[:, None] * te_logits_mat).sum(axis=0)
    oof_blend = np.clip(expit(blend_logit_oof), EPS, 1 - EPS)
    te_blend  = np.clip(expit(blend_logit_te),  EPS, 1 - EPS)

    return oof_blend, te_blend

#Optimize ensemble weights for AUC (random search with Dirichlet)
def optimize_auc_weights(oof_list, y, test_list):
    n = len(oof_list)
    best_w = np.ones(n) / n
    best_auc = roc_auc_score(y, sum(best_w[i]*oof_list[i] for i in range(n)))
    for _ in range(8000):
        w = np.random.dirichlet(np.ones(n))
        blend = sum(w[i]*oof_list[i] for i in range(n))
        auc = roc_auc_score(y, blend)
        if auc > best_auc: best_auc, best_w = auc, w
    te_out = sum(best_w[i]*test_list[i] for i in range(n))
    print(f"  AUC weights: {best_w.round(3)}, AUC={best_auc:.6f}")
    return np.clip(te_out, EPS, 1-EPS), best_auc




def train_tabnet_binary_cv(X_train, y, X_test, cat_cols, sample_weight=None, seed=42, target_name="07"):

    """
    Train TabNet with 5-fold CV for a single target
    """
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    from sklearn.preprocessing import LabelEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    # Clean data
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    text_cols_to_drop = [
        c for c in X_train_clean.columns
        if X_train_clean[c].dtype == "object" and c not in cat_cols
    ]
    if text_cols_to_drop:
        print(f"  Dropping unexpected text columns: {text_cols_to_drop}")
        X_train_clean = X_train_clean.drop(columns=text_cols_to_drop)
        X_test_clean = X_test_clean.drop(columns=text_cols_to_drop)

    cat_cols_clean = [c for c in cat_cols if c in X_train_clean.columns]
    cat_idx = [X_train_clean.columns.get_loc(c) for c in cat_cols_clean]

    X_encoded = X_train_clean.copy()
    X_test_encoded = X_test_clean.copy()

    label_encoders = {}
    for c in cat_cols_clean:
        le = LabelEncoder()
        combined = pd.concat(
            [X_encoded[c].astype(str), X_test_encoded[c].astype(str)],
            ignore_index=True
        )
        le.fit(combined)
        X_encoded[c] = le.transform(X_encoded[c].astype(str)).astype(np.int64)
        X_test_encoded[c] = le.transform(X_test_encoded[c].astype(str)).astype(np.int64)
        label_encoders[c] = le

    cat_dims = [len(label_encoders[c].classes_) for c in cat_cols_clean]

    for c, dim in zip(cat_cols_clean, cat_dims):
        tr_min, tr_max = int(X_encoded[c].min()), int(X_encoded[c].max())
        te_min, te_max = int(X_test_encoded[c].min()), int(X_test_encoded[c].max())
        if tr_min < 0 or tr_max >= dim:
            raise ValueError(f"Bad TRAIN category ids for {c}: min={tr_min}, max={tr_max}, dim={dim}")
        if te_min < 0 or te_max >= dim:
            raise ValueError(f"Bad TEST category ids for {c}: min={te_min}, max={te_max}, dim={dim}")

    # Fill NaNs for numeric columns only (categoricals should already be ints)
    num_cols = [c for c in X_encoded.columns if c not in cat_cols_clean]
    X_encoded[num_cols] = X_encoded[num_cols].fillna(-999)
    X_test_encoded[num_cols] = X_test_encoded[num_cols].fillna(-999)

    # Convert to numpy
    X_np = X_encoded.values.astype(np.float32)
    X_test_np = X_test_encoded.values.astype(np.float32)
    y_np = y.astype(np.int64)

    print(f"  Final shapes: train={X_np.shape}, test={X_test_np.shape}")

    # CV Setup
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_train), dtype=np.float64)
    te = np.zeros(len(X_test), dtype=np.float64)

    # Class weights for imbalance
    pos, neg = float(y.sum()), float(len(y) - y.sum())
    class_weight_ratio = neg / max(pos, 1.0)
    print(f"  Class distribution: pos={int(pos)}, neg={int(neg)}, ratio={class_weight_ratio:.2f}")

    tabnet_params = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-4,
        "momentum": 0.3,
        "clip_value": 2.0,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 2e-3, "weight_decay": 1e-2},
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_params": {"mode": "min", "patience": 10, "factor": 0.5, "min_lr": 1e-5},
        "mask_type": "entmax",
        "seed": seed,
        "verbose": 1,
        "device_name": device,
    }

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y_np), 1):
        print(f"\n  {'='*60}")
        print(f"  TabNet Fold {fold}/{N_SPLITS} - Target: {target_name}d")
        print(f"  {'='*60}")

        sw_tr = sample_weight[tr_idx].astype(np.float32) if sample_weight is not None else None

        model = TabNetClassifier(
            cat_idxs=cat_idx,
            cat_dims=cat_dims,
            cat_emb_dim=1,
            **tabnet_params
        )

        class_weights = torch.FloatTensor([1.0, class_weight_ratio])
        if device == "cuda":
            class_weights = class_weights.cuda()

        try:
            model.fit(
                X_train=X_np[tr_idx],
                y_train=y_np[tr_idx],
                eval_set=[(X_np[va_idx], y_np[va_idx])],
                eval_name=["validation"],
                eval_metric=["auc", "logloss"],
                max_epochs=300,
                patience=40,
                batch_size=2048,
                virtual_batch_size=512,
                num_workers=0,
                drop_last=False,
                weights=sw_tr,
                loss_fn=torch.nn.CrossEntropyLoss(),
            )
        except Exception as e:
            if "device-side assert" in str(e).lower():
                raise RuntimeError(
                    "CUDA device-side assert triggered. This is almost always an out-of-range "
                    "categorical index (cat_dims mismatch). With CUDA asserts you MUST restart "
                    "the process after fixing the root cause."
                ) from e
            raise

        oof[va_idx] = model.predict_proba(X_np[va_idx])[:, 1]
        te += model.predict_proba(X_test_np)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_np[va_idx], oof[va_idx])
        fold_ll = log_loss(y_np[va_idx], np.clip(oof[va_idx], EPS, 1 - EPS))
        print(f"  Fold {fold} -> AUC: {fold_auc:.6f}, LL: {fold_ll:.6f}")

    oof = np.clip(oof, EPS, 1 - EPS)
    te = np.clip(te, EPS, 1 - EPS)

    final_auc = roc_auc_score(y, oof)
    final_ll = log_loss(y, oof)

    print(f"\n  {'='*60}")
    print(f"  TabNet {target_name}d Final OOF:")
    print(f"     AUC:     {final_auc:.6f}")
    print(f"     LogLoss: {final_ll:.6f}")
    print(f"  {'='*60}\n")

    return oof, te

def main():

    """
    Main pipeline workflow:
    - Load data (Train, Test, Prior)
    - Feature engineering (topics, dates, historical rates, etc.)
    - Test-era features extraction (calibration target correction)
    - Feature selection (stability-based)
    - Model training (TabNet, LightGBM, XGBoost with multiple seeds)
    - Calibration (single pass with test-weighted base rates)
    - Monotone enforcement (p7 ≤ p90 ≤ p120)
    - Submission generation
    """
    print("="*80)
    print("DigiCOW Farmer Traning Model P_B")
    print("="*80 + "\n")
    import os
    os.chdir('/home/rmissodey/DigiCow/')
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("Test.csv")
    prior = pd.read_csv("Prior.csv")
    sample = pd.read_csv("SampleSubmission.csv")

    if "Prior ID" in prior.columns and "ID" not in prior.columns:
        prior = prior.rename(columns={"Prior ID": "ID"})

    for df in [train, test, prior]:
        df["trainer_first"] = df["trainer"].apply(parse_trainer_first)

    train = add_date_features(train)
    test = add_date_features(test)
    prior = add_date_features(prior)

    train = add_training_dt(train)
    test = add_training_dt(test)
    prior = add_training_dt(prior)

    for df in [train, test, prior]:
        df["trainer_first"] = df["trainer"].apply(parse_trainer_first)

    train = add_topic_features(train)
    test = add_topic_features(test)
    prior = add_topic_features(prior)

    for df in (train, test, prior):
        df["belong_to_cooperative"] = pd.to_numeric(df["belong_to_cooperative"], errors="coerce").fillna(0).astype(int)
        df["has_topic_trained_on"] = pd.to_numeric(df["has_topic_trained_on"], errors="coerce").fillna(0).astype(int)

    train = add_hier_keys(train)
    test = add_hier_keys(test)
    prior = add_hier_keys(prior)

    top_topics = get_top_topics(train, prior, k=TOPK_TOPICS)
    train = add_topk_topic_flags(train, top_topics)
    test = add_topk_topic_flags(test, top_topics)
    prior = add_topk_topic_flags(prior, top_topics)

    train_svd, test_svd, prior_svd = build_topic_svd_embeddings(
        train["topics_text"], test["topics_text"], prior["topics_text"], n_components=20)
    for i in range(20):
        train[f"svd_{i:02d}"] = train_svd[:, i]
        test[f"svd_{i:02d}"] = test_svd[:, i]
        prior[f"svd_{i:02d}"] = prior_svd[:, i]

    print("Pre-computing topic lift scores...")
    lift_dict, base_rates = build_topic_lift_scores(prior)

    # ========================
    # Feature engineering
    # ========================
    print("Building prior-asof features...")
    prior_feat = build_prior_asof_features(prior, prior, time_col="training_dt")
    train_feat = build_prior_asof_features(prior, train, time_col="training_dt")
    test_feat = build_prior_asof_features(prior, test, time_col="training_dt")

    print("Adding topic history features...")
    prior_feat = add_topic_hist_features(prior, prior_feat, top_topics, alpha=TE_ALPHA, time_col="training_dt")
    train_feat = add_topic_hist_features(prior, train_feat, top_topics, alpha=TE_ALPHA, time_col="training_dt")
    test_feat = add_topic_hist_features(prior, test_feat, top_topics, alpha=TE_ALPHA, time_col="training_dt")

    print("Adding hierarchical historical rates...")
    all_hist_keys = HIST_KEYS + ["group_trainer", "ward_trainer"]
    for k in all_hist_keys:
        if k in prior.columns and k in train_feat.columns and k in test_feat.columns:
            prior_feat = add_hist_rates_from_prior(prior, prior_feat, k, alpha=TE_ALPHA, time_col="training_dt")
            train_feat = add_hist_rates_from_prior(prior, train_feat, k, alpha=TE_ALPHA, time_col="training_dt")
            test_feat = add_hist_rates_from_prior(prior, test_feat, k, alpha=TE_ALPHA, time_col="training_dt")

    # ========================
    # Recency-weighted historical rates for key groupings
    # ========================
    print("Adding recency-weighted historical rates...")
    for key in ["county", "trainer_first", "subcounty", "ward"]:
        if key in prior.columns:
            prior_feat = add_recency_weighted_hist_rates(prior, prior_feat, key=key,
                                                          time_col="training_dt", halflife_days=180)
            train_feat = add_recency_weighted_hist_rates(prior, train_feat, key=key,
                                                          time_col="training_dt", halflife_days=180)
            test_feat  = add_recency_weighted_hist_rates(prior, test_feat,  key=key,
                                                          time_col="training_dt", halflife_days=180)

    print("Adding rolling window features...")
    for key in ["farmer_name", "trainer_first", "group_name"]:
        prior_feat = add_window_features_from_prior(prior, prior_feat, key, WINDOWS_DAYS, time_col="training_dt")
        train_feat = add_window_features_from_prior(prior, train_feat, key, WINDOWS_DAYS, time_col="training_dt")
        test_feat = add_window_features_from_prior(prior, test_feat, key, WINDOWS_DAYS, time_col="training_dt")

    print("Adding days-since-last-adopt...")
    prior_feat = add_days_since_last_adopt(prior, prior_feat, key="farmer_name", time_col="training_dt")
    train_feat = add_days_since_last_adopt(prior, train_feat, key="farmer_name", time_col="training_dt")
    test_feat = add_days_since_last_adopt(prior, test_feat, key="farmer_name", time_col="training_dt")

    print("Adding trainer effectiveness features...")
    prior_feat = add_trainer_effectiveness(prior, prior_feat, time_col="training_dt")
    train_feat = add_trainer_effectiveness(prior, train_feat, time_col="training_dt")
    test_feat = add_trainer_effectiveness(prior, test_feat, time_col="training_dt")

    print("Adding peer pressure features...")
    prior_feat = add_peer_pressure_features(prior, prior_feat, time_col="training_dt")
    train_feat = add_peer_pressure_features(prior, train_feat, time_col="training_dt")
    test_feat = add_peer_pressure_features(prior, test_feat, time_col="training_dt")

    print("Adding geographic temporal trends...")
    prior_feat = add_geo_temporal_trends(prior, prior_feat, time_col="training_dt")
    train_feat = add_geo_temporal_trends(prior, train_feat, time_col="training_dt")
    test_feat = add_geo_temporal_trends(prior, test_feat, time_col="training_dt")

    print("Adding topic lift scores...")
    prior_feat = add_topic_lift_features(prior_feat, lift_dict, base_rates)
    train_feat = add_topic_lift_features(train_feat, lift_dict, base_rates)
    test_feat = add_topic_lift_features(test_feat, lift_dict, base_rates)

    print("Adding group/ward recency...")
    prior_feat = add_group_ward_recency(prior, prior_feat, time_col="training_dt")
    train_feat = add_group_ward_recency(prior, train_feat, time_col="training_dt")
    test_feat = add_group_ward_recency(prior, test_feat, time_col="training_dt")

    print("Adding trainer workload...")
    prior_feat = add_trainer_workload(prior, prior_feat, time_col="training_dt")
    train_feat = add_trainer_workload(prior, train_feat, time_col="training_dt")
    test_feat = add_trainer_workload(prior, test_feat, time_col="training_dt")

    # ========================
    # Test-era features + corrected calibration base rates
    # ========================
    print("\nAdding test-era features (FIX 1: correct calibration target)...")
    train_feat, test_feat, era_base_rates, test_weighted_rates = add_test_era_features(
        prior, train_feat, test_feat, time_col="training_dt"
    )

    for w in ["07","90","120"]:
        for grp in ["county","trainer","ward","sub"]:
            col = f"era_{grp}_{w}_rate"
            if col in train_feat.columns and col not in prior_feat.columns:
                prior_feat[col] = era_base_rates[w]
                prior_feat[f"era_{grp}_{w}_n"] = 0

    # ========================
    # Interaction features (registration×county, coop×county, etc.)
    # ========================
    print("Adding interaction features (FIX 2)...")
    train_feat = add_interaction_features(train_feat, prior)
    test_feat  = add_interaction_features(test_feat,  prior)
    prior_feat = add_interaction_features(prior_feat, prior)

    # ========================
    # Derived features
    # ========================
    for df in (prior_feat, train_feat, test_feat):
        df["is_repeat_trainee"] = (df["prior_training_count"] > 0).astype(int)
        df["log_prior_training_count"] = np.log1p(df["prior_training_count"])
        df["log_secs_since_prior"] = np.log1p(df["secs_since_prior"])
        df["topic_intensity"] = df["num_topics"] / (df["topics_diversity"] + 1.0)
        for w in ["07","90","120"]:
            df[f"prior_any_adopt{w}"] = (df[f"prior_adopted_{w}_sum"] > 0).astype(int)
            df[f"log_secs_since_last_adopt_{w}"] = np.log1p(df[f"secs_since_last_adopt_{w}"])
        df["coop_x_prior_count"] = df["belong_to_cooperative"] * df["prior_training_count"]
        df["has_topic_x_prior_adopt120"] = df["has_topic_trained_on"] * df.get("prior_adopted_120_rate", 0)
        df["trainer_x_group_rate120"] = df.get("trainer_adopt_120_rate", 0) * df.get("peer_adopt_ratio_120", 0)
        df["ewm_x_peer120"] = df.get("ewm_adopted_120", 0) * df.get("peer_adopt_ratio_120", 0)
        df["prior_count_x_adopt_rate120"] = df["prior_training_count"] * df.get("prior_adopted_120_rate", 0)
        df["lift_x_farmer_rate_120"] = df.get("topic_max_lift_120", 1.0) * df.get("prior_adopted_120_rate", 0)
        df["svd00_x_prior_rate120"] = df.get("svd_00", 0) * df.get("prior_adopted_120_rate", 0)
        df["prior_rate120_x_geo_trend120"] = df.get("prior_adopted_120_rate", 0) * df.get("subcounty_trend_120", 0)
        df["ewm120_x_geo_ewm120"] = df.get("ewm_adopted_120", 0) * df.get("county_ewm_rate_120", 0)
        df["workload_x_trainer_rate120"] = df.get("trainer_workload_7d", 0) * df.get("trainer_adopt_120_rate", 0)
        df["streak_x_prior_rate120"] = df.get("non_adopt_streak_120", 0) * df.get("prior_adopted_120_rate", 0)
        for w in ["90","120"]:
            hist_col = f"trainer_adopt_{w}_rate"
            recent_col = f"trainer_recent_{w}"
            if hist_col in df.columns and recent_col in df.columns:
                df[f"trainer_momentum_{w}"] = df[recent_col] - df[hist_col]

        for w in ["07","90","120"]:
            # County era momentum
            hist_col = f"county_hist_rate{w}"
            era_col  = f"era_county_{w}_rate"
            if hist_col in df.columns and era_col in df.columns:
                df[f"county_era_vs_hist_{w}"] = df[era_col] / (df[hist_col] + 1e-10)

            # Trainer era momentum
            t_hist = f"trainer_adopt_{w}_rate"
            t_era  = f"era_trainer_{w}_rate"
            if t_hist in df.columns and t_era in df.columns:
                df[f"trainer_era_vs_hist_{w}"] = df[t_era] / (df[t_hist] + 1e-10)

            # Subcounty era momentum
            sub_hist = f"subcounty_hist_rate{w}"
            sub_era  = f"era_sub_{w}_rate"
            if sub_hist in df.columns and sub_era in df.columns:
                df[f"sub_era_vs_hist_{w}"] = df[sub_era] / (df[sub_hist] + 1e-10)

            # Ward era momentum
            ward_hist = f"ward_hist_rate{w}"
            ward_era  = f"era_ward_{w}_rate"
            if ward_hist in df.columns and ward_era in df.columns:
                df[f"ward_era_vs_hist_{w}"] = df[ward_era] / (df[ward_hist] + 1e-10)

            # Recency-weighted vs era rate (how much is rw rate diverging from era?)
            rw_col = f"county_rw_rate{w}"
            if rw_col in df.columns and era_col in df.columns:
                df[f"county_rw_vs_era_{w}"] = df[rw_col] / (df[era_col] + 1e-10)

    # County features
    comb = pd.concat([train_feat, prior_feat], ignore_index=True)
    county_size = comb["county"].value_counts()
    g_topic = comb["has_topic_trained_on"].mean()
    county_topic_rate = comb.groupby("county")["has_topic_trained_on"].mean()
    for df in (prior_feat, train_feat, test_feat):
        df["county_size"] = df["county"].map(county_size).fillna(0).astype(int)
        df["log_county_size"] = np.log1p(df["county_size"])
        df["county_topic_rate"] = df["county"].map(county_topic_rate).fillna(g_topic)

    # ========================
    # Build labeled set
    # ========================
    targets = ["adopted_within_07_days","adopted_within_90_days","adopted_within_120_days"]
    for t in targets:
        prior_feat[t] = pd.to_numeric(prior_feat.get(t, 0), errors="coerce").fillna(0).astype(int)
        train_feat[t] = pd.to_numeric(train_feat.get(t, 0), errors="coerce").fillna(0).astype(int)

    n_train = len(train_feat)
    n_prior = len(prior_feat)
    labeled = pd.concat([train_feat, prior_feat], ignore_index=True)

    train_mask = np.zeros(len(labeled), dtype=bool)
    train_mask[:n_train] = True
    print(f"\n  Train rows: {train_mask.sum()}, Prior rows: {(~train_mask).sum()}")

    y7 = labeled["adopted_within_07_days"].astype(int).values
    y90 = labeled["adopted_within_90_days"].astype(int).values
    y120 = labeled["adopted_within_120_days"].astype(int).values

    print(f"\nTarget distribution:")
    print(f"  ALL  -> 7d: {y7.mean():.6f}, 90d: {y90.mean():.6f}, 120d: {y120.mean():.6f}")
    print(f"  TRAIN-> 7d: {y7[train_mask].mean():.6f}, 90d: {y90[train_mask].mean():.6f}, 120d: {y120[train_mask].mean():.6f}")
    print(f"  PRIOR-> 7d: {y7[~train_mask].mean():.6f}, 90d: {y90[~train_mask].mean():.6f}, 120d: {y120[~train_mask].mean():.6f}")

    # ========================
    #  Use test-weighted rates as calibration targets 
    # ========================
    train_rate_07  = y7[train_mask].mean()
    train_rate_90  = y90[train_mask].mean()
    train_rate_120 = y120[train_mask].mean()

    est_rate_07  = test_weighted_rates["07"]
    est_rate_90  = test_weighted_rates["90"]
    est_rate_120 = test_weighted_rates["120"]

    print(f"\n  CALIBRATION TARGET SHIFT (FIX 4):")
    print(f"    07d:  {train_rate_07:.6f} (train) → {est_rate_07:.6f} (test-county-weighted)")
    print(f"    90d:  {train_rate_90:.6f} (train) → {est_rate_90:.6f} (test-county-weighted)")
    print(f"    120d: {train_rate_120:.6f} (train) → {est_rate_120:.6f} (test-county-weighted)")

    # Temporal sample weights
    sw_temporal = get_sample_weights(labeled["training_day"], decay=0.3)

        # ============================================================
    # TF-IDF TEXT FEATURE ENGINEERING (for TabNet)
    # ============================================================
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("Converting topics_text to TF-IDF features for TabNet...")


    # Ensure topics_text exists
    for df in [prior_feat, train_feat, test_feat]:
        if "topics_text" not in df.columns:
            df["topics_text"] = ""
        df["topics_text"] = df["topics_text"].fillna("").astype(str)

    # Combine all text data for vocabulary building
    all_topics_text = pd.concat([
        prior_feat["topics_text"],
        train_feat["topics_text"],
        test_feat["topics_text"]
    ], ignore_index=True)

    print(f"  Total text samples: {len(all_topics_text)}")
    print(f"  Non-empty samples: {(all_topics_text != '').sum()}")

    # TF-IDF parameters optimized for topic text
    tfidf = TfidfVectorizer(
        max_features=100,        
        ngram_range=(1, 3),      
        min_df=10,               
        max_df=0.7,              
        sublinear_tf=True,       
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )

    # Fit on all data
    tfidf_matrix = tfidf.fit_transform(all_topics_text)

    print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Split back to original dataframes
    n_prior = len(prior_feat)
    n_train = len(train_feat)
    n_test = len(test_feat)

    tfidf_prior = tfidf_matrix[:n_prior]
    tfidf_train = tfidf_matrix[n_prior:n_prior+n_train]
    tfidf_test = tfidf_matrix[n_prior+n_train:]

    # Create column names
    tfidf_cols = [f"tfidf_{i:03d}" for i in range(tfidf_matrix.shape[1])]

    # Convert to DataFrames
    tfidf_prior_df = pd.DataFrame(
        tfidf_prior.toarray(), 
        columns=tfidf_cols, 
        index=prior_feat.index
    )
    tfidf_train_df = pd.DataFrame(
        tfidf_train.toarray(), 
        columns=tfidf_cols, 
        index=train_feat.index
    )
    tfidf_test_df = pd.DataFrame(
        tfidf_test.toarray(), 
        columns=tfidf_cols, 
        index=test_feat.index
    )

    # Add TF-IDF features to dataframes
    prior_feat = pd.concat([prior_feat, tfidf_prior_df], axis=1)
    train_feat = pd.concat([train_feat, tfidf_train_df], axis=1)
    test_feat = pd.concat([test_feat, tfidf_test_df], axis=1)

    print(f"  Added {len(tfidf_cols)} TF-IDF features")
    print(f"  Top 10 terms: {list(tfidf.vocabulary_.keys())[:10]}")

    if tfidf_matrix.shape[1] >= 10:
        feature_names = tfidf.get_feature_names_out()
        mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = np.argsort(mean_tfidf)[-10:][::-1]
        print("\n  Most important terms (by mean TF-IDF):")
        for idx in top_idx:
            print(f"    {feature_names[idx]:30s}: {mean_tfidf[idx]:.4f}")

    # NOW drop the original text column (TabNet can't use it)
    print("\n  Dropping original topics_text column...")
    prior_feat = prior_feat.drop(columns=["topics_text"], errors='ignore')
    train_feat = train_feat.drop(columns=["topics_text"], errors='ignore')
    test_feat = test_feat.drop(columns=["topics_text"], errors='ignore')

    print("  TF-IDF conversion complete!\n")
    # rebuild labeled AFTER TF-IDF so columns match
    labeled = pd.concat([train_feat, prior_feat], ignore_index=True)

    n_train = len(train_feat)
    train_mask = np.zeros(len(labeled), dtype=bool)
    train_mask[:n_train] = True

    y7   = labeled["adopted_within_07_days"].astype(int).values
    y90  = labeled["adopted_within_90_days"].astype(int).values
    y120 = labeled["adopted_within_120_days"].astype(int).values

    # recompute temporal weights to stay aligned with labeled rows
    sw_temporal = get_sample_weights(labeled["training_day"], decay=0.3)


    # ========================
    # Feature matrix
    # ========================
    drop_cols = [
        "ID","farmer_name","training_day","training_dt","trainer",
        "topics_list","topics_flat","topics_norm","topics_token_str",
        "prior_last_topics_token_str","prior_last_trainer_first",
        "intra_day_rank","topic_categories","topic_cat_str",
    ]

    X_cols = [c for c in labeled.columns if c not in drop_cols + targets]
    X_train_full = labeled[X_cols].copy()
    X_test_full = test_feat[X_cols].copy()

    cat_cols = [
        "gender","registration","age","county","subcounty","ward","group_name",
        "trainer_first","county_subcounty","subcounty_ward","county_trainer",
        "subcounty_trainer","group_trainer","ward_trainer","primary_topic_cat",
    ]
    cat_cols = [c for c in cat_cols if c in X_train_full.columns]
    text_cols = ["topics_text"] if "topics_text" in X_train_full.columns else []

    allowed_obj = set(cat_cols + text_cols)
    bad_obj = [c for c in X_train_full.columns if X_train_full[c].dtype == "object" and c not in allowed_obj]
    if bad_obj:
        print("Dropping bad object cols:", bad_obj)
        X_train_full.drop(columns=bad_obj, inplace=True)
        X_test_full.drop(columns=bad_obj, inplace=True)

    print(f"\nFeatures before selection: {X_train_full.shape[1]}")

    keep_feats = stability_feature_selection(X_train_full, y120, cat_cols,
                                              target_n=TARGET_N_FEATURES)
    X_train = X_train_full[keep_feats].copy()
    X_test = X_test_full[keep_feats].copy()
    cat_cols = [c for c in cat_cols if c in keep_feats]
    text_cols = [c for c in text_cols if c in keep_feats]

    print(f"Features after selection: {X_train.shape[1]}, "
          f"Train+Prior: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

    adv_weights = compute_adversarial_weights(X_train, X_test, keep_feats, cat_cols)

    sw = sw_temporal * adv_weights
    sw[train_mask] *= 1.1
    sw = sw / sw.mean()
    print(f"Combined weights: min={sw.min():.4f}, max={sw.max():.4f}\n")

    gpu_available = USE_GPU
    if USE_GPU:
        try:
            import xgboost as xgb
            xgb.DMatrix(np.random.randn(10, 3))
            gpu_available = True
            print("GPU: Available\n")
        except:
            gpu_available = False
            print("GPU: Not available\n")


# ============================================================
    # BINARY MODELS: TabNet + LGB + XGB ---> ensemble
    # ============================================================
    final_oof = {}
    final_te = {}

    for label, y_true, est_rate in [
        ("07", y7, est_rate_07),
        ("90", y90, est_rate_90),
        ("120", y120, est_rate_120),
    ]:
        print(f"\n{'='*70}")
        print(f"  HORIZON: {label} days | est_rate: {est_rate:.6f}")
        print(f"{'='*70}")

        oof_models = {}
        te_models = {}

        # --- TabNet (3 seeds) ---
        oof_tab_sum = np.zeros(len(X_train))
        te_tab_sum = np.zeros(len(X_test))
        for s in SEEDS:
            print(f"\n[TabNet {label}d seed={s}]")
            oof_s, te_s = train_tabnet_binary_cv(
                X_train, y_true, X_test, cat_cols,
                sample_weight=sw, seed=s, target_name=label
            )
            oof_tab_sum += oof_s
            te_tab_sum += te_s
        oof_models["TABNET"] = oof_tab_sum / len(SEEDS)
        te_models["TABNET"] = te_tab_sum / len(SEEDS)

        # --- LightGBM (3 seeds) ---
        oof_lgb_sum = np.zeros(len(X_train))
        te_lgb_sum = np.zeros(len(X_test))
        for s in SEEDS:
            print(f"\n[LGB {label} seed={s}]")
            oof_l, te_l = train_lgbm_binary_cv(
                X_train, y_true, X_test, cat_cols,
                sample_weight=sw, seed=s + int(label) * 10)
            oof_lgb_sum += oof_l
            te_lgb_sum += te_l
        oof_models["LGB"] = oof_lgb_sum / len(SEEDS)
        te_models["LGB"] = te_lgb_sum / len(SEEDS)

        # --- XGBoost (3 seeds) ---
        oof_xgb_sum = np.zeros(len(X_train))
        te_xgb_sum = np.zeros(len(X_test))
        for s in SEEDS:
            print(f"\n[XGB {label} seed={s}]")
            oof_x, te_x = train_xgb_binary_cv(
                X_train, y_true, X_test, cat_cols,
                sample_weight=sw, seed=s + int(label) * 100,
                use_gpu=gpu_available)
            oof_xgb_sum += oof_x
            te_xgb_sum += te_x
        oof_models["XGB"] = oof_xgb_sum / len(SEEDS)
        te_models["XGB"] = te_xgb_sum / len(SEEDS)

        # --- Print individual scores ---
        print(f"\n  Individual OOF scores ({label}):")
        for nm in oof_models:
            o = oof_models[nm]
            print(f"    {nm}: AUC={roc_auc_score(y_true, o):.6f}, "
                  f"LL={log_loss(y_true, np.clip(o, EPS, 1-EPS)):.6f}, "
                  f"LL_train={log_loss(y_true[train_mask], np.clip(o[train_mask], EPS, 1-EPS)):.6f}")

        # --- LL ensemble (logit-space) ---
        ll_keys = ["TABNET", "LGB", "XGB"]
        oof_list = [oof_models[k] for k in ll_keys]
        te_list = [te_models[k] for k in ll_keys]

        print(f"\n  Optimizing LL ensemble ({label}) in logit space: {ll_keys}...")
        oof_ll_blend, te_ll_blend = optimize_logloss_weights_logit(
            oof_list, y_true, te_list)

        # --- AUC ensemble (random search) ---
        print(f"  Optimizing AUC ensemble ({label})...")
        te_auc_blend, _ = optimize_auc_weights(
            [oof_models[k] for k in ll_keys], y_true,
            [te_models[k] for k in ll_keys])

        # Store RAW blends (NO calibration here)
        final_oof[f"{label}_ll"] = oof_ll_blend
        final_te[f"{label}_ll"] = te_ll_blend
        final_te[f"{label}_auc"] = te_auc_blend

        gc.collect()

    # ============================================================
    # SINGLE CALIBRATION PASS (no double-calibration)
    # ============================================================
    print("\n" + "="*80)
    print("CALIBRATION: single pass with test-county-weighted base rate")
    print("="*80)

    estimated_rates = {"07": est_rate_07, "90": est_rate_90, "120": est_rate_120}

    for label in ["07", "90", "120"]:
        y_true = {"07": y7, "90": y90, "120": y120}[label]

        te_cal, oof_cal = calibrate(
            final_oof[f"{label}_ll"], y_true, final_te[f"{label}_ll"],
            name=f"{label}_ll",
            train_mask=train_mask,
            target_base_rate=estimated_rates[label]
        )
        final_te[f"{label}_ll"] = te_cal

    # ============================================================
    # Build submission
    # ============================================================
    out = pd.DataFrame({"ID": test_feat["ID"].values})
    for col in sample.columns:
        if col == "ID":
            continue
        if col == "Target_07_AUC":        out[col] = final_te["07_auc"]
        elif col == "Target_90_AUC":      out[col] = final_te["90_auc"]
        elif col == "Target_120_AUC":     out[col] = final_te["120_auc"]
        elif col == "Target_07_LogLoss":  out[col] = final_te["07_ll"]
        elif col == "Target_90_LogLoss":  out[col] = final_te["90_ll"]
        elif col == "Target_120_LogLoss": out[col] = final_te["120_ll"]
        else:
            out[col] = 0.0

    out = out[sample.columns]
    out.to_csv("submission_v15_ensemble_1.csv", index=False)

    print("\n" + "="*80)
    print("SUBMISSION SAVED: submission_v15_ensemble_1.csv")
    print("="*80)
    print(out.head(10))
    print("\nPrediction stats:")
    for col in out.columns:
        if col != "ID":
            print(f"  {col:25s}: mean={out[col].mean():.6f}, "
                  f"min={out[col].min():.8f}, max={out[col].max():.6f}")


if __name__ == "__main__":
    main()