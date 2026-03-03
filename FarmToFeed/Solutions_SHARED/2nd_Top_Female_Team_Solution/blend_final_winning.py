"""
WINNING SOLUTION - Final Blend (V2 + V4 + V11)
===============================================
Farm to Feed Challenge - AntiGravity

This script creates the TOP SCORING submission by blending three pipelines:
- V2: Target Encoding features (for diversity)
- V4: Advanced Features (best for MAE/quantity)
- V11: SVD Embeddings (best for WAUC/classification)

Scores Achieved:
- Target 1 WAUC: 0.965566394
- Target 2 WAUC: 0.965629073
- Target 1 W Qty MAE: 0.290462697
- Target 2 W Qty MAE: 0.470406171

Blending Strategy:
- WAUC predictions: Blend V2 + V11 (V11 dominant, V2 for diversity)
- MAE predictions: 100% V4 (specialized for quantity)

Prerequisites:
- Run farm_to_feed_v2.py -> submission_v2.csv
- Run farm_to_feed_v4.py -> submission_v4.csv
- Run farm_to_feed_v11.py -> submission_v11.csv

Output: submission_blend_v2_v4_v11.csv
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("FARM TO FEED - WINNING BLEND")
print("=" * 60)

# =============================================================================
# BLEND WEIGHTS (Optimized through experimentation)
# =============================================================================

# Target 1 (1-week purchase prediction)
T1_V2_WEIGHT = 0.35      # V2 contributes 35%
T1_V11_WEIGHT = 0.65     # V11 contributes 65%

# Target 2 (2-week purchase prediction)
T2_V2_WEIGHT = 0.25      # V2 contributes 25%
T2_V11_WEIGHT = 0.75     # V11 contributes 75%

# Quantity predictions: 100% from V4 (best MAE performance)

print(f"\nBlend Weights:")
print(f"  Target 1 WAUC: {T1_V2_WEIGHT:.0%} V2 + {T1_V11_WEIGHT:.0%} V11")
print(f"  Target 2 WAUC: {T2_V2_WEIGHT:.0%} V2 + {T2_V11_WEIGHT:.0%} V11")
print(f"  Quantity:      100% V4")

# =============================================================================
# LOAD SUBMISSIONS
# =============================================================================

print("\nLoading base submissions...")

try:
    v2 = pd.read_csv('submission_v2.csv')
    print(f"  V2 loaded: {v2.shape}")
except FileNotFoundError:
    raise FileNotFoundError("submission_v2.csv not found. Run farm_to_feed_v2.py first.")

try:
    v4 = pd.read_csv('submission_v4.csv')
    print(f"  V4 loaded: {v4.shape}")
except FileNotFoundError:
    raise FileNotFoundError("submission_v4.csv not found. Run farm_to_feed_v4.py first.")

try:
    v11 = pd.read_csv('submission_v11.csv')
    print(f"  V11 loaded: {v11.shape}")
except FileNotFoundError:
    raise FileNotFoundError("submission_v11.csv not found. Run farm_to_feed_v11.py first.")

# Verify alignment
assert len(v2) == len(v4) == len(v11), "Submission files have different lengths!"
assert list(v2['ID']) == list(v4['ID']) == list(v11['ID']), "ID columns don't match!"

# =============================================================================
# CREATE BLEND
# =============================================================================

print("\nCreating blend...")

blend = v2[['ID']].copy()

# WAUC Target 1: Blend V2 + V11
blend['Target_purchase_next_1w'] = (
    T1_V2_WEIGHT * v2['Target_purchase_next_1w'] + 
    T1_V11_WEIGHT * v11['Target_purchase_next_1w']
)

# WAUC Target 2: Blend V2 + V11
blend['Target_purchase_next_2w'] = (
    T2_V2_WEIGHT * v2['Target_purchase_next_2w'] + 
    T2_V11_WEIGHT * v11['Target_purchase_next_2w']
)

# Quantity: Use V4 (100% - best for MAE)
blend['Target_qty_next_1w'] = v4['Target_qty_next_1w']
blend['Target_qty_next_2w'] = v4['Target_qty_next_2w']

# =============================================================================
# SAVE SUBMISSION
# =============================================================================

output_file = 'submission_blend_v2_v4_v11.csv'
blend.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 60)
print("BLEND STATISTICS")
print("=" * 60)

print("\nPrediction Ranges:")
for col in ['Target_purchase_next_1w', 'Target_purchase_next_2w', 
            'Target_qty_next_1w', 'Target_qty_next_2w']:
    print(f"  {col}:")
    print(f"    mean={blend[col].mean():.6f}, std={blend[col].std():.6f}")
    print(f"    min={blend[col].min():.6f}, max={blend[col].max():.6f}")

print("\n" + "=" * 60)
print("EXPECTED SCORES (from leaderboard)")
print("=" * 60)
print("  Target 1 WAUC:    0.965566394")
print("  Target 2 WAUC:    0.965629073")
print("  Target 1 W MAE:   0.290462697")
print("  Target 2 W MAE:   0.470406171")
print("=" * 60)

print(f"\nSubmission file ready: {output_file}")
print("Upload to Zindi for final submission!")
