"""
Blend V4 + V11

Strategy:
- V11: Better WAUC (purchase predictions) 
- V4: Better MAE (quantity predictions)

Output: submission_blend_v4_v11.csv
"""

import pandas as pd
import numpy as np

# Load submissions
v4 = pd.read_csv('submission_v4.csv')
v11 = pd.read_csv('submission_v11.csv')

print(f"V4 shape: {v4.shape}")
print(f"V11 shape: {v11.shape}")

# Create blend
blend = v4[['ID']].copy()

# WAUC targets: Use V11 (better classification)
blend['Target_purchase_next_1w'] = v11['Target_purchase_next_1w']
blend['Target_purchase_next_2w'] = v11['Target_purchase_next_2w']

# MAE targets: Use V4 (better quantity predictions)
blend['Target_qty_next_1w'] = v4['Target_qty_next_1w']
blend['Target_qty_next_2w'] = v4['Target_qty_next_2w']

# Save
blend.to_csv('submission_blend_v4_v11.csv', index=False)
print("\nSaved: submission_blend_v4_v11.csv")

# Summary
print("\n" + "="*60)
print("BLEND SUMMARY")
print("="*60)
print("Purchase predictions (WAUC): V11")
print("Quantity predictions (MAE):  V4")
print("="*60)

# Stats comparison
print("\nPrediction Statistics:")
print(f"\nTarget_purchase_next_1w (V11): mean={v11['Target_purchase_next_1w'].mean():.4f}, std={v11['Target_purchase_next_1w'].std():.4f}")
print(f"Target_purchase_next_2w (V11): mean={v11['Target_purchase_next_2w'].mean():.4f}, std={v11['Target_purchase_next_2w'].std():.4f}")
print(f"Target_qty_next_1w (V4):       mean={v4['Target_qty_next_1w'].mean():.4f}, std={v4['Target_qty_next_1w'].std():.4f}")
print(f"Target_qty_next_2w (V4):       mean={v4['Target_qty_next_2w'].mean():.4f}, std={v4['Target_qty_next_2w'].std():.4f}")
