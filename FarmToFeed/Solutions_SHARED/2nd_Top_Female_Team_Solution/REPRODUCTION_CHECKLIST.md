# Reproduction Checklist

Quick-reference checklist to reproduce the winning solution.

## Pre-flight Checks

- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Data files in place:
  - [ ] `Train.csv` (275 MB)
  - [ ] `Test.csv` (27 MB)
  - [ ] `SampleSubmission.csv` (7 MB)

## Training Pipeline

### Step 1: V2 Pipeline (Target Encoding)
```bash
python farm_to_feed_v2.py
```
- [ ] Script completes without errors (~2-3 hours)
- [ ] `submission_v2.csv` created

### Step 2: V4 Pipeline (Advanced Features)
```bash
python farm_to_feed_v4.py
```
- [ ] Script completes without errors (~2-3 hours)
- [ ] `submission_v4.csv` created

### Step 3: V11 Pipeline (SVD Embeddings)
```bash
python farm_to_feed_v11.py
```
- [ ] Script completes without errors (~3-4 hours)
- [ ] `submission_v11.csv` created

### Step 4: Create Final Blend
```bash
python blend_final_winning.py
```
- [ ] Script completes successfully
- [ ] `submission_blend_v2_v4_v11.csv` created

## Verification

Check output file statistics match expected ranges:
- [ ] `Target_purchase_next_1w`: mean ~0.15-0.20
- [ ] `Target_purchase_next_2w`: mean ~0.25-0.30
- [ ] `Target_qty_next_1w`: mean ~1-3
- [ ] `Target_qty_next_2w`: mean ~3-6

## Submission

- [ ] Upload `submission_blend_v2_v4_v11.csv` to Zindi
- [ ] Verify leaderboard scores match expected:
  - Target 1 WAUC: ~0.9656
  - Target 2 WAUC: ~0.9656
  - Target 1 W MAE: ~0.29
  - Target 2 W MAE: ~0.47

## Backup Submission

If needed, create backup submission:
```bash
python blend_v4_v11.py
```
- [ ] `submission_blend_v4_v11.csv` created

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory errors | Reduce batch sizes or use machine with 32GB+ RAM |
| GPU errors | Set `USE_GPU = False` in V4 script |
| Missing files | Verify all base submissions exist before blending |
| Different scores | Check random seed is set to 42 |

---

*Last updated: January 2026*
