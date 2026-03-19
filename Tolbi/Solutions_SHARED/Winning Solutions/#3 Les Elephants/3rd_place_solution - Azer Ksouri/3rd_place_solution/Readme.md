# ML Solution Reproduction Guide

## 🚀 Steps to Reproduce Our Solution

---

### **1. Train Swin Transformer Models (Kaggle, P100 GPU)**

**Notebooks to run:**
- `swin_small_mixup20_focalloss.ipynb`
- `swin_small_morebands.ipynb`
- `swin_small_newloss_morebands_scheduler_optim.ipynb`

**How:**
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Open each notebook above, set **Accelerator** to `GPU (P100)`, and run all cells.

> Each notebook will generate two files:
> - `train_oof_proba`
> - `test_oof_proba`

**Save these outputs for the blending step.**

---

### **2. Train ViT Model (PaperSpace, A6000 GPU)**

1. Launch a notebook on [PaperSpace](https://www.paperspace.com/) with an **A6000 GPU**.
2. Open the terminal and run:
    ```bash
    chmod +x setup-env.sh
    sh setup-env.sh
    ```
3. Open `vit__newloss_morebands_scheduler_optim.ipynb`.
4. Run the pip install cells.
5. Restart the kernel.
6. (Optional) Comment out the pip install cells after restart to avoid re-installation.
7. Run all cells.

> This will also output:
> - `train_oof_proba`
> - `test_oof_proba`

---

### **3. Blending Final Predictions**

1. Collect **all** `train_oof_proba` and `test_oof_proba` files from previous steps.
2. Open and run `BlendingNotebook.ipynb`.
   - Make sure the output files are available in the working directory or update notebook paths as needed.
   - Run all cells to generate your final blended predictions.

---

## 📝 **Notes & Tips**

- Make sure all paths to data and output files are correct for your environment.
- The `setup-env.sh` script is only needed on PaperSpace, not Kaggle.
- You may need to adjust dataset paths if your files are stored differently.
- Keep pip package versions as close to the original environments as possible for reproducibility.
- All model training steps are independent, but **blending** needs the outputs from every notebook.
---
