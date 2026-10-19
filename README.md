# Lab I — Multi-View VAE for MNIST and SVHN

Implements a **multi-view variational autoencoder** that learns a shared latent space for
paired MNIST (grey-scale) and SVHN (colour) digit images. Two fusion methods are compared:
**Product of Experts** (MVAE objective) and **Mixture of Experts** (MMVAE objective).
Evaluation: reconstruction, prior samples, cross-domain generation, coherence measured with
digit classifiers, linear probes and t-SNE.

## Quick start

### Option A — Colab (recommended, GPU)

1. Upload the repo to Google Drive or clone it inside Colab.
2. Open `notebooks/Lab_MultiView_VAE.ipynb` with Colab.
3. Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all.
   On GPU the full dataset (~56k pairs) is used automatically. Both models train in ~25 min total.
4. Download `checkpoints/` and `report/figures/` before the runtime is closed.

### Option B — Local CPU

```bash
# 1. Create / activate the virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and open the notebook
jupyter lab notebooks/Lab_MultiView_VAE.ipynb
```

On CPU the notebook uses a 12,000-pair subset automatically (`MAX_TRAIN_PAIRS`). Expect over 1 h for both models.

### Option C — Execute headlessly (nbconvert)

```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output notebooks/Lab_MultiView_VAE_executed.ipynb \
    notebooks/Lab_MultiView_VAE.ipynb
```

## Repository layout

```
notebooks/
  Lab_MultiView_VAE.ipynb   ← main deliverable
reference/
  Lab_VAEs_solved_CelebA.ipynb
  STUDENT_Lab_VAEs_CelebA.ipynb
  VAE_checkpoint_CelebA.pth
report/
  REPORT_NOTES.md           ← notes for the 2-page report
  figures/                  ← saved figures (generated at runtime)
data/                       ← downloaded datasets (git-ignored)
checkpoints/                ← model checkpoints (git-ignored)
```

## Outputs produced by the notebook (`report/figures/`)

| File | Content |
|------|---------|
| `01_pairs.png` | Example MNIST / SVHN training pairs |
| `02_loss.png` | Train / test loss for PoE and MoE |
| `03_recon_{poe,moe}.png` | Reconstruction from one view and from both views |
| `04_prior_{poe,moe}.png` | Samples from the prior decoded to both views |
| `05_cross_{poe,moe}.png` | Cross-generation MNIST → SVHN and SVHN → MNIST |
| `06_tsne_{poe,moe}.png` | t-SNE: joint latent by digit, unimodal latents by digit and by view |
| `summary.csv` | All metrics: recon MSE, joint/cross coherence, linear probes |
