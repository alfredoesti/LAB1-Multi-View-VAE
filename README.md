# Lab I — Multi-View VAE for MNIST and SVHN

Implements a **multi-view variational autoencoder** that learns a shared latent space for
paired MNIST (grey-scale) and SVHN (colour) digit images.

## Quick start

### Option A — Colab (recommended, GPU)

1. Upload the repo to Google Drive or clone it inside Colab.
2. Open `notebooks/Lab_MultiView_VAE.ipynb` with Colab.
3. Set `MAX_TRAIN_PAIRS = None` and `EPOCHS = 30` in Section 2 / 4 for full-quality training.
4. Run all cells (Runtime → Run all).

### Option B — Local CPU

```bash
# 1. Create / activate the virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install torch torchvision matplotlib scikit-learn jupyter

# 3. Launch Jupyter and open the notebook
jupyter lab notebooks/Lab_MultiView_VAE.ipynb
```

With the default `MAX_TRAIN_PAIRS = 12_000` the training loop takes **~35 min on a modern CPU**.
Remove the cap (`None`) or increase `EPOCHS` for better quality at the cost of more time.

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

## Outputs produced by the notebook

| Figure | Content |
|--------|---------|
| `01_paired_samples.png` | Example MNIST / SVHN pairs (same digit) |
| `02_loss_curves.png` | Training loss and ELBO components |
| `03a_mnist_recon.png` | MNIST within-view reconstruction |
| `03b_svhn_recon.png` | SVHN within-view reconstruction |
| `04_prior_samples.png` | Samples from the prior decoded to both views |
| `05a_cross_m2s.png` | Cross-generation MNIST → SVHN |
| `05b_cross_s2m.png` | Cross-generation SVHN → MNIST |
| `06a_tsne_label.png` | t-SNE coloured by digit label |
| `06b_tsne_view.png` | t-SNE coloured by view (MNIST vs SVHN) |
