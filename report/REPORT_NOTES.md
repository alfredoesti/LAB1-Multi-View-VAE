# Report Notes — Multi-View VAE (MNIST + SVHN)

Use this as a draft outline for the 2-page PDF report.

---

## 1. Introduction (≈ 3 sentences)

We trained a multi-view variational autoencoder (MV-VAE) to learn a shared latent space
for two digit datasets: MNIST (grey-scale) and SVHN (colour, 32×32).
The model can reconstruct both views from either view alone, generate new digit pairs
from the latent prior, and perform cross-domain translation (MNIST → SVHN and vice versa).

---

## 2. Method

### Dataset
- Paired MNIST + SVHN: for each class 0–9, indices are shuffled and zipped up to
  `min(n_mnist, n_svhn)` pairs (~12 k used for training on CPU; full ~60 k for GPU).
- MNIST resized to 32×32 (1 channel); SVHN is already 32×32 (3 channels).
- Both normalised to [−1, 1].

### Architecture
- **Encoder per view** (conv, adapted from the CelebA reference):
  `C_in → 32 → 64 → 128 → 256 → Linear(2·d_z)`, Softplus variance, logvar output.
- **Decoder per view** (transposed conv, Tanh output):
  `d_z → 256 → 128 → 64 → 32 → C_out`.
- Latent dimension: **d_z = 64**.

### Joint posterior — Product of Experts (PoE)
The joint posterior is the product of the two encoder Gaussians and the prior N(0,I):

  1/σ²_* = 1 + Σᵢ 1/σ²ᵢ ,   μ_* = σ²_* · Σᵢ μᵢ/σ²ᵢ

Including the prior expert keeps the posterior regularised when only one view is present.

### Training objective
Three sub-ELBOs are minimised jointly:

  L = −[ ELBO_joint + α·(ELBO_MNIST + ELBO_SVHN) ]

where ELBO_joint uses both decoders from the PoE posterior, and the unimodal ELBOs
use each encoder independently. α = 1.  
A KL warm-up (β: 0 → 1 over the first 10 epochs) prevents posterior collapse.

### Hyperparameters
| Parameter | Value |
|---|---|
| d_z | 64 |
| σ_x (recon. noise) | 0.1 |
| Batch size | 128 |
| Learning rate | 1 × 10⁻³ (Adam) |
| Epochs | 30 |
| α (unimodal weight) | 1.0 |
| KL warmup | 10 epochs |

---

## 3. Results

### Key figures to include in the report

1. **`01_paired_samples.png`** — Show 10 paired examples; comment that labels match.
2. **`03a_mnist_recon.png` + `03b_svhn_recon.png`** — Within-view reconstruction quality.
3. **`05a_cross_m2s.png`** — MNIST → SVHN cross-generation (most interesting result).
4. **`06a_tsne_label.png`** — t-SNE latent space with digit clusters.
5. **`02_loss_curves.png`** — Training curves (optional, for appendix).

### Observations
- MNIST reconstructions are sharp because the MNIST distribution is simpler.
- SVHN reconstructions are blurrier; SVHN has more visual variety (backgrounds, fonts).
- Cross-generation preserves digit identity in most cases; the linear probe accuracy
  on the joint latent space confirms class structure (target >80%).
- t-SNE shows well-separated clusters by digit; MNIST and SVHN unimodal encodings
  partially overlap, confirming the shared space is aligned.

---

## 4. Conclusions and limitations
- The MV-VAE successfully learns a shared latent space for two visually distinct domains.
- Cross-domain generation works qualitatively; SVHN quality is the main bottleneck.
- Future work: β-VAE scheduling for better latent space, more training epochs, or a
  larger SVHN decoder; MoPoE to support missing-modality inference.
