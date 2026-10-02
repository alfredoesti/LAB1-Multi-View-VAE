# Report — Multi-View VAE (MNIST + SVHN)

**Authors:** Alfredo Estirado Cubero, [Student 2], [Student 3]

**Code:** https://github.com/alfredoesti/LAB1-Multi-View-VAE

---

## 1. Introduction

We trained a multi-view variational autoencoder (MV-VAE) to learn a shared latent space
for two digit datasets: MNIST (grey-scale, 28×28) and SVHN (colour, 32×32).
The model can reconstruct both views from the joint encoding, generate new digit pairs
from the latent prior, and perform cross-domain translation (MNIST → SVHN and SVHN → MNIST).

---

## 2. Method

### Dataset
- Paired MNIST + SVHN: for each class 0–9, indices are shuffled and zipped up to
  `min(n_mnist, n_svhn)` pairs. 12,000 pairs were used for training and 10,000 for testing.
- MNIST is resized to 32×32 (1 channel); SVHN is already 32×32 (3 channels).
- Both are normalised to [−1, 1].

### Architecture
- **Encoder per view** (adapted from the CelebA reference, 4 strided conv layers for 32×32):
  `C_in → 32 → 64 → 128 → 256 → Linear(2·d_z)`, Softplus variance.
- **Decoder per view** (transposed conv, Tanh output):
  `d_z → 256 → 128 → 64 → 32 → C_out`.
- Latent dimension: **d_z = 64**. Total parameters: **2,857,092**.

### Joint posterior — Product of Experts (PoE)
The joint posterior is the product of the two encoder Gaussians and the prior N(0, I):

$$\frac{1}{\sigma_*^2} = 1 + \sum_i \frac{1}{\sigma_i^2}, \qquad \mu_* = \sigma_*^2 \sum_i \frac{\mu_i}{\sigma_i^2}$$

Including the prior expert keeps the posterior regularised when only one view is available at inference time.

### Training objective
Three sub-ELBOs are minimised jointly:

$$\mathcal{L} = -\bigl[\,\text{ELBO}_\text{joint} + \alpha\cdot(\text{ELBO}_\text{MNIST} + \text{ELBO}_\text{SVHN})\bigr]$$

where ELBO_joint uses both decoders from the PoE posterior, and the unimodal ELBOs use
each encoder independently. This ensures each encoder can be used alone for cross-generation.
A KL warm-up (β: 0 → 1 over the first 10 epochs) prevents early posterior collapse.

### Hyperparameters

| Parameter | Value |
|---|---|
| d_z | 64 |
| σ_x (reconstruction noise) | 0.1 |
| Batch size | 128 |
| Learning rate | 1 × 10⁻³ (Adam) |
| Epochs | 30 |
| α (unimodal weight) | 1.0 |
| KL warmup | 10 epochs |

---

## 3. Results

### 3.1 Training

The model was trained on GPU (CUDA T4 via Colab), with each epoch taking approximately 10 seconds.
The total loss decreased from 340 at epoch 5 to −1059 at epoch 30.
The KL warm-up prevented early collapse: the ELBO components became positive after epoch 10
and continued growing steadily until the end of training.

| Epoch | β | Total loss | ELBO_joint | ELBO_MNIST | ELBO_SVHN |
|---|---|---|---|---|---|
| 5  | 0.40 |    340.28 | −167.60 | −344.57 |  171.89 |
| 10 | 0.90 |  −567.32 |  288.68 |  −26.38 |  305.03 |
| 15 | 1.00 |  −804.26 |  405.97 |   31.09 |  367.20 |
| 20 | 1.00 |  −930.52 |  468.01 |   59.15 |  403.36 |
| 25 | 1.00 | −1000.84 |  502.35 |   74.98 |  423.51 |
| 30 | 1.00 | −1058.89 |  531.02 |   85.39 |  442.48 |

### 3.2 Reconstruction

Joint encoding of both views followed by decoding produces sharp MNIST reconstructions
and blurrier but recognisable SVHN reconstructions. The quality difference is expected:
SVHN has higher visual complexity (varied fonts, backgrounds, lighting), making its
decoder harder to optimise.

**Figure:** `03a_mnist_recon.png`, `03b_svhn_recon.png`

### 3.3 Generation from the prior

Sampling z ~ N(0, I) and decoding with both decoders produces plausible digit images.
In most samples, the MNIST and SVHN outputs represent the same digit class, confirming
that the shared latent space has learned a class-structured representation.

**Figure:** `04_prior_samples.png`

### 3.4 Cross-domain generation

- **MNIST → SVHN**: encoding a MNIST image with enc_m and decoding with dec_s produces
  SVHN-style images that preserve the digit identity in most cases.
- **SVHN → MNIST**: encoding a SVHN image with enc_s and decoding with dec_m produces
  recognisable digit strokes, with some noise for inputs with strong background clutter.

**Figure:** `05a_cross_m2s.png`, `05b_cross_s2m.png`

### 3.5 Latent space quality — linear probe

A logistic regression trained on the joint latent means achieves **93.6% accuracy** on the
test set. This result (well above the 80% threshold) confirms that the latent space organises
digit identity in a linearly separable way, despite no explicit classification loss.

### 3.6 t-SNE visualisation

- **By digit label** (joint encoding): well-separated clusters for most digit classes,
  consistent with the 93.6% accuracy. Small overlap between visually similar digits (4, 7, 9).
- **By view** (unimodal encodings): MNIST (blue) and SVHN (red) point clouds show
  substantial overlap, confirming that the two encoders have aligned their representations
  in the shared latent space, as required for cross-domain generation to work.

**Figure:** `06a_tsne_label.png`, `06b_tsne_view.png`

---

## 4. Conclusions and Limitations

The MV-VAE successfully learns a shared latent space for two visually distinct domains.
Cross-domain generation works in both directions and the latent space achieves 93.6%
linear classification accuracy without any supervision.

**Limitations:**
- SVHN reconstruction and cross-generation quality is limited by the visual complexity of
  the dataset. More training epochs or a larger decoder architecture would improve this.
- Only 12,000 training pairs were used (CPU-friendly setting); training on the full ~60,000
  pairs would likely improve results further.
- No explicit disentanglement: digit identity and style share the same latent dimensions.
