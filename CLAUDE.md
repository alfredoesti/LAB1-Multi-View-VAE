# Project: Multi-View VAE (MNIST + SVHN) — UC3M Deep Learning Lab I

See LAB_STATEMENT.md for the assignment. Deliverables: notebook `notebooks/Lab_MultiView_VAE.ipynb` + 2-page report in `report/`.

## Conventions
- PyTorch; reuse architecture ideas from `reference/Lab_VAEs_solved_CelebA.ipynb` (conv encoder/decoder, softplus variance, reparameterization, Gaussian ELBO with sigma_x).
- Paired dataset: for each label 0-9, randomly pair an MNIST image with an SVHN image of the same label (SVHN label 10 / '0' handling: torchvision >=0.x already maps to 0-9). Resize MNIST to 32x32 and replicate to 3 channels, or use separate decoders per view with native shapes.
- One encoder + one decoder per view; joint posterior via Product of Experts (include prior expert) or MoE.
- Train with sub-ELBOs (joint + each unimodal) so cross-generation works.
- Required outputs: reconstructions, prior samples, cross-domain generation, t-SNE of latent space colored by label and by view.
- Keep code in complete, runnable cells/files; data goes to `data/` (git-ignored), checkpoints to `checkpoints/`.
- Must run on Colab GPU as well as locally.
- Code comments in English; the user (Alfredo) prefers concise answers.
