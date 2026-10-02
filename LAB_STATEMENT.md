# Lab I: Multi-View VAE (MNIST + SVHN)

Implement a joint (multi-view) VAE to model MNIST and SVHN. Both datasets come from torchvision.

1. Build the database correctly: every sample X is a pair (MNIST image, SVHN image) sharing the same label.
2. Write the multi-view VAE. Reuse the CelebA VAE example (see `reference/`) or any repo; understand the details.
3. Projection into the shared latent space via product of experts (PoE) or mixture of experts (MoE).
4. Show examples of: reconstruction, generation from latent space, and cross-domain generation (MNIST->SVHN and SVHN->MNIST).
5. Visualize the latent space with t-SNE.

Deliverable: two-page report (names of all members at the start) with a link to the code (GitHub/Drive).
Individual or groups up to 3. Worth up to 1.25 points. Deadline: October 23rd (flexible).
