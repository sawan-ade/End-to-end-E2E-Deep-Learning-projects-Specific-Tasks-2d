"""
models/autoencoder.py

Two autoencoder variants built on the shared SparseEncoder/SparseDecoder:

  BaselineAE  — standard MSE reconstruction, used for Step 1 pretraining.
  SparseAE    — adds an L1 penalty on bottleneck activations (bonus task),
                encouraging most neurons to be silent for any given input.

Both expose a `Classifier` that reuses the pretrained encoder for the
binary classification task in Step 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sparse_cnn import SparseEncoder, SparseDecoder


# ── Baseline autoencoder (Step 1) ────────────────────────────────────────────

class BaselineAE(nn.Module):
    """
    Standard sparse autoencoder trained with MSE reconstruction loss.

    Forward returns (reconstruction, latent_code) so the training loop
    can compute  loss = F.mse_loss(recon, x).
    """

    def __init__(self, in_ch: int = 2):
        super().__init__()
        self.encoder = SparseEncoder(in_ch)
        self.decoder = SparseDecoder(in_ch)

    def forward(self, x: torch.Tensor):
        z    = self.encoder(x)
        recon = self.decoder(z)
        # Resize if the decoder output doesn't match the input resolution
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon, z


# ── Sparse autoencoder (Bonus) ────────────────────────────────────────────────

class SparseAE(nn.Module):
    """
    Autoencoder with an L1-penalised bottleneck (sparse coding).

    A wider linear layer (128 → 256) with ReLU sits between the encoder
    and decoder.  The L1 penalty on its activations pushes most units to
    zero for any given input, producing truly sparse codes.

    Forward returns (reconstruction, sparse_code, sparsity_loss).
    Total loss = F.mse_loss(recon, x) + sparsity_loss.
    """

    def __init__(self, in_ch: int = 2, sparsity_weight: float = 1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.encoder         = SparseEncoder(in_ch)
        self.sparse_proj     = nn.Linear(128, 256)   # wider bottleneck
        self.compress        = nn.Linear(256, 128)
        self.decoder         = SparseDecoder(in_ch)

    def forward(self, x: torch.Tensor):
        z        = self.encoder(x)
        z_sparse = F.relu(self.sparse_proj(z))                   # (B, 256)
        s_loss   = self.sparsity_weight * z_sparse.abs().mean()  # L1 penalty
        recon    = self.decoder(self.compress(z_sparse))
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon, z_sparse, s_loss


# ── Classifier (Step 2) ───────────────────────────────────────────────────────

class Classifier(nn.Module):
    """
    Binary classifier: SparseEncoder + two-layer MLP head.

    The encoder weights are initialised from a pretrained BaselineAE or
    SparseAE via `load_encoder()`.  The head is always trained from scratch.

    Forward returns a raw logit (B, 1).  Use F.binary_cross_entropy_with_logits
    as the loss function.
    """

    def __init__(self, in_ch: int = 2):
        super().__init__()
        self.encoder = SparseEncoder(in_ch)
        self.head    = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def load_encoder(self, ae: nn.Module):
        """Copy encoder weights from a pretrained autoencoder."""
        self.encoder.load_state_dict(ae.encoder.state_dict())
        print("Encoder weights transferred from pretrained autoencoder.")
