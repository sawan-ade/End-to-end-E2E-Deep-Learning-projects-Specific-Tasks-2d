"""
models/autoencoder.py

Two autoencoder variants built on the shared SparseEncoder/SparseDecoder:

  BaselineAE  — standard MSE reconstruction, used for Step 1 pretraining.
  SparseAE    — adds an L1 penalty on bottleneck activations (bonus task),
                encouraging most neurons to be silent for any given input.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sparse_cnn import SparseEncoder, SparseDecoder


# ── Baseline autoencoder 

class BaselineAE(nn.Module):

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


# ── Sparse autoencoder

class SparseAE(nn.Module):
   
    def __init__(self, in_ch: int = 2, sparsity_weight: float = 1e-3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.encoder         = SparseEncoder(in_ch)
        self.sparse_proj     = nn.Linear(128, 256)   
        self.compress        = nn.Linear(256, 128)
        self.decoder         = SparseDecoder(in_ch)

    def forward(self, x: torch.Tensor):
        z        = self.encoder(x)
        z_sparse = F.relu(self.sparse_proj(z))                  
        s_loss   = self.sparsity_weight * z_sparse.abs().mean()  
        recon    = self.decoder(self.compress(z_sparse))
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(
                recon, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return recon, z_sparse, s_loss


# ── Classifier

class Classifier(nn.Module):
     
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
