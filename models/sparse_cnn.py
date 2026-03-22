"""
models/sparse_cnn.py

Submanifold Sparse Convolutional Network for event-based data.

Architecture based on:
    Graham & van der Maaten, "Submanifold Sparse Convolutional Networks"
    arXiv:1706.01307

Key idea — Valid Sparse Convolution (VSC):
    In standard convolutions every output site becomes active if ANY input
    in the receptive field is non-zero.  After a few layers, the initially
    sparse activation map is almost completely dense.

    VSC restricts output activity: a site is active only when the CENTRAL
    input of its receptive field is already active.  This keeps the sparsity
    pattern constant across layers, so deep networks remain efficient.

Implementation note:
    We emulate VSC with dense PyTorch ops + explicit binary masks.
    A production deployment would use spconv or MinkowskiEngine for the
    actual sparse memory savings; here the mask makes the semantics clear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VSCBlock(nn.Module):
    """
    One Valid Sparse Convolution block:
      1. Build binary mask from active (non-zero) input sites.
      2. Zero the ground state before convolution (VSC requirement).
      3. Apply Conv2d → BN → ReLU.
      4. Propagate the mask so inactive sites stay silent in the output.

    Args:
        in_ch  : number of input feature channels
        out_ch : number of output feature channels
        kernel : convolution kernel size (default 3)
        stride : convolution stride     (default 1)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=stride, padding=kernel // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Which spatial sites are active?
        mask = (x.abs().max(dim=1, keepdim=True).values > 0).float()

        # Zero inactive sites (ground-state substitution)
        out = F.relu(self.bn(self.conv(x * mask)), inplace=True)

        # Propagate mask to output resolution
        if self.conv.stride[0] > 1:
            out_mask = F.max_pool2d(mask, self.conv.stride[0], self.conv.stride[0])
        else:
            out_mask = mask
        out_mask = (
            F.interpolate(out_mask, size=out.shape[-2:], mode="nearest") > 0
        ).float()

        return out * out_mask


class SparseEncoder(nn.Module):
    """
    Four-stage VSC encoder followed by global average pooling.

    Input  : (B, C, H, W)  — sparse event image
    Output : (B, 128)       — latent feature vector

    Stage channels: C → 16 → 32 → 64 → 128, each stage halved by MaxPool2d.
    """

    def __init__(self, in_ch: int = 2):
        super().__init__()
        self.stage1 = nn.Sequential(VSCBlock(in_ch, 16), VSCBlock(16, 16))
        self.pool1  = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(VSCBlock(16, 32), VSCBlock(32, 32))
        self.pool2  = nn.MaxPool2d(2, 2)

        self.stage3 = nn.Sequential(VSCBlock(32, 64), VSCBlock(64, 64))
        self.pool3  = nn.MaxPool2d(2, 2)

        self.stage4 = nn.Sequential(VSCBlock(64, 128), VSCBlock(128, 128))

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        return self.global_pool(x).flatten(1)   # (B, 128)


class SparseDecoder(nn.Module):
    """
    Mirror decoder: 128-D latent code → (C, H, W) reconstruction.
    Used only during autoencoder pretraining (Step 1).

    Input  : (B, 128)
    Output : (B, C, H, W)
    """

    def __init__(self, out_ch: int = 2):
        super().__init__()
        self.proj = nn.Linear(128, 128 * 4 * 4)
        self.net  = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),  nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).view(z.size(0), 128, 4, 4)
        return self.net(x)
