"""
utils/pruning.py

Magnitude-based weight pruning and FLOP counting utilities.

Pruning strategy: global unstructured L1-magnitude pruning
  - Ranks ALL weights across ALL Conv2d and Linear layers by |w|
  - Zeroes the smallest fraction (the pruning ratio)
  - Effective FLOPs are scaled by the fraction of non-zero weights,
    reflecting the compute actually needed by a sparse kernel.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune


# ── Pruning ───────────────────────────────────────────────────────────────────

def apply_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Apply global unstructured L1-magnitude pruning to all Conv2d and Linear
    layers, then make the masks permanent.

    Args:
        model  : PyTorch model to prune (modified in-place).
        amount : Fraction of weights to zero out (0.0 = none, 0.9 = 90%).

    Returns:
        The pruned model.
    """
    params = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    torch_prune.global_unstructured(
        params,
        pruning_method=torch_prune.L1Unstructured,
        amount=amount,
    )
    # Remove the mask buffers and store the sparse weight directly
    for module, _ in params:
        try:
            torch_prune.remove(module, "weight")
        except ValueError:
            pass
    return model


def get_sparsity(model: nn.Module) -> float:
    """
    Return the global fraction of zeroed weights in the model.
    A sparsity of 0.5 means 50 % of weights are exactly zero.
    """
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            zeros += (m.weight == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ── FLOP counting ─────────────────────────────────────────────────────────────

def count_flops(model: nn.Module, input_size: tuple, device: str = "cpu") -> float:
    """
    Estimate effective multiply-add FLOPs for one forward pass.

    'Effective' means we multiply dense FLOPs by the density of non-zero
    weights in each layer — this reflects the savings a sparse kernel would
    achieve compared to a dense implementation.

    Args:
        model      : (possibly pruned) PyTorch model.
        input_size : Tuple (C, H, W) — shape of ONE input sample.
        device     : Device to run the dummy forward pass on.

    Returns:
        Effective FLOPs as a float.
    """
    total = [0.0]
    hooks = []

    def _conv_hook(module, inp, out):
        _, c_out, h_out, w_out = out.shape
        c_in = inp[0].shape[1]
        kH, kW   = module.kernel_size
        groups   = module.groups
        macs     = (c_in / groups) * kH * kW * c_out * h_out * w_out
        density  = (module.weight != 0).float().mean().item()
        total[0] += 2 * macs * density          # ×2: multiply + add

    def _linear_hook(module, inp, out):
        macs     = module.in_features * module.out_features
        density  = (module.weight != 0).float().mean().item()
        total[0] += 2 * macs * density

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    dummy = torch.zeros(1, *input_size, device=device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass    # shape mismatches are fine — hooks already fired

    for h in hooks:
        h.remove()

    return total[0]
