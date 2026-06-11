"""
Optimizer and learning rate schedule.

REVIEW: Two things to understand here:
  1. Why the LR schedule is shaped the way it is (warmup + cosine decay)
  2. Why different parameter types use different settings

TODO (L20): replace build_optimizer with Muon from nanochat optim.py.
     Copy nanochat/nanochat/optim.py here, then swap the AdamW call below
     for Muon on the linear weight matrices and AdamW on everything else.
     The parameter group split in build_optimizer already prepares for this.
"""

import math
import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    betas: tuple) -> torch.optim.Optimizer:
    """
    Build AdamW with two parameter groups.

    REVIEW: Not all parameters should have weight decay.
    Weight decay penalises large weights (L2 regularisation). It helps
    for weight matrices (prevents overfitting) but hurts for:
      - Biases: no reason to shrink them toward zero
      - LayerNorm weights/biases: their scale has a specific meaning
      - Embeddings: shrinking them distorts the token/position space

    So we split into two groups:
      decay group:    all weight matrices (nn.Linear weights)
      no-decay group: biases, LayerNorm params, embedding params

    REVIEW: This split also matters when swapping to Muon later.
    Muon only applies to weight matrices (it uses spectral methods that
    assume a 2D matrix). Embeddings and LayerNorm use AdamW regardless.
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # REVIEW: 1D tensors are biases, LN weights/biases, embedding vectors.
        # Weight matrices are always 2D or higher.
        if param.ndim == 1 or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def get_lr(step: int, max_steps: int, warmup_steps: int, lr: float) -> float:
    """
    Linear warmup followed by cosine decay to lr/10.

    REVIEW: Two phases:
      1. Warmup (steps 0 → warmup_steps):
         LR increases linearly from 0 to `lr`.
         Why: large LR at step 0 with random weights causes instability.
         Gradients are huge and random — a small LR prevents early divergence.

      2. Cosine decay (warmup_steps → max_steps):
         LR follows a cosine curve from `lr` down to `lr * 0.1`.
         Why cosine: smooth decay avoids a sharp LR drop that can destabilise
         training. The minimum is 0.1× not 0 — completely zeroing LR at the
         end wastes the final steps.

    REVIEW: `progress` goes from 0 → 1 as training proceeds.
    cos(0) = 1, cos(π) = -1, so:
      0.5 * (1 + cos(π * progress)) goes from 1 → 0
    Scaled: 0.1 + 0.9 * that expression goes from 1.0 → 0.1
    Multiplied by lr gives the final schedule.
    """
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
