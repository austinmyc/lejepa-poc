"""
Collapse diagnostics for embedding quality monitoring.

Run every N steps during training to catch representation collapse early.
Collapse is the most common failure mode in SSL — if you don't catch it
early you waste the entire compute budget.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def embedding_rank(embs: torch.Tensor, thresh: float = 0.99) -> int:
    """
    Effective rank: how many principal components explain `thresh` of variance.

    Collapse reads:
      rank = 1-2  → representational collapse (all tokens embed alike)
      rank = D    → random init (every dimension used equally)
      rank moderate, stable → healthy training
    """
    embs = embs.float()
    embs = embs - embs.mean(dim=0)
    _, s, _ = torch.linalg.svd(embs, full_matrices=False)
    cumvar = (s ** 2).cumsum(0) / (s ** 2).sum()
    return int((cumvar < thresh).sum().item()) + 1


@torch.no_grad()
def mean_cosine_sim(embs: torch.Tensor, n_sample: int = 512) -> float:
    """
    Mean pairwise cosine similarity across a random subsample.

    Collapse reads:
      ~0.0  → healthy (embeddings point in varied directions)
      ~1.0  → directional collapse (all embeddings point the same way)
    """
    if embs.shape[0] > n_sample:
        idx  = torch.randperm(embs.shape[0], device=embs.device)[:n_sample]
        embs = embs[idx]

    embs = F.normalize(embs.float(), dim=-1)
    sim  = embs @ embs.T
    mask = ~torch.eye(embs.shape[0], dtype=torch.bool, device=embs.device)
    return sim[mask].mean().item()


def log_diagnostics(clean_embs: torch.Tensor, step: int, wandb=None) -> dict:
    """
    Compute and print collapse diagnostics. Call every cfg.rank_every steps.

    Args:
        clean_embs: (B, L, D) — clean encoder output from the current batch
        step:       current training step
        wandb:      wandb module if logging, else None

    Monitors:
        enc_rank  — effective rank of token embeddings (want HIGH, not collapsed)
        cos_sim   — mean pairwise cosine similarity (want LOW, near 0)

    If enc_rank ≤ 2: representational collapse — increase lam or check SIGReg gradient path.
    If cos_sim  > 0.9: directional collapse — same action.
    """
    flat = clean_embs.reshape(-1, clean_embs.shape[-1]).detach()

    enc_rank = embedding_rank(flat)
    cos      = mean_cosine_sim(flat)

    print(f"  → enc_rank: {enc_rank}  |  cos_sim: {cos:.4f}")

    if enc_rank <= 2:
        print("  ⚠️  enc_rank collapsed — increase lam (SIGReg weight)")
    if cos > 0.9:
        print("  ⚠️  high cosine sim — directional collapse, check SIGReg gradient path")

    metrics = {"enc_rank": enc_rank, "mean_cos_sim": cos}
    if wandb is not None:
        wandb.log(metrics, step=step)
    return metrics
