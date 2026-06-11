"""
Collapse diagnostics for embedding quality monitoring.

Run every N steps during training to catch representation collapse early.

REVIEW: Understand both functions before the first training run.
Collapse is the most common failure mode in SSL — if you don't catch it
early you waste the entire compute budget.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def embedding_rank(embs: torch.Tensor, thresh: float = 0.99) -> int:
    """
    Effective rank of a batch of embeddings.

    Counts how many principal components (singular values) are needed to
    explain `thresh` fraction of the total variance. This is identical to
    PCA's "number of components for 99% explained variance".

    REVIEW: Three steps to understand:
      1. Center: subtract the mean so we're measuring spread, not offset.
      2. SVD: decomposes the centered matrix into singular values s.
         s[i]^2 is proportional to variance explained by component i.
         SVD is equivalent to PCA (via the covariance matrix eigenvectors).
      3. Cumulative variance: find how many components sum to >= thresh.

    What to expect:
      - Random init:   rank ≈ D (all dims used equally, high rank)
      - Trained well:  rank is moderate — structured, not collapsed
      - Collapsed:     rank = 1 or 2 (all embeddings nearly identical)

    Args:
        embs:   (N, D) float tensor — a batch of embeddings
        thresh: fraction of variance to explain (default 0.99)

    Returns:
        Integer effective rank.
    """
    # REVIEW: why .float()? SVD can be numerically unstable in bf16/fp16.
    # Always cast to float32 for this computation regardless of training dtype.
    embs = embs.float()
    embs = embs - embs.mean(dim=0)                        # center

    _, s, _ = torch.linalg.svd(embs, full_matrices=False) # s: (min(N,D),)

    # REVIEW: s^2 = variance explained by each component (unnormalised).
    # cumsum gives running total; dividing by total gives fraction.
    cumvar = (s ** 2).cumsum(0) / (s ** 2).sum()

    # REVIEW: (cumvar < thresh) is True for components we still need.
    # .sum() counts them; +1 because the first component that crosses
    # the threshold is also needed.
    rank = int((cumvar < thresh).sum().item()) + 1
    return rank


@torch.no_grad()
def mean_cosine_sim(embs: torch.Tensor, n_sample: int = 512) -> float:
    """
    Mean pairwise cosine similarity across a random subsample of embeddings.

    REVIEW: Cosine similarity measures the angle between two vectors,
    ignoring magnitude. Range: -1 (opposite) to +1 (identical direction).

    What to expect:
      - Healthy:   mean close to 0.0 (embeddings point in varied directions)
      - Collapsed: mean close to 1.0 (all embeddings point the same way)

    This catches a different failure mode than rank:
      - Rank collapse: embeddings live in a low-dimensional subspace
      - Cosine collapse: embeddings all point in the same direction within
        that subspace (rank could be > 1 but cosine sim still high)

    Args:
        embs:     (N, D) tensor
        n_sample: subsample size to keep compute cheap

    Returns:
        Float in [-1, 1].
    """
    if embs.shape[0] > n_sample:
        idx  = torch.randperm(embs.shape[0], device=embs.device)[:n_sample]
        embs = embs[idx]

    # REVIEW: F.normalize divides each vector by its L2 norm → unit vectors.
    # After this, dot product between two rows = cosine similarity.
    embs = F.normalize(embs.float(), dim=-1)              # (N, D), unit vectors
    sim  = embs @ embs.T                                  # (N, N) cosine sim matrix

    # REVIEW: exclude diagonal (similarity of a vector with itself = 1.0 always).
    # Including it would inflate the mean artificially.
    mask = ~torch.eye(embs.shape[0], dtype=torch.bool, device=embs.device)
    return sim[mask].mean().item()


def log_diagnostics(chunk_embs: torch.Tensor,
                    proj_detached: torch.Tensor,
                    step: int,
                    wandb=None) -> dict:
    """
    Compute and print collapse diagnostics. Call every cfg.rank_every steps.

    REVIEW: We track TWO ranks with the same goal — want both HIGH (not collapsed):
      enc_rank  — encoder output rank. Want HIGH (diverse, non-collapsed representations).
      proj_rank — projection output rank. Want HIGH (SIGReg pushing toward isotropy).

    If proj_rank collapses → increase lam (SIGReg weight).
    If enc_rank collapses (→ 1 or 2) → representation collapse, check target stop-gradient.
    """
    enc_flat  = chunk_embs.reshape(-1, chunk_embs.shape[-1]).detach()
    proj_flat = proj_detached.reshape(-1, proj_detached.shape[-1]).detach()

    enc_rank  = embedding_rank(enc_flat)
    proj_rank = embedding_rank(proj_flat)
    cos       = mean_cosine_sim(enc_flat)

    print(f"  → enc_rank: {enc_rank}  |  proj_rank: {proj_rank}  |  cos_sim: {cos:.4f}")

    if proj_rank <= 2:
        print("  ⚠️  proj_rank collapsed — increase lam")
    if enc_rank <= 2:
        print("  ⚠️  enc_rank collapsed — representation collapse, check target stop-gradient")

    metrics = {"enc_rank": enc_rank, "proj_rank": proj_rank, "mean_cos_sim": cos}
    if wandb is not None:
        wandb.log(metrics, step=step)
    return metrics
