"""
LeJEPA-Text model.

ChunkEncoder   – bidirectional transformer, mean-pools each 64-token chunk → 1 vector
ChunkPredictor – causal transformer over the sequence of chunk vectors
LeJEPAText     – combines both; forward() returns (pred_embs, target_embs, all_chunk_embs)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ── encoder ───────────────────────────────────────────────────────────────────

class ChunkEncoder(nn.Module):
    """
    Encodes each chunk independently with bidirectional attention, then mean-pools.

    Input:  (B, N, C)  — B=batch, N=num_chunks, C=chunk_size (token ids)
    Output: (B, N, D)  — one embedding per chunk
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, chunk_size, dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(chunk_size, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm: more stable
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.apply(_init_weights)

    def forward(self, x):
        B, N, C = x.shape
        # flatten chunks → (B*N, C) so each chunk is one sequence
        x_flat = x.reshape(B * N, C)

        pos = torch.arange(C, device=x.device).unsqueeze(0)        # (1, C)
        h = self.drop(self.tok_emb(x_flat) + self.pos_emb(pos))    # (B*N, C, D)

        # bidirectional — no mask
        h = self.transformer(h)          # (B*N, C, D)
        h = self.norm(h)

        # mean pool over token dimension
        emb = h.mean(dim=1)              # (B*N, D)
        return emb.reshape(B, N, -1)    # (B, N, D)


# ── predictor ─────────────────────────────────────────────────────────────────

class ChunkPredictor(nn.Module):
    """
    Causal transformer over a sequence of chunk embeddings.
    Predicts embedding[t] from embeddings[0..t-1].

    Input:  (B, N, D)  — context chunk embeddings (all but last)
    Output: (B, N, D)  — predicted next-chunk embeddings
    """

    def __init__(self, d_model, n_heads, n_layers, max_chunks, dropout=0.1):
        super().__init__()

        self.pos_emb = nn.Embedding(max_chunks, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.apply(_init_weights)

    def forward(self, x):
        B, N, D = x.shape
        pos = torch.arange(N, device=x.device).unsqueeze(0)
        h = self.drop(x + self.pos_emb(pos))

        # causal mask so position t can only attend to 0..t
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.proj(h)


# ── projection MLP ────────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """
    Maps encoder output to a more isotropic space before the predictor.
    SIGReg is applied in this space; the stop-gradient ensures SIGReg
    trains only this MLP, not the encoder.

    Input/Output: (*, D)
    """

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


# ── full model ────────────────────────────────────────────────────────────────

class LeJEPAText(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = ChunkEncoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.enc_layers,
            chunk_size=cfg.chunk_size,
        )
        self.projection = ProjectionMLP(
            d_model=cfg.d_model,
            hidden_dim=cfg.proj_hidden,
        )
        self.predictor = ChunkPredictor(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.pred_layers,
            max_chunks=cfg.max_chunks,
        )

    def forward(self, x):
        """
        x: (B, N, C)  token ids

        Gradient flows
        --------------
        MSE path: MSE → predictor → proj(context chunks) → encoder
                  target chunks are detached — prevents co-collapse
        SIGReg path: SIGReg → projection only (encoder stop-grad)

        Returns
        -------
        pred_embs        : (B, N-1, D)  predictor output (in projected space)
        target_proj      : (B, N-1, D)  attached projection of target chunks (MSE target)
        proj_detached    : (B, N,   D)  detached projection of all chunks (SIGReg input)
        chunk_embs       : (B, N,   D)  raw encoder output (for downstream eval)
        """
        chunk_embs = self.encoder(x)                          # (B, N, D)

        # MSE path — gradient flows through predictor/encoder via context chunks only;
        # target is detached so predictions and targets can't co-collapse.
        proj_attached = self.projection(chunk_embs)                    # (B, N, D)
        pred_embs     = self.predictor(proj_attached[:, :-1])          # (B, N-1, D)
        target_proj   = proj_attached[:, 1:].detach()                  # (B, N-1, D)

        # SIGReg path — stop-gradient: encoder is not updated by SIGReg
        proj_detached = self.projection(chunk_embs.detach())  # (B, N, D)

        return pred_embs, target_proj, proj_detached, chunk_embs

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
