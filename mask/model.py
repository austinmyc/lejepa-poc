"""
Masked latent-prediction model with projection-space SIGReg.

    TokenEncoder   (B, L) tokens   → (B, L, D)   bidirectional transformer, anisotropic OK
    ProjectionMLP  (B, L, D)       → (B, L, P)   maps encoder space → isotropic space
    SpanPredictor  (B, L, P)       → (B, L, P)   light transformer, predicts in proj space
    LeJEPAText     ties all three; forward(x_clean, x_masked, mask)

Two encoder+projection passes share weights:

  Clean path:  z_clean = proj(encoder(x_clean))
    ├─ SIGReg(z_clean)               full grad → pushes PROJECTION toward isotropy
    └─ target = z_clean[mask].detach()   stop-grad → MSE target

  Masked path: z_masked = proj(encoder(x_masked)) → predictor → pred[mask]
    └─ MSE(pred, target)             grad → predictor + proj + encoder(masked)

Both MSE and SIGReg live in projection space (P dims). The encoder is only
constrained indirectly through the projection, so it is free to stay anisotropic.

EVAL read-out: encoder(x).mean(dim=1) (or proj output) for probing/MTEB.
The predictor is not used at eval time.
"""

import torch
import torch.nn as nn


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


# ── gradient scaling ──────────────────────────────────────────────────────────

class _GradScale(torch.autograd.Function):
    """Identity forward; scales the gradient by `alpha` on the backward pass."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return g * ctx.alpha, None


def grad_scale(x, alpha):
    """Pass `x` through unchanged but multiply its gradient by `alpha` (∈ [0, 1])."""
    return _GradScale.apply(x, alpha)


# ── encoder ───────────────────────────────────────────────────────────────────

class TokenEncoder(nn.Module):
    """Bidirectional transformer: (B, L) tokens → (B, L, D)."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop    = nn.Dropout(0.1)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.enc_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.apply(_init_weights)

    def forward(self, x):                              # x: (B, L)
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0)
        h    = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        return self.norm(self.transformer(h))          # (B, L, D)


# ── projection ────────────────────────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    """(B, L, D) → (B, L, P): maps anisotropic encoder space to isotropic space."""

    def __init__(self, d_model, d_proj):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_proj),
        )
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


# ── predictor ─────────────────────────────────────────────────────────────────

class SpanPredictor(nn.Module):
    """Lightweight bidirectional transformer in projection space: (B, L, P) → (B, L, P)."""

    def __init__(self, cfg):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_proj, nhead=max(1, cfg.n_heads // 2),
            dim_feedforward=cfg.d_proj * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.pred_layers)
        self.norm = nn.LayerNorm(cfg.d_proj)
        self.apply(_init_weights)

    def forward(self, x):                              # (B, L, P) → (B, L, P)
        return self.norm(self.transformer(x))


# ── full model ────────────────────────────────────────────────────────────────

class LeJEPAText(nn.Module):
    """
    forward() returns:
        pred    — (M, P)    predictor output at masked positions
        target  — (M, P)    z_clean at masked positions, DETACHED
        z_clean — (B, L, P) clean projection output (SIGReg space)
        h_clean — (B, L, D) clean ENCODER output (for encoder-space geometry logging)
    M = mask.sum().

    `sigreg_grad_scale` (α) throttles how much of SIGReg's gradient reaches the
    encoder, via a grad-scale layer between encoder and projection on the clean
    pass. The projection always receives the full SIGReg gradient; the encoder
    receives α×. α=1 → full (paper-faithful); α=0 → encoder shielded entirely
    (projection does all isotropy shaping, but no collapse insurance on encoder).
    The MSE/prediction path is unaffected — the encoder is still shaped fully by
    prediction through the masked pass.
    """

    def __init__(self, cfg):
        super().__init__()
        self.encoder   = TokenEncoder(cfg)
        self.proj      = ProjectionMLP(cfg.d_model, cfg.d_proj)
        self.predictor = SpanPredictor(cfg)
        self.sigreg_grad_scale = cfg.sigreg_grad_scale

    def forward(self, x_clean, x_masked, mask, ema_model=None):
        # Masked path — always runs through the student (full gradient).
        h_masked = self.encoder(x_masked)
        z_masked = self.proj(h_masked)
        pred     = self.predictor(z_masked)[mask]      # (M, P)

        if ema_model is not None:
            # EMA path: target comes from the frozen teacher; no gradient.
            with torch.no_grad():
                h_clean = ema_model.encoder(x_clean)
                z_clean = ema_model.proj(h_clean)
            target = z_clean[mask]                     # (M, P) — already no-grad
        else:
            # No EMA: target is the student's own clean pass (stop-grad for MSE).
            h_clean = self.encoder(x_clean)
            z_clean = self.proj(grad_scale(h_clean, self.sigreg_grad_scale))
            target  = z_clean[mask].detach()

        return pred, target, z_clean, h_clean

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
