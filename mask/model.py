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
        if module.weight is not None:               # affine-less LN (AdaLN blocks)
            nn.init.ones_(module.weight)
        if module.bias is not None:
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


# ── diffusion head ────────────────────────────────────────────────────────────

class DiffusionHead(nn.Module):
    """
    LatentLM/MAR-style noise-prediction head: ε̂ = head(z_t, t, c).

    Lightweight residual MLP with AdaLN-lite conditioning — each block's
    LayerNorm output is scaled/shifted by a conditioning vector derived from
    (timestep embedding, condition c). c is the pooled predictor output for a
    masked span; gradients flow through c into predictor → encoder.
    """

    T_EMB = 128

    def __init__(self, dim, n_layers=3):
        super().__init__()
        hidden = dim * 2
        self.cond = nn.Sequential(nn.Linear(dim + self.T_EMB, hidden), nn.SiLU())
        self.in_proj = nn.Linear(dim, hidden)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden, elementwise_affine=False)
                                   for _ in range(n_layers))
        self.adas  = nn.ModuleList(nn.Linear(hidden, 2 * hidden) for _ in range(n_layers))
        self.ffns  = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                                 nn.Linear(hidden * 2, hidden))
                                   for _ in range(n_layers))
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, dim)
        # Standard small init (NOT AdaLN-Zero): zero-init modulation would give
        # the condition c zero gradient at init — but c's gradient into the
        # predictor/encoder is the entire point of this head.
        self.apply(_init_weights)

    def _t_embed(self, t):
        half = self.T_EMB // 2
        freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32)
                          * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([ang.sin(), ang.cos()], dim=1)

    def forward(self, z_t, t, c):
        g = self.cond(torch.cat([c, self._t_embed(t)], dim=1))
        h = self.in_proj(z_t)
        for norm, ada, ffn in zip(self.norms, self.adas, self.ffns):
            scale, shift = ada(g).chunk(2, dim=1)
            h = h + ffn(norm(h) * (1 + scale) + shift)
        return self.out(self.out_norm(h))


# ── full model ────────────────────────────────────────────────────────────────

class LeJEPAText(nn.Module):
    """
    forward() returns:
        pred     — (B, L, P) FULL predictor output (masked view); index [mask]
                   for per-token terms, mean-pool for span/global terms
        target   — (M, P)    z_clean at masked positions, DETACHED
        z_clean  — (B, L, P) clean projection output (SIGReg space)
        h_clean  — (B, L, D) clean ENCODER output (geometry logging)
        h_masked — (B, L, D) masked-view ENCODER output (encoder-level MLM head)
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
        self.encoder = TokenEncoder(cfg)
        self.proj    = ProjectionMLP(cfg.d_model, cfg.d_proj)
        # latent_space="encoder": the latent task runs in raw encoder space —
        # predictor built at d_model, proj bypassed on the latent path (it
        # remains constructed for checkpoint compatibility).
        self.latent_space = getattr(cfg, "latent_space", "proj")
        if self.latent_space == "encoder":
            import dataclasses
            self.predictor = SpanPredictor(dataclasses.replace(cfg, d_proj=cfg.d_model))
        else:
            self.predictor = SpanPredictor(cfg)
        self.sigreg_grad_scale = cfg.sigreg_grad_scale

        # MLM anchor head: decodes back to token logits at masked positions.
        # Attach point (cfg.mlm_head):
        #   "pred"    — on predictor output: CE flows through the full
        #               predictor(→proj)→encoder path (anchored JEPA).
        #   "encoder" — on encoder(x_masked) output (D dims): standard BERT-style
        #               MLM; CE shapes the encoder directly (control arm).
        # Only built when mlm_beta > 0 (pure JEPA otherwise).
        # getattr for backward-compat with checkpoints pickled before these fields.
        self.mlm_head = getattr(cfg, "mlm_head", "pred")
        pred_dim = cfg.d_model if self.latent_space == "encoder" else cfg.d_proj
        d_dec = pred_dim if self.mlm_head == "pred" else cfg.d_model

        # Diffusion head (distributional span prediction) — built only when used.
        self.diff_head = (DiffusionHead(pred_dim)
                          if getattr(cfg, "w_diff", 0.0) > 0 else None)
        self.decoder = (nn.Linear(d_dec, cfg.vocab_size)
                        if getattr(cfg, "mlm_beta", 0.0) > 0 else None)
        if self.decoder is not None:
            _init_weights(self.decoder)

    def forward(self, x_clean, x_masked, mask, ema_model=None):
        enc_space = self.latent_space == "encoder"

        # Masked path — always runs through the student (full gradient).
        h_masked = self.encoder(x_masked)              # (B, L, D)
        z_masked = h_masked if enc_space else self.proj(h_masked)
        pred     = self.predictor(z_masked)            # (B, L, P|D) FULL output;
                                                       # callers index [mask] for
                                                       # per-token terms, pool for
                                                       # span/global terms

        if ema_model is not None:
            # EMA path: target comes from the frozen teacher; no gradient.
            with torch.no_grad():
                h_clean = ema_model.encoder(x_clean)
                z_clean = h_clean if enc_space else ema_model.proj(h_clean)
            target = z_clean[mask]                     # already no-grad
        else:
            # No EMA: target is the student's own clean pass (stop-grad for MSE).
            h_clean = self.encoder(x_clean)
            if enc_space:
                z_clean = grad_scale(h_clean, self.sigreg_grad_scale)
            else:
                z_clean = self.proj(grad_scale(h_clean, self.sigreg_grad_scale))
            target = z_clean[mask].detach()

        return pred, target, z_clean, h_clean, h_masked

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
