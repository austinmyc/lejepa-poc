"""
LeJEPA-Text model — token-level masked latent prediction, no EMA.

Architecture (aligned with research plan §3b):
    TokenEncoder   — bidirectional transformer over full (B, L) token sequence → (B, L, D)
    SpanPredictor  — lightweight bidirectional transformer; given the encoder output
                     of the MASKED sequence, predicts clean encoder output at masked positions
    LeJEPAText     — ties both; forward() takes (x_clean, x_masked, mask)

Gradient routing
----------------
Two encoder forward passes share the same weights:

  Clean path:  encoder(x_clean) → clean_embs
    ├─ SIGReg(clean_embs)       full gradient → trains encoder toward isotropy
    └─ MSE target = clean_embs.detach()[mask]   stop-grad → no MSE gradient from clean path

  Masked path: encoder(x_masked) → masked_embs → predictor → pred_M
    └─ MSE(pred_M, stop-grad target)   gradient → predictor + encoder(masked)

The encoder is updated by BOTH signals:
  - Prediction signal  (masked path, MSE)
  - Isotropy signal    (clean path, SIGReg)

SIGReg replaces EMA as the collapse-prevention mechanism.  This is the key
architectural choice: without EMA, the only thing preventing representational
collapse is SIGReg pushing clean_embs toward an isotropic Gaussian.

EVAL read-out (§3b of research plan)
--------------------------------------
Use encoder(x).mean(dim=1) to get a single sequence embedding for MTEB/probing.
The predictor is not used at eval time.
"""

import torch
import torch.nn as nn


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

class TokenEncoder(nn.Module):
    """
    Bidirectional transformer over a full token sequence.

    Unlike the old ChunkEncoder, there is NO chunking — the encoder attends
    over all L tokens simultaneously with bidirectional (non-causal) attention.
    This is required for the masked-span objective: unmasked context tokens
    must see each other across the full sequence to build rich contextual
    representations at masked positions.

    Input:  (B, L)    — token ids
    Output: (B, L, D) — contextual token embeddings, one per token
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-norm: more stable
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_weights)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)   # (1, L)
        h   = self.drop(self.tok_emb(x) + self.pos_emb(pos))  # (B, L, D)
        h   = self.transformer(h)                               # (B, L, D)
        return self.norm(h)                                     # (B, L, D)


# ── predictor ─────────────────────────────────────────────────────────────────

class SpanPredictor(nn.Module):
    """
    Lightweight bidirectional transformer that refines encoder output at masked positions.

    Takes the encoder output of the MASKED sequence (where masked tokens were
    replaced by mask_token_id before encoding). The encoder already propagates
    context from unmasked positions to masked positions via attention; this
    predictor adds a dedicated pass to further refine those estimates.

    We run over the FULL (B, L, D) sequence and select masked positions afterward —
    so the predictor can attend to unmasked context in its own pass.

    Input:  (B, L, D) — encoder output of masked sequence
    Output: (B, L, D) — predictions at all positions (caller selects masked subset)
    """

    def __init__(self, d_model, n_heads, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout)

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
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        h   = self.drop(x + self.pos_emb(pos))                # (B, L, D)
        h   = self.transformer(h)                              # (B, L, D)
        h   = self.norm(h)
        return self.proj(h)                                    # (B, L, D)


# ── full model ────────────────────────────────────────────────────────────────

class LeJEPAText(nn.Module):
    """
    EMA-free JEPA for token sequences.

    No target network. No EMA. No DINO tricks.
    SIGReg on the clean encoder output is the sole collapse-prevention mechanism.

    forward() takes THREE tensors:
        x_clean  — original token ids (B, L)
        x_masked — x_clean with mask_token_id at masked positions (B, L)
        mask     — bool (B, L), True at masked positions

    and returns THREE tensors for the training loop:
        pred_M      — (M, D)    predictor output at masked positions
        target_M    — (M, D)    clean encoder output at masked positions, DETACHED
        clean_embs  — (B, L, D) clean encoder output, full gradient (for SIGReg)

    M = total number of masked tokens across the batch = mask.sum().
    """

    def __init__(self, cfg):
        super().__init__()
        self.encoder   = TokenEncoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.enc_layers,
            seq_len=cfg.seq_len,
        )
        self.predictor = SpanPredictor(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.pred_layers,
            seq_len=cfg.seq_len,
        )

    def forward(self, x_clean, x_masked, mask):
        """
        x_clean:  (B, L) — original token ids
        x_masked: (B, L) — same with mask_token_id at mask==True positions
        mask:     (B, L) bool — True at positions to predict

        Gradient routing:
          encoder(x_clean) → clean_embs
            ├─ SIGReg(clean_embs)          full grad: trains encoder toward isotropy
            └─ target_M = .detach()[mask]  stop-grad: no MSE gradient from clean path

          encoder(x_masked) → predictor → pred_all
            └─ pred_M = pred_all[mask]
               MSE(pred_M, target_M)       grad: trains predictor + encoder(masked)
        """
        # ── clean path ────────────────────────────────────────────────────
        clean_embs = self.encoder(x_clean)          # (B, L, D) — full grad for SIGReg
        target_M   = clean_embs[mask].detach()      # (M, D)    — stop-grad for MSE

        # ── masked path ───────────────────────────────────────────────────
        masked_embs = self.encoder(x_masked)        # (B, L, D)
        pred_all    = self.predictor(masked_embs)   # (B, L, D)
        pred_M      = pred_all[mask]                # (M, D)

        return pred_M, target_M, clean_embs

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
