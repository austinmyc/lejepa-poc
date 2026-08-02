"""
DLLM-JEPA model: masked-diffusion backbone + JEPA predictor (+ CoT action).

    L_diff  = LLaDA SFT masked-token CE, reweighted by 1/t, response tokens only
    L_JEPA  = 1 − cos( sg(Pool(f_θ'(x_tH))), g_φ(Pool(f_θ(x_tL))) )

EMA teacher, memory note: the paper full-finetunes and keeps a full EMA copy of
an 8B backbone (a second ~16GB of weights). Under LoRA the base weights are
FROZEN and identical for student and teacher, so the teacher only needs an EMA of
the LoRA deltas — implemented here by EMA-ing the LoRA parameters into a shadow
copy and swapping them in for the no-grad target pass. Same semantics as a full
EMA teacher over the *trainable* parameters, at a few MB instead of 16GB.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_states(h, valid):
    """Paper's pooling: mean over non-masked, non-pad tokens. (B,L,D)+(B,L)→(B,D).
    LayerNorm is applied by the caller (a learned/parameter-free LN over D)."""
    w = valid.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)


class AdaLNBlock(nn.Module):
    """Transformer block whose LayerNorms are modulated by an action vector.
    action=None → plain pre-LN block (the A1 / no-action reproduction path)."""

    def __init__(self, d, n_heads, act_dim=None):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=act_dim is None)
        self.n2 = nn.LayerNorm(d, elementwise_affine=act_dim is None)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(act_dim, 4 * d) if act_dim else None
        if self.ada is not None:                     # zero-init → starts as identity
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, a=None):
        if self.ada is not None and a is not None:
            s1, b1, s2, b2 = self.ada(a).chunk(4, dim=-1)
            s1, b1, s2, b2 = (v.unsqueeze(1) for v in (s1, b1, s2, b2))
        else:
            s1 = b1 = s2 = b2 = None
        h = self.n1(x) if s1 is None else (1 + s1) * self.n1(x) + b1
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.n2(x) if s2 is None else (1 + s2) * self.n2(x) + b2
        return x + self.ff(h)


class Predictor(nn.Module):
    """g_φ: k transformer layers, hidden dim = backbone's, randomly initialised.
    Operates on the pooled state (sequence of length 1) — matching the paper's
    z_tL → ẑ_tH mapping — optionally conditioned on an action vector."""

    def __init__(self, d, n_layers=2, n_heads=8, act_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList(AdaLNBlock(d, n_heads, act_dim) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, z, a=None):                    # z: (B, D)
        x = z.unsqueeze(1)                           # (B, 1, D)
        for blk in self.blocks:
            x = blk(x, a)
        return self.norm(x.squeeze(1))


class DLLMJepa(nn.Module):
    def __init__(self, backbone, cfg, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        act_dim = hidden_size if cfg.action != "none" else None
        n_heads = max(1, min(8, hidden_size // 64))
        self.predictor = Predictor(hidden_size, cfg.pred_layers, n_heads, act_dim)
        # Parameter-free LN on pooled states (paper: "mean pooling ... followed
        # by LayerNorm"); parameter-free keeps student/teacher pooling identical.
        self.pool_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self._ema = None                             # lazily built shadow LoRA

    # ── EMA teacher over trainable (LoRA) params ───────────────────────────
    def _trainable(self):
        return [(n, p) for n, p in self.backbone.named_parameters() if p.requires_grad]

    @torch.no_grad()
    def ema_init(self):
        self._ema = {n: p.detach().clone() for n, p in self._trainable()}

    @torch.no_grad()
    def ema_update(self):
        if self._ema is None:
            self.ema_init(); return
        for n, p in self._trainable():
            self._ema[n].mul_(self.cfg.ema_decay).add_(p.detach(), alpha=1 - self.cfg.ema_decay)

    @torch.no_grad()
    def _swap_ema(self):
        """Swap EMA weights in (and current weights out) — call twice to restore."""
        if self._ema is None:
            self.ema_init()
        for n, p in self._trainable():
            tmp = p.detach().clone()
            p.data.copy_(self._ema[n])
            self._ema[n] = tmp

    # ── forward pieces ─────────────────────────────────────────────────────
    def hidden(self, ids, attn_mask=None):
        """→ (logits, hidden_states[cfg.hidden_layer]). One backbone pass serves
        both the diffusion CE and the JEPA state."""
        out = self.backbone(input_ids=ids, attention_mask=attn_mask,
                            output_hidden_states=True)
        return out.logits, out.hidden_states[self.cfg.hidden_layer]

    def state(self, ids, valid, attn_mask=None):
        """Pooled, LayerNormed JEPA state for a view."""
        _, h = self.hidden(ids, attn_mask)
        return self.pool_ln(pool_states(h, valid))

    @torch.no_grad()
    def target_state(self, ids, valid, attn_mask=None):
        """z_tH from the EMA teacher, stop-gradient."""
        self._swap_ema()
        try:
            z = self.state(ids, valid, attn_mask)
        finally:
            self._swap_ema()                         # restore student weights
        return z.detach()

    def action_vector(self, cot_ids, cot_pad, mode, generator=None):
        """a — the predictor's action input (our contribution)."""
        if mode == "none":
            return None
        if mode == "random":
            z = torch.randn(cot_ids.shape[0], self.pool_ln.normalized_shape[0],
                            device=cot_ids.device, generator=generator)
            return self.pool_ln(z.to(next(self.parameters()).dtype))
        if mode == "shuffled":                       # another example's CoT
            perm = torch.randperm(cot_ids.shape[0], device=cot_ids.device)
            cot_ids, cot_pad = cot_ids[perm], cot_pad[perm]
        with torch.no_grad():                        # action is a conditioning
            _, h = self.hidden(cot_ids, (~cot_pad).long())   # signal, not a path
            a = self.pool_ln(pool_states(h, ~cot_pad))       # for encoder grads
        return a.detach()


def diffusion_loss(model, ids, resp_mask, pad_mask, mask_id, t=None, generator=None):
    """
    LLaDA SFT objective: mask response tokens i.i.d. with prob t, predict them,
    reweight by 1/t. Returns (loss, noisy_ids, masked_positions, t).
    """
    B, L = ids.shape
    if t is None:
        t = torch.rand(B, device=ids.device, generator=generator).clamp(min=1e-3)
    p = t.unsqueeze(1).expand(B, L)
    r = torch.rand(B, L, device=ids.device, generator=generator)
    masked = (r < p) & resp_mask & ~pad_mask         # prompt/pad never masked

    # Guarantee ≥1 masked token per example. With short responses (answer-only
    # targets are 1–3 tokens) a low t often masks NOTHING, which silently
    # contributes a zero gradient and wastes the step.
    empty = resp_mask.any(1) & ~masked.any(1)
    if empty.any():
        cand = resp_mask & ~pad_mask
        pick = torch.multinomial(cand[empty].float(), 1).squeeze(1)
        masked[empty.nonzero(as_tuple=True)[0], pick] = True

    noisy = torch.where(masked, torch.full_like(ids, mask_id), ids)

    logits, _ = model.hidden(noisy, (~pad_mask).long())
    if masked.any():
        ce = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none")
        loss = (ce / p[masked]).sum() / (B * L)
    else:
        loss = logits.sum() * 0.0
    return loss, noisy, masked, t


def make_view(ids, resp_mask, pad_mask, rate, mask_id, generator=None):
    """A view of the same input at a fixed masking rate (t_L or t_H)."""
    r = torch.rand(ids.shape, device=ids.device, generator=generator)
    masked = (r < rate) & resp_mask & ~pad_mask
    noisy = torch.where(masked, torch.full_like(ids, mask_id), ids)
    valid = ~masked & ~pad_mask                      # pooling ignores masked+pad
    return noisy, valid


def jepa_loss(z_pred, z_target):
    """1 − cos(sg(z_tH), ẑ_tH)."""
    return (1 - F.cosine_similarity(z_pred.float(), z_target.float(), dim=-1)).mean()
