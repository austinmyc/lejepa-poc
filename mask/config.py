"""
Config for the masked latent-prediction model (mask/ — projection-space SIGReg).

Distinct from the root config: SIGReg lives in PROJECTION space here, so there is
a separate d_proj and the predictor operates in P dims, not d_model.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    vocab_size:    int = 50257       # GPT-2 BPE vocab
    mask_token_id: int = 50256       # GPT-2 EOS reused as [MASK]
    d_model:       int = 256
    d_proj:        int = 128         # projection space; SIGReg + MSE live here
    n_heads:       int = 8
    enc_layers:    int = 4
    pred_layers:   int = 2
    seq_len:       int = 128

    # ── Masking ────────────────────────────────────────────────────────────
    mask_ratio:    float = 0.15
    mask_strategy: str   = "random"  # "random" | "span" | "block"
    # Optional dynamic ratio: if set to (lo, hi), each batch samples its mask
    # ratio ~ Uniform(lo, hi) instead of using the fixed mask_ratio. Exposes the
    # model to a spread of difficulties. Empty () = disabled (fixed ratio).
    mask_ratio_range: tuple = ()

    # ── Prediction target ──────────────────────────────────────────────────
    # L2-normalize pred and target before MSE (direction-only prediction). The
    # predictor's final LayerNorm caps its output magnitude at ~1, so an
    # unnormalized target whose magnitude grows past 1 (SIGReg doesn't pin it)
    # makes MSE rise ~ tgt_rms². Normalizing decouples MSE from that magnitude
    # tug-of-war and lets SIGReg alone own the global geometry.
    normalize_target: bool = True

    # ── SIGReg (projection space) ──────────────────────────────────────────
    # SIGReg gradient reaches the encoder (paper-faithful: this is how it
    # replaces EMA). At lam=0.05 it outweighs the MSE gradient on the encoder
    # ~25x, which over-shapes the encoder toward isotropy. lam=0.006 calibrates
    # that to ~3x — prediction shapes the encoder, SIGReg is collapse insurance.
    lam:               float = 0.006
    sigreg_num_slices: int   = 512
    sigreg_num_points: int   = 17
    # α — fraction of SIGReg's gradient that reaches the encoder (projection always
    # gets the full gradient). 1.0 = full/paper-faithful; <1 biases isotropy shaping
    # onto the projection so the encoder stays anisotropic; 0.0 = encoder fully
    # shielded (no collapse insurance on the encoder). Decoupled from lam (which
    # sets total isotropy strength). See enc_eff_rank vs proj_eff_rank to verify.
    sigreg_grad_scale: float = 1.0

    # ── Training ───────────────────────────────────────────────────────────
    batch_size:   int   = 32
    lr:           float = 3e-4
    weight_decay: float = 0.1
    grad_clip:    float = 1.0
    max_steps:    int   = 10_000
    warmup_steps: int   = 500

    # ── Data ───────────────────────────────────────────────────────────────
    # Local validation: shakespeare=True. Full run (server): set both False to
    # stream `corpus`. Switch corpus here (must be a namespaced HF repo, e.g.
    # "HuggingFaceFW/fineweb") — single-name repos no longer resolve.
    shakespeare: bool = True
    fake_data:   bool = False
    data_cache:  str  = "./data_cache"
    corpus:      str  = "Skylion007/openwebtext"

    # ── Logging & checkpoints ──────────────────────────────────────────────
    log_every:      int = 50
    rank_every:     int = 200
    save_every:     int = 0                  # >0 = also save periodic ckpts
    checkpoint_dir: str = "./checkpoints_mask"

    # ── W&B ────────────────────────────────────────────────────────────────
    use_wandb:      bool = False
    wandb_entity:   str  = "austinmyc"
    wandb_project:  str  = "lejepa"
    run_name:       str  = "lejepa_mask"
