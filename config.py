from dataclasses import dataclass


@dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    vocab_size: int = 50257       # GPT-2 BPE vocab
    mask_token_id: int = 50256    # GPT-2 EOS token used as [MASK] (no dedicated mask in GPT-2 vocab)
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4           # bidirectional encoder layers
    pred_layers: int = 2          # span predictor layers (lighter than encoder)
    seq_len: int = 1024           # total tokens per sample

    # ── Masking ────────────────────────────────────────────────────────────
    # De-risk experiment (§5 of research plan) ablates three strategies:
    #   "random" — token-level Bernoulli masking (default, fast to implement)
    #   "span"   — contiguous span masking (planned ablation)
    #   "block"  — mask one contiguous block (I-JEPA style)
    mask_ratio: float = 0.15      # fraction of tokens masked; ablate 0.15, 0.30, 0.50
    mask_strategy: str = "random" # "random" | "span" | "block"

    # ── SIGReg ─────────────────────────────────────────────────────────────
    lam: float = 0.05             # weight on SIGReg loss (paper default)
    sigreg_num_slices: int = 512
    sigreg_num_points: int = 17

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    max_steps: int = 50_000       # ≈ 500M tokens on single GPU
    warmup_steps: int = 1_000

    # ── Data ───────────────────────────────────────────────────────────────
    fake_data: bool = False       # random tokens — no downloads (smoke tests)
    shakespeare: bool = False     # tiny-shakespeare — local dev with real text

    # ── Logging & checkpoints ──────────────────────────────────────────────
    log_every: int = 100
    rank_every: int = 500
    save_every: int = 5_000
    run_name: str = "lejepa_text"
    use_wandb: bool = False

    # ── Paths ──────────────────────────────────────────────────────────────
    data_cache: str = "./data_cache"
    checkpoint_dir: str = "./checkpoints"
