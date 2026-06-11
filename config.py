from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    vocab_size: int = 50257       # GPT-2 BPE vocab
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4           # bidirectional encoder layers
    pred_layers: int = 2          # causal predictor layers
    chunk_size: int = 64          # tokens per chunk
    max_chunks: int = 16          # chunks per sample  →  16×64 = 1024 tokens

    # ── Projection MLP ─────────────────────────────────────────────────────
    proj_hidden: int = 2048       # hidden dim of projection MLP (768→2048→768 at full scale)

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
    fake_data: bool = False       # use random tokens — no internet (smoke tests)
    shakespeare: bool = False     # use tiny-shakespeare (~1MB) — local dev with real text

    # ── Logging & checkpoints ──────────────────────────────────────────────
    log_every: int = 100
    rank_every: int = 500         # how often to compute embedding rank
    save_every: int = 5_000
    run_name: str = "lejepa_poc"
    use_wandb: bool = False       # set True if wandb is configured

    # ── Paths ──────────────────────────────────────────────────────────────
    data_cache: str = "./data_cache"
    checkpoint_dir: str = "./checkpoints"
