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
    # Fixed span length for the "span" strategy (RQ4 span-length sweep).
    # 0 = legacy behaviour (random lengths 3–9 per span).
    span_len:      int   = 0

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

    # ── Paired-view JEPA (cell 1: view gap YES, abstraction gap NO) ─────────
    # Cross-view path (train_paired.py): predict pooled proj(enc(view_B)) from
    # view_A, where A/B are a genuine semantic pair (docstring↔code, doc↔summary).
    # pair_source names a PAIR_PRESETS entry in data.py; pair_repo/col_a/col_b/
    # split override it (empty string = use the preset's value). The repo must be
    # a namespaced parquet HF repo (script-based datasets don't stream here).
    pair_source: str = "code"        # "code" | "summary" | "cnndm" | "simplify" | custom
    pair_repo:   str = ""            # override HF repo; "" = use preset
    pair_col_a:  str = ""            # override view-A column; "" = use preset
    pair_col_b:  str = ""            # override view-B column; "" = use preset
    pair_split:  str = ""            # override split; "" = use preset
    pair_config: str = ""            # override dataset config name; "" = use preset
    # Control arm: permute view B within the batch so pairs are broken. The
    # chance-floor analogue of the random-target control — if the objective still
    # helps the readout under shuffling, the A↔B pairing is not what carries the
    # signal. 0 = real pairs.
    shuffle_pairs: bool = False
    # ── Contrastive (InfoNCE) arm — the non-JEPA way to exploit the pairs ───
    # w_con > 0 adds symmetric in-batch InfoNCE between pooled encoder codes of
    # view A and view B (SimCSE/CLIP objective): matched pairs are positives,
    # the rest of the batch are negatives. The POSITIVE CONTROL for the pairing
    # question: if InfoNCE(real) ≫ InfoNCE(shuffled) while JEPA(real) ≈
    # JEPA(shuffled), the pairs carry extractable signal and prediction is what
    # fails — negatives are the missing ingredient (THEORY.md conjecture).
    # NOTE: gradient must reach both views — keep sigreg_grad_scale at 1.0.
    w_con:    float = 0.0
    con_temp: float = 0.05           # InfoNCE temperature (SimCSE default)

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

    # ── EMA teacher ────────────────────────────────────────────────────────
    use_ema:        bool  = False
    ema_decay:      float = 0.999  # τ: ema = τ*ema + (1-τ)*student each step

    # ── data2vec-style targets (the from-scratch counterexample test) ──────
    # K > 0: latent targets = instance-normalized average of the EMA teacher's
    # top-K layer outputs at masked positions (per-position, never pooled).
    # Implies an EMA teacher (auto-enabled) and latent_space="encoder" (dims).
    # Layer-averaging pulls targets toward the input-indexed embedding layer —
    # data2vec's implicit anchor. 0 = off.
    d2v_layers:      int   = 0
    # > 0: anneal EMA decay linearly from ema_decay to this over max_steps
    # (data2vec schedule, e.g. 0.999 → 0.9999).
    ema_decay_final: float = 0.0

    # ── MLM anchor (data-grounded auxiliary loss) ──────────────────────────
    # β > 0 adds a token-decoder head on the predictor output at masked
    # positions: loss += β * CE(decode(pred_M), tokens_M). This anchors the
    # latent space to data and breaks the chicken-and-egg problem of pure
    # latent prediction from scratch (random targets teach nothing).
    # β = 0 disables the head entirely (pure JEPA, original behaviour).
    mlm_beta:       float = 0.0
    # Where the CE head attaches: "pred" (predictor output — anchored JEPA) or
    # "encoder" (encoder(x_masked) output — standard BERT-style MLM control).
    mlm_head:       str   = "pred"
    # Weight on the per-token JEPA MSE term. 0 + mlm_beta>0 = pure MLM baseline
    # with the same architecture (matched-compute control).
    mse_weight:     float = 1.0

    # Where the latent-prediction task lives:
    #   "proj"    — predictor and targets in projection space (original design;
    #               SIGReg-whitened targets → pooled chance floor, see RESULTS)
    #   "encoder" — predictor and targets in raw encoder space (anisotropic,
    #               CE-grounded, semantic — data2vec-style target choice).
    #               proj is bypassed on the latent path; requires d_proj==d_model
    #               only in the sense that the predictor is built at d_model.
    latent_space:   str   = "proj"

    # ── Pooled JEPA terms (supervise what token CE cannot express) ─────────
    # w_span: MSE between the mean-pooled predictor output over each masked
    #         span and the pooled clean latent of that span — composition.
    # w_glob: MSE between the sequence-mean predictor output (masked view) and
    #         the sequence-mean clean latent — trains the mean-pool readout
    #         MTEB evaluates. SIGReg keeps these pooled targets non-degenerate.
    w_span:         float = 0.0
    w_glob:         float = 0.0

    # ── Diffusion head (LatentLM/MAR-style distributional latent prediction) ─
    # Replaces point-MSE on pooled span targets with noise-prediction:
    #   L_diff = E_t‖ε − ε_θ(√ᾱ_t·z + √(1−ᾱ_t)·ε, t, c)‖²,  c = pooled predictor
    # output per masked span. Fixes the mode-averaging failure of point
    # regression (predictor → conditional mean → chance floor). Targets are
    # pooled clean latents, per-batch standardized (σ-VAE variance-floor
    # analogue). Head is eval-time scaffolding; readout unchanged.
    w_diff:         float = 0.0    # loss weight (0 = off, head not built)
    diff_samples:   int   = 4      # sampled timesteps per span per step

    # ── Manifold contraction-consistency (training-time manifold fitting) ──
    # Pulls each sentence's pooled clean-encoder embedding toward its fitted
    # position on the manifold of recent embeddings (FIFO bank). Targets are
    # detached — no gradient through the fitting. Bakes in the post-hoc
    # cluster-tightening effect while letting CE co-adapt.
    w_contract:       float = 0.0    # loss weight (0 = off)
    contract_sigma:   float = 0.15   # manfit σ in median-distance-normalized space
    contract_bank:    int   = 4096   # FIFO bank size (fitting cloud)
    contract_start:   int   = 2000   # steps before the loss activates (bank fill + CE warmup)

    # ── Reproducibility ────────────────────────────────────────────────────
    seed:           int   = 1337

    # ── Post-training MTEB eval ────────────────────────────────────────────
    run_mteb:       bool = False   # run MTEB eval after training finishes
