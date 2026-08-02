"""
Config for the DLLM-JEPA reproduction + CoT-as-action extension (Part III).

Method reference: DLLM-JEPA (arXiv 2606.00091):
    L_total = L_diff + λ · L_JEPA
    L_JEPA  = 1 − cos( sg(z_tH), g_φ(z_tL) )
    z_tL = Pool(f_θ(x_tL)),  z_tH = Pool(f_θ'(x_tH)),  θ' = EMA(θ), τ = 0.996
    two views = the SAME input at two masking rates (t_L = 0.2, t_H = 0.7)
    Pool = mean over non-masked, non-pad tokens → LayerNorm
    predictor g_φ = k ∈ {1..5} transformer decoder layers, randomly initialised

Our extension: the predictor is conditioned on an ACTION a = Pool(f_θ(CoT)),
injected via AdaLN. LLM-JEPA and DLLM-JEPA both drop the action from LeCun's
formulation; CoT is the natural action for reasoning tasks (see PLAN.md).
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── Backbone ───────────────────────────────────────────────────────────
    # Masked-diffusion LM. LLaDA/Dream load via AutoModelForCausalLM with
    # trust_remote_code (bidirectional attention lives inside the custom code).
    model_name:    str  = "GSAI-ML/LLaDA-8B-Instruct"
    trust_remote:  bool = True
    dtype:         str  = "bfloat16"
    # Mask token id. 0 = auto-detect (tokenizer.mask_token_id, else the model
    # config's mask_token_id — LLaDA uses 126336). Override if auto-detect fails.
    mask_token_id: int  = 0

    # ── LoRA (the paper full-finetunes on 8×A100; we LoRA to fit 1–3 GPUs) ──
    use_lora:      bool  = True
    lora_r:        int   = 32
    lora_alpha:    int   = 64
    lora_dropout:  float = 0.05
    # Comma-separated module name substrings; "" = let peft pick defaults.
    lora_targets:  str   = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"

    # ── DLLM-JEPA objective ────────────────────────────────────────────────
    lam:           float = 1.0     # λ ∈ {0.5, 1.0, 2.0} in the paper
    t_low:         float = 0.2     # context view mask rate
    t_high:        float = 0.7     # target view mask rate
    pred_layers:   int   = 2       # k ∈ {1..5}
    ema_decay:     float = 0.996   # τ
    # Which hidden layer to pool for the JEPA state (-1 = last).
    hidden_layer:  int   = -1
    # Diffusion loss noise level:
    #   "uniform" — separate pass, t ~ U(0,1) (standard LLaDA SFT; keeps the
    #               model healthy at ALL mask rates, which generation needs).
    #               2 gradient passes total = the paper's 33% FLOP saving vs
    #               LLM-JEPA's 3.
    #   "context" — reuse the t_L context pass for L_diff (1 gradient pass,
    #               cheaper, but only ever trains the denoiser at t_L).
    diff_noise:    str   = "uniform"

    # ── Action conditioning (our contribution) ─────────────────────────────
    # "none"    — A1: plain DLLM-JEPA, predictor is view→view (the reproduction)
    # "cot"     — A2: a = Pool(f_θ(CoT)) of THIS example  (the contribution)
    # "shuffled"— A3: a = another example's CoT   (faithfulness control)
    # "random"  — A4: a = fixed-variance random vector (any-conditioning control)
    action:        str   = "none"
    action_inject: str   = "adaln"          # "adaln" | "prefix"

    # ── Data ───────────────────────────────────────────────────────────────
    dataset:       str  = "openai/gsm8k"
    dataset_config: str = "main"
    max_len:       int  = 256               # 256 GSM8K / 512 code tasks
    max_cot_len:   int  = 256
    # A5 comparator: put the CoT in the GENERATION target (plain CoT-SFT).
    # For A0–A4 the CoT must never enter the diffusion loss context, or the arm
    # silently becomes CoT-SFT.
    cot_in_target: bool = False

    # ── Training ───────────────────────────────────────────────────────────
    lr:            float = 1e-5             # paper's aggressive schedule
    epochs:        int   = 2
    batch_size:    int   = 2                # per device
    grad_accum:    int   = 8
    warmup_ratio:  float = 0.03
    weight_decay:  float = 0.0
    grad_clip:     float = 1.0
    grad_ckpt:     bool  = True
    seed:          int   = 1337

    # ── Eval ───────────────────────────────────────────────────────────────
    gen_steps:     int   = 128              # iterative unmasking steps
    gen_len:       int   = 256
    n_shot:        int   = 4                # paper's primary metric is 4-shot
    eval_n:        int   = 0                # 0 = full test set

    # ── Logging ────────────────────────────────────────────────────────────
    out_dir:       str  = "./ft_checkpoints"
    log_every:     int  = 10
    use_wandb:     bool = False
    wandb_entity:  str  = "austinmyc"
    wandb_project: str  = "lejepa-ft"
    run_name:      str  = "dllm_jepa"
