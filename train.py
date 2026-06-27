"""
LeJEPA-Text training loop — token-level masked latent prediction, no EMA.

Loss = (1 - lam) * MSE(pred_M, target_M)  +  lam * SIGReg(clean_embs)

where:
  pred_M      — predictor output at masked token positions       (M, D)
  target_M    — clean encoder output at masked positions, detached (M, D)
  clean_embs  — full clean encoder output, full gradient          (B, L, D)

Gradient routing (see model.py for full explanation):
  MSE path:    MSE → predictor → encoder(masked)
  SIGReg path: SIGReg → encoder(clean)      [full gradient]
  MSE target:  sg(encoder(clean)[mask])      [no gradient]

Run:
    python train.py --shakespeare --steps 500   # local dev, real text
    python train.py --fake-data   --steps 50    # loop check, no downloads
    python train.py --wandb                     # full run with W&B logging
    python train.py --mask-ratio 0.30           # de-risk ablation (§5 of plan)
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
import wandb
from config      import Config
from data        import get_dataloader
from model       import LeJEPAText
from utils       import get_device, get_dtype, make_autocast, make_scaler
from optim       import build_optimizer, get_lr
from diagnostics import log_diagnostics
from isotropy    import isotropy_loss
from dotenv import load_dotenv

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))


# ── masking ───────────────────────────────────────────────────────────────────

def make_masked_input(x: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply token masking to a batch of sequences.

    Args:
        x:   (B, L) token ids — the clean input
        cfg: Config with mask_ratio, mask_strategy, mask_token_id

    Returns:
        x_masked: (B, L) with mask_token_id at masked positions
        mask:     (B, L) bool tensor, True at masked positions

    Strategies (ablated in de-risk experiment §5 of research plan):
      "random" — independent Bernoulli per token at rate mask_ratio
      "span"   — sample contiguous spans totalling ~mask_ratio of tokens
      "block"  — mask one contiguous block of length mask_ratio * L
    """
    B, L = x.shape

    if cfg.mask_strategy == "random":
        mask = torch.bernoulli(
            torch.full((B, L), cfg.mask_ratio, device=x.device)
        ).bool()

    elif cfg.mask_strategy == "span":
        # Sample spans of average length 6, totalling ~mask_ratio of L.
        # Simple implementation: repeatedly sample start+length pairs until budget used.
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        budget = int(cfg.mask_ratio * L)
        for b in range(B):
            covered = 0
            while covered < budget:
                span_len = torch.randint(3, 10, (1,)).item()
                start    = torch.randint(0, max(1, L - span_len), (1,)).item()
                end      = min(start + span_len, L)
                mask[b, start:end] = True
                covered  = mask[b].sum().item()

    elif cfg.mask_strategy == "block":
        # One contiguous block per sequence.
        mask     = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        blk_len  = int(cfg.mask_ratio * L)
        starts   = torch.randint(0, L - blk_len + 1, (B,))
        for b in range(B):
            mask[b, starts[b]: starts[b] + blk_len] = True

    else:
        raise ValueError(f"Unknown mask_strategy: {cfg.mask_strategy!r}")

    x_masked          = x.clone()
    x_masked[mask]    = cfg.mask_token_id
    return x_masked, mask


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: Config):
    device = get_device()
    dtype  = get_dtype(device)
    print(f"Device: {device}  |  dtype: {dtype}  |  data: "
          f"{'fake' if cfg.fake_data else 'shakespeare' if cfg.shakespeare else 'openwebtext'}")
    print(f"Masking: strategy={cfg.mask_strategy}  ratio={cfg.mask_ratio}")

    model  = LeJEPAText(cfg).to(device)
    optim  = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.betas)
    scaler = make_scaler(device, dtype)
    loader = get_dataloader(cfg, num_workers=0 if device in ("cpu", "mps") else 4)

    if cfg.use_wandb:
        wandb.init(entity="austinmyc", project="lejepa", name=cfg.run_name, config=cfg.__dict__)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    print(f"Parameters: {model.count_params():,}")

    data_iter = iter(loader)
    step = 0
    t0   = time.time()

    while step < cfg.max_steps:

        # ── fetch batch ───────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        x_clean = batch.to(device)   # (B, L)

        # ── mask ──────────────────────────────────────────────────────────
        x_masked, mask = make_masked_input(x_clean, cfg)

        # ── learning rate ─────────────────────────────────────────────────
        lr = get_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # ── forward + loss ────────────────────────────────────────────────
        optim.zero_grad(set_to_none=True)

        with make_autocast(device, dtype):
            pred_M, target_M, clean_embs = model(x_clean, x_masked, mask)

            # MSE on masked token positions only.
            # target_M is already detached (stop-grad in model.forward).
            l_pred = F.mse_loss(pred_M, target_M)

            # SIGReg on all clean token embeddings — full gradient.
            # This is what trains the encoder toward isotropy and prevents
            # representational collapse without an EMA teacher.
            B, L, D = clean_embs.shape
            flat_clean = clean_embs.reshape(B * L, D).float()
            l_isotropy = isotropy_loss(
                flat_clean,
                global_step=step,
                num_slices=cfg.sigreg_num_slices,
                num_points=cfg.sigreg_num_points,
            )

            loss = (1 - cfg.lam) * l_pred + cfg.lam * l_isotropy

        # ── backward + step ───────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optim)
        scaler.update()
        # NOTE: no update_ema() — EMA is removed by design.

        step += 1

        # ── logging ───────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            n_masked = mask.sum().item()
            tokens   = step * cfg.batch_size * cfg.seq_len
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"pred {l_pred.item():.4f} | isotropy {l_isotropy.item():.4f} | "
                f"masked_toks {n_masked} | lr {lr:.2e} | "
                f"tokens {tokens/1e6:.1f}M | t {(time.time()-t0)/60:.1f}min"
            )
            if cfg.use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "l_pred": l_pred.item(),
                    "l_isotropy": l_isotropy.item(),
                    "lr": lr,
                    "masked_tokens": n_masked,
                    "tokens_M": tokens / 1e6,
                }, step=step)

        # ── collapse diagnostics ──────────────────────────────────────────
        if step % cfg.rank_every == 0:
            log_diagnostics(
                clean_embs.detach(),
                step,
                wandb=(__import__("wandb") if cfg.use_wandb else None),
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if step % cfg.save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:06d}.pt")
            torch.save({
                "step":      step,
                "model":     model.state_dict(),
                "optimizer": optim.state_dict(),
                "config":    cfg,
            }, path)
            print(f"  Saved → {path}")

    print("Training complete.")
    if cfg.use_wandb:
        wandb.finish()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",         type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--lam",           type=float, default=None)
    parser.add_argument("--mask-ratio",    type=float, default=None,
                        help="Fraction of tokens to mask (de-risk ablation: 0.15, 0.30, 0.50)")
    parser.add_argument("--mask-strategy", type=str,   default=None,
                        choices=["random", "span", "block"],
                        help="Masking strategy (de-risk ablation, §5 of research plan)")
    parser.add_argument("--wandb",         action="store_true")
    parser.add_argument("--fake-data",     action="store_true")
    parser.add_argument("--shakespeare",   action="store_true")
    parser.add_argument("--tiny",          action="store_true",
                        help="Small model for fast local iteration (d_model=64, 2 enc layers)")
    args = parser.parse_args()

    cfg = Config()
    if args.steps:          cfg.max_steps      = args.steps
    if args.batch_size:     cfg.batch_size     = args.batch_size
    if args.lam:            cfg.lam            = args.lam
    if args.mask_ratio:     cfg.mask_ratio     = args.mask_ratio
    if args.mask_strategy:  cfg.mask_strategy  = args.mask_strategy
    if args.wandb:          cfg.use_wandb      = True
    if args.fake_data:      cfg.fake_data      = True
    if args.shakespeare:    cfg.shakespeare    = True
    if args.tiny:
        cfg.d_model      = 64
        cfg.n_heads      = 4
        cfg.enc_layers   = 2
        cfg.pred_layers  = 1
        cfg.warmup_steps = 200

    train(cfg)
