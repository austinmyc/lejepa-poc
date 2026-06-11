"""
LeJEPA-Text training loop.

Loss = (1 - lam) * MSE(pred, target_proj)  +  lam * isotropy_loss(proj_detached)

REVIEW: Two gradient paths — understand both before reading this file:
  MSE path  (attached): MSE → predictor → projection → encoder
                        All three components are trained by prediction signal.
  Isotropy path (detached): isotropy_loss → projection only
                            encoder.detach() breaks the gradient at the encoder.
                            Only the projection is pushed toward isotropy.

Run:
    python train.py --shakespeare --steps 500   # local dev, real text
    python train.py --fake-data   --steps 50    # loop check, no downloads
    python train.py --wandb                     # full run with W&B logging
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

def train(cfg: Config):
    device = get_device()
    dtype  = get_dtype(device)
    print(f"Device: {device}  |  dtype: {dtype}  |  data: "
          f"{'fake' if cfg.fake_data else 'shakespeare' if cfg.shakespeare else 'openwebtext'}")

    model   = LeJEPAText(cfg).to(device)
    optim   = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.betas)
    scaler  = make_scaler(device, dtype)
    loader  = get_dataloader(cfg, num_workers=0 if device in ("cpu", "mps") else 4)

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
        batch = batch.to(device)   # (B, N, C)

        # ── learning rate ─────────────────────────────────────────────────
        lr = get_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # ── forward + loss ────────────────────────────────────────────────
        optim.zero_grad(set_to_none=True)

        with make_autocast(device, dtype):
            pred_embs, target_proj, proj_detached, chunk_embs = model(batch)

            # REVIEW: MSE target is `target_proj` — the projection of the
            # target chunk with gradient attached. This trains encoder +
            # projection + predictor jointly via the prediction signal.
            l_pred = F.mse_loss(pred_embs, target_proj)

            # REVIEW: Isotropy loss input is `proj_detached` — the projection
            # of ALL chunks but with the encoder output detached.
            # Gradient only reaches the projection, not the encoder.
            # isotropy_loss pushes the projection toward high, equal, independent
            # variance per dimension — without distorting the encoder's geometry.
            # See isotropy.py for the variance + covariance term breakdown.
            B, N, D = proj_detached.shape
            flat_detached = proj_detached.reshape(B * N, D).float()
            l_isotropy = isotropy_loss(flat_detached, global_step=step,
                                       num_slices=cfg.sigreg_num_slices,
                                       num_points=cfg.sigreg_num_points)

            loss = (1 - cfg.lam) * l_pred + cfg.lam * l_isotropy

        # ── backward + step ───────────────────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optim)
        scaler.update()

        step += 1

        # ── logging ───────────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            tokens = step * cfg.batch_size * cfg.max_chunks * cfg.chunk_size
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"pred {l_pred.item():.4f} | isotropy {l_isotropy.item():.4f} | "
                f"lr {lr:.2e} | tokens {tokens/1e6:.1f}M | "
                f"t {(time.time()-t0)/60:.1f}min"
            )
            if cfg.use_wandb:
                
                wandb.log({
                    "loss": loss.item(), "l_pred": l_pred.item(),
                    "l_isotropy": l_isotropy.item(), "lr": lr,
                    "tokens_M": tokens / 1e6,
                }, step=step)

        # ── collapse diagnostics ──────────────────────────────────────────
        if step % cfg.rank_every == 0:
            log_diagnostics(
                chunk_embs, proj_detached, step,
                wandb=(__import__("wandb") if cfg.use_wandb else None),
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if step % cfg.save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"ckpt_{step:06d}.pt")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "config": cfg,
            }, path)
            print(f"  Saved → {path}")

    print("Training complete.")
    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",       type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--lam",         type=float, default=None)
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--fake-data",   action="store_true")
    parser.add_argument("--shakespeare", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.steps:       cfg.max_steps   = args.steps
    if args.batch_size:  cfg.batch_size  = args.batch_size
    if args.lam:         cfg.lam         = args.lam
    if args.wandb:       cfg.use_wandb   = True
    if args.fake_data:   cfg.fake_data   = True
    if args.shakespeare: cfg.shakespeare = True

    train(cfg)
