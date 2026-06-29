"""
Training loop for the masked latent-prediction model (projection-space SIGReg).

Loss (literal to masked_prediction_plan.md):
    loss = MSE(pred[mask], z_clean[mask].detach()) + lam * SIGReg(z_clean)

Both terms live in projection space. SIGReg flows proj → encoder(clean) and is the
sole collapse-prevention mechanism (no EMA). Collapse diagnostics probe projection
space only (z_clean), per the plan's "What to Check" table.

Run:
    python mask/train.py                 # Shakespeare (cfg default)
"""

import math
import os

import torch
import torch.nn.functional as F

from config import Config
from model  import LeJEPAText
from sigreg import sigreg_loss
from data   import get_dataloader, make_masked_input


def get_lr(step, cfg):
    """Linear warmup → cosine decay to 10% of base lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def space_geometry(x):
    """
    Geometry of an embedding cloud x: (B, L, D). Returns a dict:
      rank      — hard SVD rank (coarse; misses directional collapse)
      eff_rank  — participation ratio (Σλ)²/Σλ²  ∈ [1, D]; D = isotropic, 1 = collapsed
      iso       — eff_rank / D  ∈ (0, 1]; 1 = perfectly isotropic, low = anisotropic
      mean_cos  — mean pairwise cosine; ~0 healthy, →1 directional collapse
      rms       — root-mean-square magnitude (SIGReg target is 1.0)
    Used on BOTH the projection (z, should be isotropic) and the encoder (h, should
    stay anisotropic) — the empirical test of whether the projection absorbs the
    isotropy shaping rather than the encoder.
    """
    D = x.shape[-1]
    flat = x.reshape(-1, D).float()
    # svdvals isn't implemented on MPS, so do the SVD on CPU.
    centered = (flat - flat.mean(0, keepdim=True)).cpu()
    sv = torch.linalg.svdvals(centered)
    ev = sv ** 2                                       # ∝ covariance eigenvalues
    rank = int((sv > 1e-5 * sv.max()).sum().item())
    eff_rank = (ev.sum() ** 2 / ev.pow(2).sum()).item()
    n = min(512, flat.shape[0])
    idx = torch.randperm(flat.shape[0], device=flat.device)[:n]
    u = F.normalize(flat[idx], dim=1)
    cos = u @ u.T
    mean_cos = ((cos.sum() - n) / (n * (n - 1))).item()
    rms = flat.pow(2).mean().sqrt().item()
    return {"rank": rank, "eff_rank": eff_rank, "iso": eff_rank / D,
            "mean_cos": mean_cos, "rms": rms}


def train(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  data: "
          f"{'fake' if cfg.fake_data else 'shakespeare' if cfg.shakespeare else 'owt'}  |  "
          f"mask: {cfg.mask_strategy}@{cfg.mask_ratio_range or cfg.mask_ratio}  |  "
          f"d_model={cfg.d_model} d_proj={cfg.d_proj}")

    model = LeJEPAText(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = get_dataloader(cfg)
    print(f"Parameters: {model.count_params():,}")

    if cfg.use_wandb:
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                   name=cfg.run_name, config=cfg.__dict__)

    data_iter = iter(loader)
    for step in range(cfg.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        x_clean = batch.to(device)                       # (B, L)
        x_masked, mask = make_masked_input(x_clean, cfg)

        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        pred, target, z_clean, h_clean = model(x_clean, x_masked, mask)

        B, L, P = z_clean.shape
        if cfg.normalize_target:
            # Direction-only prediction: scale-match the LayerNorm-capped
            # predictor to the target. target is already detached.
            mse = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))
        else:
            mse = F.mse_loss(pred, target)
        reg = sigreg_loss(z_clean.reshape(B * L, P),
                          num_slices=cfg.sigreg_num_slices,
                          num_points=cfg.sigreg_num_points,
                          global_step=step)
        loss = mse + cfg.lam * reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            print(f"step {step:05d} | loss {loss.item():.4f} | mse {mse.item():.4f} | "
                  f"reg {reg.item():.4f} | masked {int(mask.sum())} | lr {lr:.2e}")
            if cfg.use_wandb:
                wandb.log({"loss": loss.item(), "mse": mse.item(), "reg": reg.item(),
                           "lr": lr, "masked_tokens": int(mask.sum())}, step=step)

        if step % cfg.rank_every == 0:
            zg = space_geometry(z_clean.detach())   # projection space (want isotropic)
            hg = space_geometry(h_clean.detach())   # encoder space (want anisotropic)
            # Skip the collapse flag at step 0 — random init is anisotropic by
            # nature (high mean_cos), so it's a guaranteed false alarm.
            flag = "  ⚠️ possible collapse" if (step > 0 and (zg["rank"] <= 2 or zg["mean_cos"] > 0.5)) else ""
            # The design premise holds when proj_iso ≫ enc_iso (projection absorbs
            # isotropy; encoder stays anisotropic).
            print(f"  [diag] proj_iso {zg['iso']:.2f} (rank {zg['rank']}/{P}, cos {zg['mean_cos']:.3f}) | "
                  f"enc_iso {hg['iso']:.2f} (eff_rank {hg['eff_rank']:.0f}/{h_clean.shape[-1]}) | "
                  f"tgt_rms {zg['rms']:.2f}{flag}")
            if cfg.use_wandb:
                wandb.log({
                    "proj_rank": zg["rank"], "proj_eff_rank": zg["eff_rank"],
                    "proj_iso": zg["iso"], "mean_cos": zg["mean_cos"], "tgt_rms": zg["rms"],
                    "enc_eff_rank": hg["eff_rank"], "enc_iso": hg["iso"],
                    "enc_mean_cos": hg["mean_cos"], "enc_rms": hg["rms"],
                    "iso_gap": zg["iso"] - hg["iso"],   # >0 = projection more isotropic than encoder
                }, step=step)

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(model, cfg, step)

    save_checkpoint(model, cfg, cfg.max_steps, final=True)

    if cfg.use_wandb:
        wandb.finish()


def save_checkpoint(model, cfg, step, final=False):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    tag = "final" if final else f"{step:06d}"
    path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_{tag}.pt")
    torch.save({"step": step, "model": model.state_dict(), "config": cfg}, path)
    print(f"  saved checkpoint → {path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train the mask/ masked-latent-prediction model.")
    p.add_argument("--steps",       type=int,   default=Config.max_steps)
    p.add_argument("--corpus",      type=str,   default=Config.corpus,
                   help="Namespaced HF repo for the full run (ignored if --shakespeare/--fake-data).")
    p.add_argument("--shakespeare", action="store_true", help="Use tiny-shakespeare (local dev).")
    p.add_argument("--fake-data",   action="store_true", help="Use random tokens (smoke).")
    p.add_argument("--lr",          type=float, default=Config.lr,
                   help="Peak learning rate. Scale up with batch size (≈√ rule).")
    p.add_argument("--warmup-steps", type=int,  default=Config.warmup_steps)
    p.add_argument("--lam",         type=float, default=Config.lam)
    p.add_argument("--sigreg-grad-scale", type=float, default=Config.sigreg_grad_scale,
                   help="α: fraction of SIGReg gradient reaching the encoder (1=full, 0=shielded).")
    p.add_argument("--d-model",     type=int,   default=Config.d_model)
    p.add_argument("--d-proj",      type=int,   default=Config.d_proj)
    p.add_argument("--n-heads",     type=int,   default=Config.n_heads)
    p.add_argument("--enc-layers",  type=int,   default=Config.enc_layers)
    p.add_argument("--pred-layers", type=int,   default=Config.pred_layers)
    p.add_argument("--batch-size",  type=int,   default=Config.batch_size)
    p.add_argument("--seq-len",     type=int,   default=Config.seq_len)
    p.add_argument("--no-normalize-target", action="store_true",
                   help="Predict raw (unnormalized) targets — known to diverge; for ablation only.")
    p.add_argument("--save-every",  type=int,   default=Config.save_every)
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--run-name",    type=str,   default=Config.run_name)
    a = p.parse_args()

    train(Config(
        max_steps=a.steps, corpus=a.corpus,
        shakespeare=a.shakespeare, fake_data=a.fake_data,
        lr=a.lr, warmup_steps=a.warmup_steps,
        lam=a.lam, sigreg_grad_scale=a.sigreg_grad_scale,
        d_model=a.d_model, d_proj=a.d_proj,
        n_heads=a.n_heads, enc_layers=a.enc_layers, pred_layers=a.pred_layers,
        batch_size=a.batch_size, seq_len=a.seq_len,
        normalize_target=not a.no_normalize_target,
        save_every=a.save_every, use_wandb=a.wandb, run_name=a.run_name,
    ))
