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

import copy
import math
import os

import torch
import torch.nn.functional as F

from config import Config
from model  import LeJEPAText
from sigreg import sigreg_loss
from data   import get_dataloader, make_masked_input
from eval_mteb import run_mteb_eval
from eval_manifold import manfit


@torch.no_grad()
def update_ema(ema_model, student, decay):
    for ema_p, s_p in zip(ema_model.parameters(), student.parameters()):
        ema_p.lerp_(s_p, 1 - decay)


def masked_span_ids(mask):
    """(B, L) bool → (B, L) int: 0 = unmasked; 1..K = contiguous-span index per row."""
    prev = torch.zeros_like(mask)
    prev[:, 1:] = mask[:, :-1]
    starts = mask & ~prev
    return torch.cumsum(starts.long(), dim=1) * mask.long()


def pool_by_span(flat, mask, span_ids):
    """
    Mean-pool (M, P) masked-position vectors by contiguous span.
    Returns (S, P), one pooled vector per span; (0, P) if no spans.
    """
    B, L = mask.shape
    K = int(span_ids.max())
    if K == 0:
        return flat.new_zeros((0, flat.shape[1]))
    row = torch.arange(B, device=mask.device).unsqueeze(1).expand(B, L)
    key = (row * (K + 1) + span_ids)[mask]              # (M,) flat group ids
    n_groups = B * (K + 1)
    counts = torch.bincount(key, minlength=n_groups)
    used = counts > 0
    denom = counts.clamp(min=1).unsqueeze(1).float()
    s = torch.zeros(n_groups, flat.shape[1], device=flat.device, dtype=flat.dtype)
    s.index_add_(0, key, flat)
    return (s / denom)[used]


def span_pooled_mse(pred_flat, target_flat, mask, span_ids):
    """MSE between mean-pooled prediction and target over each masked span."""
    p = pool_by_span(pred_flat, mask, span_ids)
    if p.shape[0] == 0:
        return pred_flat.new_tensor(0.0)
    return F.mse_loss(p, pool_by_span(target_flat, mask, span_ids))


def cosine_alpha_bar(T=1000, s=0.008):
    """Nichol–Dhariwal cosine schedule: ᾱ_t for t = 0..T-1."""
    t = torch.arange(T + 1, dtype=torch.float32) / T
    f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    ab = (f / f[0])[:-1]
    return ab.clamp(1e-5, 1.0)


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
    torch.manual_seed(cfg.seed)
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

    ema_model = None
    if cfg.use_ema or cfg.d2v_layers > 0:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        sched = f"→{cfg.ema_decay_final}" if cfg.ema_decay_final > 0 else ""
        d2v = f", d2v top-{cfg.d2v_layers} layer-avg targets" if cfg.d2v_layers > 0 else ""
        print(f"EMA teacher enabled (decay={cfg.ema_decay}{sched}{d2v})")

    alpha_bar = cosine_alpha_bar().to(device) if cfg.w_diff > 0 else None
    if cfg.w_diff > 0:
        print(f"Diffusion head enabled (w={cfg.w_diff}, samples={cfg.diff_samples})")

    # FIFO bank of pooled clean embeddings for the contraction loss.
    bank = None
    bank_ptr = 0
    bank_full = False
    if cfg.w_contract > 0:
        bank = torch.zeros(cfg.contract_bank, cfg.d_model, device=device)
        print(f"Manifold contraction enabled (w={cfg.w_contract}, σ={cfg.contract_sigma}, "
              f"bank={cfg.contract_bank}, start={cfg.contract_start})")

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

        pred_all, target, z_clean, h_clean, h_masked = model(x_clean, x_masked, mask, ema_model=ema_model)
        pred = pred_all[mask]                            # (M, P) per-token view

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
        loss = cfg.mse_weight * mse + cfg.lam * reg

        # Pooled JEPA terms — supervise composition (span) and the mean-pool
        # readout (global), which token-level CE cannot express.
        l_span = torch.tensor(0.0, device=device)
        l_glob = torch.tensor(0.0, device=device)
        if cfg.w_span > 0:
            l_span = span_pooled_mse(pred, target, mask, masked_span_ids(mask))
            loss = loss + cfg.w_span * l_span
        if cfg.w_glob > 0:
            l_glob = F.mse_loss(pred_all.mean(dim=1), z_clean.mean(dim=1).detach())
            loss = loss + cfg.w_glob * l_glob

        # Diffusion head: distributional prediction of pooled span targets.
        # Targets per-batch standardized (σ-VAE variance-floor analogue);
        # condition c = pooled predictor output carries the gradient.
        l_diff = torch.tensor(0.0, device=device)
        if model.diff_head is not None:
            seg = masked_span_ids(mask)
            c_pool = pool_by_span(pred, mask, seg)              # (S, P) — with grad
            z_pool = pool_by_span(target, mask, seg)            # (S, P) — detached
            S = z_pool.shape[0]
            if S > 1:
                mu = z_pool.mean(0, keepdim=True)
                sd = z_pool.std(0, keepdim=True).clamp(min=1e-4)
                z_std = (z_pool - mu) / sd
                K = cfg.diff_samples
                zr, cr = z_std.repeat(K, 1), c_pool.repeat(K, 1)
                t = torch.randint(0, alpha_bar.shape[0], (S * K,), device=device)
                ab = alpha_bar[t].unsqueeze(1)
                eps = torch.randn_like(zr)
                z_t = ab.sqrt() * zr + (1 - ab).sqrt() * eps
                l_diff = F.mse_loss(model.diff_head(z_t, t, cr), eps)
                loss = loss + cfg.w_diff * l_diff

        # Manifold contraction-consistency: pull pooled sentence embeddings
        # toward their fitted position on the manifold of recent embeddings.
        # Targets detached — no gradient through the fitting.
        l_contract = torch.tensor(0.0, device=device)
        if bank is not None:
            pooled = h_clean.mean(dim=1)                        # (B, D) — with grad
            if bank_full and step >= cfg.contract_start:
                with torch.no_grad():
                    k = min(256, cfg.contract_bank // 2)
                    sub = bank[torch.randperm(cfg.contract_bank, device=device)[:2 * k]]
                    scale = torch.cdist(sub[:k], sub[k:]).median().clamp(min=1e-6)
                    target_c = manfit(bank / scale, pooled.detach() / scale,
                                      cfg.contract_sigma, device=device,
                                      as_numpy=False) * scale
                l_contract = F.mse_loss(pooled, target_c)
                loss = loss + cfg.w_contract * l_contract
            with torch.no_grad():                               # bank update (FIFO)
                n = pooled.shape[0]
                idx = (bank_ptr + torch.arange(n, device=device)) % cfg.contract_bank
                bank[idx] = pooled.detach()
                bank_ptr = (bank_ptr + n) % cfg.contract_bank
                if bank_ptr < n:
                    bank_full = True

        # MLM anchor: decode back to token logits at masked positions.
        # Grounds the latent space in data (see config.mlm_beta / mlm_head).
        ce = torch.tensor(0.0, device=device)
        if model.decoder is not None:
            src = pred if model.mlm_head == "pred" else h_masked[mask]
            logits = model.decoder(src)                # (M, vocab)
            ce = F.cross_entropy(logits, x_clean[mask])
            loss = loss + cfg.mlm_beta * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if ema_model is not None:
            decay = cfg.ema_decay
            if cfg.ema_decay_final > 0:                # data2vec anneal
                decay += (cfg.ema_decay_final - cfg.ema_decay) * step / cfg.max_steps
            update_ema(ema_model, model, decay)

        if step % cfg.log_every == 0:
            print(f"step {step:05d} | loss {loss.item():.4f} | mse {mse.item():.4f} | "
                  f"reg {reg.item():.4f} | ce {ce.item():.4f} | "
                  f"span {l_span.item():.4f} | glob {l_glob.item():.4f} | "
                  f"contract {l_contract.item():.4f} | diff {l_diff.item():.4f} | "
                  f"masked {int(mask.sum())} | lr {lr:.2e}")
            if cfg.use_wandb:
                wandb.log({"loss": loss.item(), "mse": mse.item(), "reg": reg.item(),
                           "ce": ce.item(), "l_span": l_span.item(), "l_glob": l_glob.item(),
                           "l_contract": l_contract.item(), "l_diff": l_diff.item(),
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

    if cfg.run_mteb:
        device_str = str(next(model.parameters()).device)
        wandb_run = wandb.run if cfg.use_wandb else None
        run_mteb_eval(model, cfg, cfg.max_steps, device_str, wandb_run=wandb_run)

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
    p.add_argument("--mask-ratio",    type=float, default=Config.mask_ratio)
    p.add_argument("--mask-strategy", type=str,   default=Config.mask_strategy,
                   choices=["random", "span", "block"])
    p.add_argument("--span-len",      type=int,   default=Config.span_len,
                   help="Fixed span length for span strategy (0 = random 3-9).")
    p.add_argument("--w-span",        type=float, default=Config.w_span,
                   help="Weight of the span-pooled latent MSE (composition term).")
    p.add_argument("--w-glob",        type=float, default=Config.w_glob,
                   help="Weight of the global (sequence-mean) latent MSE.")
    p.add_argument("--w-diff",         type=float, default=Config.w_diff,
                   help="Diffusion-head loss weight (0 = off; distributional span prediction).")
    p.add_argument("--diff-samples",   type=int,   default=Config.diff_samples)
    p.add_argument("--w-contract",     type=float, default=Config.w_contract,
                   help="Manifold contraction-consistency weight (0 = off).")
    p.add_argument("--contract-sigma", type=float, default=Config.contract_sigma)
    p.add_argument("--contract-bank",  type=int,   default=Config.contract_bank)
    p.add_argument("--contract-start", type=int,   default=Config.contract_start)
    p.add_argument("--seed",          type=int,   default=Config.seed)
    p.add_argument("--latent-space",  type=str,   default=Config.latent_space,
                   choices=["proj", "encoder"],
                   help="Space for the latent-prediction task (targets + predictor).")
    p.add_argument("--no-normalize-target", action="store_true",
                   help="Predict raw (unnormalized) targets — known to diverge; for ablation only.")
    p.add_argument("--save-every",  type=int,   default=Config.save_every)
    p.add_argument("--ema",         action="store_true", help="Enable EMA teacher.")
    p.add_argument("--ema-decay",   type=float, default=Config.ema_decay)
    p.add_argument("--ema-decay-final", type=float, default=Config.ema_decay_final,
                   help="Anneal EMA decay linearly to this value (data2vec schedule).")
    p.add_argument("--d2v-layers",  type=int,   default=Config.d2v_layers,
                   help="K>0: data2vec targets — instance-normed avg of teacher's top-K layers.")
    p.add_argument("--mlm-beta",    type=float, default=Config.mlm_beta,
                   help="Weight of the MLM anchor CE loss (0 = pure JEPA, no decoder head).")
    p.add_argument("--mse-weight",  type=float, default=Config.mse_weight,
                   help="Weight of the JEPA MSE term (0 + --mlm-beta = pure MLM baseline).")
    p.add_argument("--mlm-head",    type=str,   default=Config.mlm_head,
                   choices=["pred", "encoder"],
                   help="CE head attach point: predictor output (anchored JEPA) or encoder output (BERT-style control).")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--mteb",        action="store_true",
                   help="Run MTEB eval on the final checkpoint after training.")
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
        mask_ratio=a.mask_ratio, mask_strategy=a.mask_strategy, span_len=a.span_len,
        save_every=a.save_every,
        use_ema=a.ema, ema_decay=a.ema_decay, ema_decay_final=a.ema_decay_final,
        d2v_layers=a.d2v_layers,
        mlm_beta=a.mlm_beta, mse_weight=a.mse_weight, mlm_head=a.mlm_head,
        w_span=a.w_span, w_glob=a.w_glob, seed=a.seed, latent_space=a.latent_space,
        w_contract=a.w_contract, contract_sigma=a.contract_sigma,
        contract_bank=a.contract_bank, contract_start=a.contract_start,
        w_diff=a.w_diff, diff_samples=a.diff_samples,
        use_wandb=a.wandb, run_mteb=a.mteb, run_name=a.run_name,
    ))
