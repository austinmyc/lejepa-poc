"""
Diagnose the flat-MSE mystery: is the latent-prediction task degenerate?

Every run so far shows MSE pinned at ~0.0025 from early in training while other
loss terms move. This script answers three questions on a trained checkpoint:

  1. TRIVIAL-BASELINE R² — how much better is the model's prediction than just
     predicting the global mean target? R² ≈ 0 means the predictor has learned
     nothing beyond the mean → the task is degenerate. This is the single
     number that explains (or exonerates) every flat-MSE run.

  2. TARGET GEOMETRY — variance, effective rank, mean pairwise cosine of the
     targets. Near-constant targets (tiny variance, high cosine) mean there is
     nothing to predict in the first place.

  3. CONTEXT SENSITIVITY — replace all unmasked (context) tokens with random
     tokens and re-run the masked path. If predictions barely change
     (cos ≈ 1), the predictor ignores context entirely and outputs a function
     of position/mask alone.

Usage (server; checkpoints live there):
    python mask/diagnose.py checkpoints_mask/<run>_final.pt [--batches 10]
    GPU=0 python mask/diagnose.py <ckpt>          # default GPU=3

Read-out is stdout only — no W&B. Runs in no-grad; light footprint (safe to
run next to a training job on the same GPU if needed).
"""

import argparse
import os

import torch
import torch.nn.functional as F

from config import Config          # noqa: F401 (needed to unpickle cfg in ckpt)
from model import LeJEPAText
from data import get_dataloader, make_masked_input


@torch.no_grad()
def geometry(x, name):
    """Eff-rank / cosine / variance summary of a (N, D) cloud."""
    D = x.shape[-1]
    flat = x.reshape(-1, D).float()
    centered = (flat - flat.mean(0, keepdim=True)).cpu()
    sv = torch.linalg.svdvals(centered)
    ev = sv ** 2
    eff_rank = (ev.sum() ** 2 / ev.pow(2).sum()).item()
    n = min(1024, flat.shape[0])
    idx = torch.randperm(flat.shape[0])[:n]
    u = F.normalize(flat[idx].cpu(), dim=1)
    cos = u @ u.T
    mean_cos = ((cos.sum() - n) / (n * (n - 1))).item()
    var = centered.pow(2).mean().item()          # per-element variance
    rms = flat.pow(2).mean().sqrt().item()
    print(f"  {name:22} var/elem={var:.6f}  rms={rms:.4f}  "
          f"eff_rank={eff_rank:.1f}/{D}  mean_cos={mean_cos:.4f}")
    return var


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Diagnose latent-prediction degeneracy.")
    ap.add_argument("checkpoint")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override cfg batch size (default: use checkpoint's).")
    args = ap.parse_args()

    gpu = os.environ.get("GPU", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if args.batch_size:
        cfg.batch_size = args.batch_size
    model = LeJEPAText(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Checkpoint: {args.checkpoint}  (step {ckpt['step']})")
    print(f"Config: lam={cfg.lam} mask={cfg.mask_strategy}@{cfg.mask_ratio} "
          f"normalize_target={cfg.normalize_target} d_model={cfg.d_model} d_proj={cfg.d_proj}")
    print(f"Device: {device}  |  batches: {args.batches} × {cfg.batch_size}\n")

    loader = get_dataloader(cfg)
    data_iter = iter(loader)

    preds, targets, preds_shuf = [], [], []
    for b in range(args.batches):
        x_clean = next(data_iter).to(device)
        x_masked, mask = make_masked_input(x_clean, cfg)

        pred, target, z_clean, h_clean, h_masked = model(x_clean, x_masked, mask)
        preds.append(pred.float().cpu())
        targets.append(target.float().cpu())

        # Context-shuffle: random tokens at all context positions, same mask.
        x_shuf = torch.randint(0, cfg.vocab_size, x_masked.shape, device=device)
        x_shuf[mask] = cfg.mask_token_id
        h_s = model.encoder(x_shuf)
        z_s = model.proj(h_s)
        preds_shuf.append(model.predictor(z_s)[mask].float().cpu())

    pred = torch.cat(preds)          # (N, P)
    target = torch.cat(targets)      # (N, P)
    pred_shuf = torch.cat(preds_shuf)
    N, P = target.shape
    print(f"Collected {N} masked-position pairs, dim {P}\n")

    # ── 1. trivial-baseline R² ───────────────────────────────────────────────
    print("── 1. Trivial-baseline R² " + "─" * 40)
    mse_model = F.mse_loss(pred, target).item()
    mse_mean  = F.mse_loss(target.mean(0, keepdim=True).expand_as(target), target).item()
    r2 = 1 - mse_model / mse_mean if mse_mean > 0 else float("nan")
    print(f"  MSE(model)              = {mse_model:.6f}")
    print(f"  MSE(global-mean target) = {mse_mean:.6f}   (= target variance)")
    print(f"  R² over mean-predictor  = {r2:.4f}")
    print(f"  → {'DEGENERATE: no gain over the mean predictor' if r2 < 0.05 else 'model predicts real structure beyond the mean'}")

    # Direction-only version (what normalize_target=True runs optimized).
    pn, tn = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
    mse_model_n = F.mse_loss(pn, tn).item()
    tmean_dir = F.normalize(tn.mean(0, keepdim=True), dim=-1)
    mse_mean_n = F.mse_loss(tmean_dir.expand_as(tn), tn).item()
    r2_n = 1 - mse_model_n / mse_mean_n if mse_mean_n > 0 else float("nan")
    cos_pt = (pn * tn).sum(-1).mean().item()
    print(f"  [normalized] R² = {r2_n:.4f}   mean cos(pred, target) = {cos_pt:.4f}")

    # ── 2. target & prediction geometry ─────────────────────────────────────
    print("\n── 2. Geometry " + "─" * 51)
    var_t = geometry(target, "targets z_clean[mask]")
    var_p = geometry(pred,   "predictions")
    print(f"  Var(pred)/Var(target) = {var_p / var_t:.4f}   "
          f"(≪1 → predictor outputs near-constant)")

    # ── 3. context sensitivity ───────────────────────────────────────────────
    print("\n── 3. Context sensitivity " + "─" * 40)
    cos_ctx = (F.normalize(pred, dim=-1) * F.normalize(pred_shuf, dim=-1)).sum(-1).mean().item()
    rel_shift = (pred - pred_shuf).pow(2).mean().item() / max(var_t, 1e-12)
    print(f"  cos(pred, pred_random_context)      = {cos_ctx:.4f}")
    print(f"  ||Δpred||² / Var(target)            = {rel_shift:.4f}")
    print(f"  → {'predictor IGNORES context (position-only output)' if cos_ctx > 0.95 else 'predictor uses context'}")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n── Verdict " + "─" * 55)
    if r2 < 0.05 and cos_ctx > 0.95:
        print("  Task is degenerate: predictions are a context-independent constant")
        print("  ≈ the mean target. MSE was flat because there was nothing to learn.")
    elif r2 < 0.05:
        print("  Predictor responds to context but gains no MSE over the mean —")
        print("  targets likely carry no predictable structure (random-ish encoder).")
    else:
        print("  Prediction task is non-trivial (R² > 0.05); flat MSE needs another")
        print("  explanation (e.g. loss scale, or MSE converged early and stayed).")


if __name__ == "__main__":
    main()
