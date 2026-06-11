"""
Smoke test — run this first before committing to a full training run.
Checks that:
  1. Isotropy loss API works as expected
  2. Model forward pass runs without error
  3. Stop-gradient wiring is correct
  4. Loss is finite
  5. Backward pass runs without NaN gradients
  6. Embedding rank is healthy (> 1) on random inputs

Should complete in < 30 seconds on CPU.

Usage:
    python smoke_test.py
"""

import torch
import torch.nn.functional as F

from config    import Config
from model     import LeJEPAText
from isotropy  import isotropy_loss


def check(cond, msg):
    status = "✓" if cond else "✗  FAILED"
    print(f"  {status}  {msg}")
    if not cond:
        raise AssertionError(msg)


def main():
    cfg = Config()
    cfg.batch_size = 4
    cfg.max_chunks = 8
    cfg.chunk_size = 32
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.enc_layers = 2
    cfg.pred_layers = 1
    cfg.proj_hidden = 128
    cfg.fake_data = True

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nDevice: {device}")

    # ── 1. Isotropy loss API ───────────────────────────────────────────────
    print("\n[1] Isotropy loss API")
    dummy_embs = torch.randn(32, cfg.d_model)
    l_iso = isotropy_loss(dummy_embs)
    check(torch.isfinite(l_iso), f"isotropy_loss is finite: {l_iso.item():.4f}")
    # On random normal input, variance term should be near zero (std ≈ 1 already)
    # and covariance term should be small. Total loss should be low.
    check(l_iso.item() < 50.0, f"isotropy_loss reasonable on N(0,1) input: {l_iso.item():.4f}")

    # ── 2. Model forward ──────────────────────────────────────────────────
    print("\n[2] Model forward pass")
    model = LeJEPAText(cfg).to(device)
    print(f"  Parameters: {model.count_params():,}")
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_chunks, cfg.chunk_size)).to(device)
    pred_embs, target_proj, proj_detached, chunk_embs = model(x)
    exp_shape = (cfg.batch_size, cfg.max_chunks - 1, cfg.d_model)
    check(pred_embs.shape    == exp_shape, f"pred_embs shape: {pred_embs.shape}")
    check(target_proj.shape  == exp_shape, f"target_proj shape: {target_proj.shape}")
    check(proj_detached.shape == (cfg.batch_size, cfg.max_chunks, cfg.d_model),
          f"proj_detached shape: {proj_detached.shape}")

    # ── 3. Stop-gradient check ────────────────────────────────────────────
    print("\n[3] Stop-gradient verification")
    check(chunk_embs.requires_grad, "chunk_embs has grad (encoder trained by MSE)")
    # proj_detached flows through projection MLP params (requires_grad=True is correct),
    # but must NOT flow gradients back to the encoder. Verify by checking the grad_fn
    # chain does not reach the encoder: chunk_embs should be a leaf in proj_detached's graph.
    B_iso, N_iso, D_iso = proj_detached.shape
    iso_loss = isotropy_loss(proj_detached.reshape(B_iso * N_iso, D_iso).float())
    iso_loss.backward(retain_graph=True)
    encoder_grad_norm = sum(
        p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None
    )
    # zero out the grads we just computed so they don't pollute later checks
    model.zero_grad()
    check(encoder_grad_norm == 0.0, "encoder gets no grad from isotropy loss (stop-grad works)")

    # ── 4. Loss ───────────────────────────────────────────────────────────
    print("\n[4] Loss computation")
    l_pred = F.mse_loss(pred_embs, target_proj)
    B, N, D = proj_detached.shape
    flat_detached = proj_detached.reshape(B * N, D).float()
    l_isotropy = isotropy_loss(flat_detached)
    loss = (1 - cfg.lam) * l_pred + cfg.lam * l_isotropy
    check(torch.isfinite(loss),       f"Total loss finite: {loss.item():.4f}")
    check(torch.isfinite(l_pred),     f"Pred loss finite: {l_pred.item():.4f}")
    check(torch.isfinite(l_isotropy), f"Isotropy loss finite: {l_isotropy.item():.4f}")

    # ── 5. Backward ───────────────────────────────────────────────────────
    print("\n[5] Backward pass")
    loss.backward()
    max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
    check(torch.isfinite(torch.tensor(max_grad)), f"Max gradient finite: {max_grad:.4f}")

    # ── 6. Embedding rank ─────────────────────────────────────────────────
    print("\n[6] Embedding rank (random init — expect ~D)")
    with torch.no_grad():
        enc_flat = chunk_embs.reshape(-1, cfg.d_model).detach()
        enc_flat = enc_flat - enc_flat.mean(0)
        _, s, _ = torch.linalg.svd(enc_flat.float(), full_matrices=False)
        cumvar = (s ** 2).cumsum(0) / (s ** 2).sum()
        rank = int((cumvar < 0.99).sum().item()) + 1
    check(rank > 1, f"Encoder embedding rank > 1: {rank}")

    print("\n✓ All checks passed — safe to launch training.\n")


if __name__ == "__main__":
    main()
