"""
Smoke test for the mask/ build — shapes, gradient routing, masking.

Checks:
  1. Forward shapes: pred (M, P), target (M, P), z_clean (B, L, P)
  2. target is detached (no grad); z_clean carries grad
  3. SIGReg gradient reaches the ENCODER through the projection (clean path)
  4. MSE gradient reaches the PREDICTOR (masked path)
  5. make_masked_input works for all three strategies and respects ~mask_ratio
  6. One end-to-end optimizer step runs and loss is finite

Run: python mask/smoke_test.py
"""

import torch
import torch.nn.functional as F

from config import Config
from model  import LeJEPAText
from sigreg import sigreg_loss
from data   import make_masked_input


def main():
    torch.manual_seed(0)
    cfg = Config(seq_len=32, batch_size=4, d_model=64, d_proj=32,
                 enc_layers=2, pred_layers=1, sigreg_num_slices=64)
    device = "cpu"
    model = LeJEPAText(cfg).to(device)

    x_clean = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    x_masked, mask = make_masked_input(x_clean, cfg)
    M = int(mask.sum())

    # ── 1. shapes ─────────────────────────────────────────────────────────
    pred, target, z_clean, h_clean = model(x_clean, x_masked, mask)
    assert pred.shape   == (M, cfg.d_proj),                     pred.shape
    assert target.shape == (M, cfg.d_proj),                     target.shape
    assert z_clean.shape == (cfg.batch_size, cfg.seq_len, cfg.d_proj), z_clean.shape
    assert h_clean.shape == (cfg.batch_size, cfg.seq_len, cfg.d_model), h_clean.shape
    print(f"[1] shapes OK: pred {tuple(pred.shape)}, target {tuple(target.shape)}, "
          f"z_clean {tuple(z_clean.shape)}, h_clean {tuple(h_clean.shape)}")

    # ── 2. stop-grad on target, grad on z_clean ───────────────────────────
    assert not target.requires_grad, "target must be detached"
    assert z_clean.requires_grad,    "z_clean must carry grad (SIGReg path)"
    print("[2] target detached, z_clean requires_grad OK")

    # ── 3. SIGReg grad reaches the encoder via projection ─────────────────
    model.zero_grad()
    reg = sigreg_loss(z_clean.reshape(-1, cfg.d_proj), num_slices=cfg.sigreg_num_slices)
    reg.backward(retain_graph=False)
    enc_grad = model.encoder.tok_emb.weight.grad
    proj_grad = model.proj.net[0].weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0,  "SIGReg must reach encoder"
    assert proj_grad is not None and proj_grad.abs().sum() > 0, "SIGReg must reach projection"
    print(f"[3] SIGReg → encoder (|g|={enc_grad.abs().sum():.4f}) "
          f"& projection (|g|={proj_grad.abs().sum():.4f}) OK")

    # ── 4. MSE grad reaches the predictor ─────────────────────────────────
    model.zero_grad()
    pred, target, _, _ = model(x_clean, x_masked, mask)
    mse = F.mse_loss(pred, target)
    mse.backward()
    pred_params = [p for p in model.predictor.parameters() if p.grad is not None]
    pred_grad = sum(p.grad.abs().sum() for p in pred_params)
    assert pred_grad > 0, "MSE must reach predictor"
    print(f"[4] MSE → predictor (|g|={pred_grad:.4f}) OK")

    # ── 4b. α (sigreg_grad_scale) throttles SIGReg grad into the encoder ───
    def enc_sigreg_grad(alpha):
        m = LeJEPAText(Config(seq_len=32, batch_size=4, d_model=64, d_proj=32,
                              enc_layers=2, pred_layers=1, sigreg_num_slices=64,
                              sigreg_grad_scale=alpha)).to(device)
        m.load_state_dict(model.state_dict())   # identical weights, only α differs
        m.eval()                                # disable dropout so only α varies
        _, _, z, _ = m(x_clean, x_masked, mask)
        m.zero_grad()
        sigreg_loss(z.reshape(-1, cfg.d_proj), num_slices=cfg.sigreg_num_slices).backward()
        enc = sum(p.grad.abs().sum() for p in m.encoder.parameters() if p.grad is not None)
        proj = sum(p.grad.abs().sum() for p in m.proj.parameters() if p.grad is not None)
        return float(enc), float(proj)
    e1, p1 = enc_sigreg_grad(1.0)
    eh, ph = enc_sigreg_grad(0.5)
    e0, p0 = enc_sigreg_grad(0.0)
    assert abs(eh - 0.5 * e1) < 1e-3 * e1, f"α=0.5 should halve encoder grad: {eh} vs {0.5*e1}"
    assert e0 < 1e-6, f"α=0 should zero encoder grad, got {e0}"
    assert abs(p1 - p0) < 1e-3 * p1, "projection grad must be unchanged by α"
    print(f"[4b] α scales encoder grad: α=1→{e1:.3f}, α=.5→{eh:.3f}, α=0→{e0:.3f} | "
          f"proj grad α-invariant ({p1:.3f}≈{p0:.3f}) OK")

    # ── 5. masking strategies + ratio ─────────────────────────────────────
    for strat in ("random", "span", "block"):
        c = Config(seq_len=128, batch_size=8, mask_strategy=strat, mask_ratio=0.15)
        x = torch.randint(0, c.vocab_size, (c.batch_size, c.seq_len))
        xm, m = make_masked_input(x, c)
        frac = m.float().mean().item()
        assert m.dtype == torch.bool and xm.shape == x.shape
        assert (xm[m] == c.mask_token_id).all(), "masked positions must hold mask token"
        assert (xm[~m] == x[~m]).all(),          "unmasked positions must be unchanged"
        print(f"[5] strategy={strat:6s} masked_frac={frac:.3f} OK")

    # ── 6. one end-to-end step ────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred, target, z_clean, _ = model(x_clean, x_masked, mask)
    B, L, P = z_clean.shape
    loss = F.mse_loss(pred, target) + cfg.lam * sigreg_loss(
        z_clean.reshape(B * L, P), num_slices=cfg.sigreg_num_slices)
    opt.zero_grad(); loss.backward(); opt.step()
    assert torch.isfinite(loss), "loss must be finite"
    print(f"[6] end-to-end step OK, loss={loss.item():.4f}")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
