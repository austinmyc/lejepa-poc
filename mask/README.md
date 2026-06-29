# `mask/` — Masked Latent Prediction with Projection-Space SIGReg

A BERT-like masked-prediction model where the targets are **embeddings** (not
tokens), collapse is prevented by **SIGReg** (not EMA), and isotropy is enforced
in a **projection space** so the encoder is free to stay anisotropic.

This is a clean-slate build under `mask/`; it does not touch the root files.

---

## Files

| File | Role |
|------|------|
| `config.py` | All hyperparameters (dataclass). |
| `model.py` | `TokenEncoder` → `ProjectionMLP` → `SpanPredictor`, tied in `LeJEPAText`. |
| `sigreg.py` | SIGReg isotropy loss (Epps-Pulley CF test), applied in projection space. |
| `data.py` | Datasets (fake / Shakespeare / streaming corpus) + `make_masked_input`. |
| `train.py` | Training loop + CLI. Logs to W&B, saves checkpoints. |
| `eval_sts.py` | STS-B evaluation (a cheap MTEB task) for a checkpoint. |
| `smoke_test.py` | Shapes + gradient-routing + masking sanity checks. |
| `setup.sh` / `wandb_login.sh` / `run_owt.sh` | Server: install, auth, launch OWT. |

---

## How it works

Two forward passes share the **same** encoder + projection (no EMA, no teacher):

```
clean:  z_clean  = proj(encoder(x_clean))     # full grad
          ├─ SIGReg(z_clean)                  → trains encoder+proj toward isotropy
          └─ target = z_clean[mask].detach()  → stop-grad MSE target

masked: z_masked = proj(encoder(x_masked)) → predictor → pred[mask]
          └─ MSE(normalize(pred), normalize(target))    → trains predictor+proj+encoder

loss = MSE + lam * SIGReg(z_clean)
```

Eval read-out: **encoder → mean-pool** (the projection/predictor are training-time
only). `eval_sts.py --readout encoder` is the standard; `proj` is available but
scored lower in testing.

---

## Key decisions & findings (don't re-litigate)

- **SIGReg flows into the encoder** (paper-faithful — this is how SIGReg replaces
  EMA). The earlier `methods_note.md` §4 proposed *stop-gradding* it (projection-only)
  to protect the encoder's geometry; we chose paper-faithful because cutting the
  gradient removes collapse prevention. See the long discussion if revisiting.
- **`lam = 0.006`** — calibrated so SIGReg's gradient on the encoder is ~3× the MSE
  gradient (at `lam=0.05` it was ~25×, over-shaping the encoder toward isotropy).
- **`sigreg_grad_scale = α` (default 1.0)** — biases *where* the isotropy shaping
  happens. SIGReg's gradient reaches the encoder through the projection; a grad-scale
  layer on the clean pass multiplies the part that flows *into the encoder* by α, while
  the projection always gets the full gradient. So the projection can do the whitening
  (`z` isotropic) while the encoder is shielded from being dragged isotropic. `α=1` =
  full/paper-faithful; `α<1` pushes the shaping onto the projection; `α=0` = encoder
  fully shielded (≈ the stop-grad design, but *no* collapse insurance on the encoder).
  α is **decoupled from λ**: λ sets total isotropy strength, α sets the encoder's share.
  Sweep with `--sigreg-grad-scale`; verify via `enc_iso` vs `proj_iso` (below). Note
  the encoder is *always* shaped fully by prediction (the masked pass is unaffected).
- **Normalized target is essential.** With raw MSE, the predictor (LayerNorm-capped,
  output magnitude ~1) cannot match a target whose `tgt_rms` grows past 1, so MSE
  **diverges** (climbed to ~6 by step 1800). Normalizing pred+target (direction-only
  prediction) decouples MSE from magnitude and keeps it flat (~0.01). Toggle off only
  for ablation: `--no-normalize-target`.
- **`d_proj=128` is fine at POC scale.** 128 vs 256 gave identical quality
  (cosine ~0.36) — the bottleneck wasn't binding. At scale, use `d_proj >= d_model`
  (expander, VICReg-style), per `methods_note.md`. **This result won't transfer to
  large models/data** — re-check at scale.
- **To match other embedding models, scale `d_model`** (the read-out width), not
  `d_proj`. 768 = BERT-base parity.

## Diagnostics

Logged to console + W&B every `rank_every` steps, on **both** the projection space
(`proj_*`, want isotropic) and the **encoder** space (`enc_*`, want anisotropic):
- `proj_rank` — hard singular-value rank. **Coarse**: stayed 128/128 even during a full
  directional collapse, so don't trust it alone.
- `mean_cos` — mean pairwise cosine. ~0 = isotropic/healthy; →1 = directional collapse.
- `tgt_rms` — target magnitude. SIGReg's target is 1.0 (N(0,1)); it often drifts to
  3–5 (SIGReg isn't pinning scale), harmless now that MSE is normalized.
- `proj_eff_rank` / `enc_eff_rank` — **participation ratio** `(Σλ)²/Σλ²` ∈ [1, D].
  D = isotropic, 1 = collapsed. More sensitive than `rank` — it catches the
  variance-pancake collapse that `rank` misses.
- `proj_iso` / `enc_iso` — `eff_rank / D` ∈ (0, 1]; 1 = perfectly isotropic.
- `iso_gap = proj_iso − enc_iso` — **the test of the design premise.** Should be
  **positive** (projection more isotropic than the encoder) if the projection is
  absorbing the isotropy shaping and the encoder is staying anisotropic. If `enc_iso`
  is collapsing along with `proj_iso`, lower α to shield the encoder.

---

## Running

**Local (Shakespeare, fast, on MPS/CPU):**
```bash
python mask/train.py --shakespeare --steps 2000 --wandb --run-name dev
python mask/eval_sts.py checkpoints_mask/dev_final.pt
```

**Server (full OpenWebText):**
```bash
bash mask/setup.sh                      # conda env + requirements
echo "WANDB_API_KEY=<key>" > .env       # .env is gitignored — recreate it here
bash mask/wandb_login.sh
bash mask/run_owt.sh                     # edit the config block at the top first
```
`run_owt.sh` defaults to 768/12L (~110M), `Skylion007/openwebtext`, 50k steps,
`lam=0.006`, normalized target, checkpoints every 5k. Switch corpus to
`HuggingFaceFW/fineweb` or drop to 256/4L in the script's config block.

**Sweep the isotropy gate α:**
```bash
GPU=0 bash mask/sweep_alpha.sh          # runs α=1.0, 0.3, 0.0 (10k steps each)
```
Then compare `enc_iso` / `proj_iso` / `iso_gap` across the runs in W&B — a positive,
rising `iso_gap` means the projection is absorbing the isotropy while the encoder
stays anisotropic.

**Pin to one GPU (e.g. an L20):** the script sets `CUDA_VISIBLE_DEVICES` from the
`GPU` variable (default `0`). Find the index with `nvidia-smi`, then:
```bash
GPU=0 bash mask/run_owt.sh          # or edit GPU= in the script
```
`train.py` auto-selects CUDA when available, so no other change is needed.

---

## Validation status

- **Stability / correctness / pipeline: proven** locally. No collapse, no divergence,
  STS pipeline + checkpointing + baseline-relative measurement all work.
- **Embedding quality: not yet proven.** STS-B random-init baseline ≈ **0.32**
  (lexical-overlap floor); Shakespeare-trained **dropped to ~0.11** — expected domain
  mismatch (tiny, archaic, memorized corpus). A real verdict needs the **OWT server
  run**, then full MTEB vs a matched-compute BERT-base baseline.
