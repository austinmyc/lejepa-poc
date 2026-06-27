# LeJEPA-Text: 3-Month Research Plan (v2)

**Updated:** 2026-06-20  
**Goal:** Top-venue paper on EMA-free masked latent prediction for language, gated on a de-risk experiment that decides between Path A (JEPA init for masked diffusion) and Path B (SIGReg geometry fixes latent text diffusion).

---

## The one decision that drives everything

Run the de-risk experiment in Week 3. Its outcome determines the entire second month:

- **Latents smooth + isotropic + decodable → Path B** (spotlight shot: "geometry is the missing ingredient for latent text diffusion")
- **Latents entangled or poorly decodable → Path A** (solid paper: "EMA-free JEPA is a better from-scratch initializer for masked diffusion than diffusion-from-scratch")

Do not commit to either path before that experiment. Everything in Month 1 is shared.

---

## Month 1 — Foundation and de-risk (Weeks 1–4)

### Week 1–2: Self-implement the training objective

**Why implement it yourself:**  
You need to be able to explain every line to a reviewer and in an oral. The scaffolded code is correct, but understanding it and being able to write it are different things. The gradient routing in particular — two encoder passes, stop-grad on the target, full grad for SIGReg — is the core architectural claim. If you can't derive it from first principles you can't defend it.

**What to implement from scratch (do not copy the scaffolded files):**

1. **The two-pass forward function.** Write `model.py` yourself. The key is understanding *why* you run the encoder twice:
   - Pass 1 (clean sequence): gives you the target embeddings. You detach them before MSE so the encoder is not rewarded for making the target easy to predict — only for making the masked-sequence representation predictive.
   - Pass 2 (masked sequence): the encoder sees `[MASK]` tokens at masked positions and bidirectional context everywhere else. The predictor refines these into estimates of the clean embeddings.
   - The stop-grad on the clean target is what prevents the trivial solution (encoder collapses to a constant, predictor trivially matches).

2. **The masking function.** Implement `make_masked_input()` for all three strategies: random, span, block. Understand why span masking is the right default:
   - Random token masking: each token is independent, so the model can reconstruct each masked token from its immediate neighbors. Not much long-range structure learned.
   - Span masking: removes a contiguous chunk, forcing the model to use broader context to fill in the gap. This is why BERT uses whole-word masking and I-JEPA uses block masking.
   - Block masking: one large contiguous block — maximum pressure on long-range structure.
   - The mask ratio is also critical: too low (< 15%) and the task is trivial; too high (> 50%) and the context is too sparse for the MSE to be low-uncertainty, risking mean-collapse.

3. **SIGReg.** Re-implement `isotropy.py` from the LeJEPA paper description. The key idea: project the batch of embeddings onto many random 1D directions. For each direction, compare the empirical characteristic function (ECF) of the projected values against the theoretical CF of a standard Gaussian N(0,1). The ECF is just the sample average of exp(i·t·x) at a grid of t values. The distance between ECFs is the loss. This pushes the marginal distribution in every direction toward Gaussian — which means isotropic Gaussian in aggregate. Understand why this replaces EMA: EMA prevents collapse by keeping the target distribution stable (the teacher moves slowly). SIGReg prevents collapse by directly penalizing degenerate distributions.

4. **The training loop.** Implement `train.py` from scratch. The important parts:
   - Two encoder passes per step (compute cost doubles vs. a single-pass model)
   - SIGReg gradient flows to encoder through the clean pass — this is how the encoder is trained toward isotropy
   - MSE gradient flows to predictor + encoder through the masked pass
   - No `update_ema()` call anywhere

**What to borrow (do not re-implement):**
- `nn.TransformerEncoderLayer` — standard, well-tested, not a research contribution
- The data pipeline (`data.py`) — packing/streaming is engineering, not research
- The optimizer (`optim.py`) — Muon is borrowed from nanochat, cite it
- W&B logging

**Deliverable at end of Week 2:**  
`python smoke_test.py` passes all 8 checks on your self-implemented code.

---

### Week 3: De-risk experiment

This is the most important experiment in the project. It is cheap (124M model, Shakespeare or small OWT slice), and its outcome gates the entire strategy.

**What to run:**

Train 6 small models (d_model=128, 4 enc layers, 2 pred layers, ~20M params) for 10k steps each:

| Experiment | Masking strategy | Mask ratio |
|---|---|---|
| A1 | next-chunk (causal predictor, old code) | — |
| A2 | span | 0.15 |
| A3 | span | 0.30 |
| A4 | span | 0.50 |
| A5 | random | 0.15 |
| A6 | block | 0.30 |

**What to measure on each:**

1. **Isotropy (SIGReg residual):** run the SIGReg loss on the trained encoder's embeddings without backprop. Lower = more isotropic. You want A2–A6 to clearly beat A1.

2. **Embedding rank:** effective dimensionality (99% variance). Want high (close to d_model), not collapsed.

3. **Local smoothness:** perturb a clean embedding by small Gaussian noise → decode → check that the decoded token is semantically close to the original. Measures whether the latent space is smooth enough for diffusion. (Requires a simple linear decoder trained on top of the frozen encoder.)

4. **Decoder reconstruction fidelity:** train a tiny linear decoder (frozen encoder → vocab logits) for 1k steps. Measure reconstruction accuracy. If the encoder's embeddings carry enough information to recover the original token, they're rich enough to serve as diffusion targets.

5. **Mean-collapse check for A1 (next-chunk):** plot the cosine similarity between pred_M and the actual target over training. If it collapses to a constant (sim → 1.0, all predictions identical), A1 is confirmed as collapsed. This is the empirical demonstration of mean-collapse that justifies your design choice.

**Go/no-go decision:**
- Best masked-span variant (A2 or A3) achieves: rank > d_model/2, smoothness probe passes, decoder accuracy > 60% → **green-light Path B**
- Latents entangled or decoder accuracy < 40% → **fall back to Path A**

**Deliverable:** A single plot grid (6 subplots) showing the 5 metrics above for each variant. This becomes Figure 1 or Figure 2 in the paper.

---

### Week 4: Baselines and scale-up prep

Set up the comparison baselines before the main runs so they're ready to launch in Month 2:

1. **data2vec equivalent (EMA baseline):** Use the scaffolded EMA infrastructure from the old `model.py` (it's still in git history). Train on identical data and compute. This is your primary comparison — you need to beat it on at least one named axis (stability under LR variation, data efficiency, or geometry quality).

2. **Random-encoder + SIGReg geometry (Path B isolation):** Train a random encoder (no JEPA pretraining), apply SIGReg to shape its latents, then test diffusability. If this works as well as JEPA + SIGReg, the JEPA pretraining is not necessary — geometry alone is sufficient. If it fails, both geometry AND representation quality matter. This ablation is critical for the causal claim in Path B.

3. **Confirm training stability across 3 seeds:** Run the best de-risk config (e.g., A2) with 3 random seeds. Plot variance. LeJEPA's stability claim (EMA-free is more stable) needs empirical support here.

---

## Month 2 — Main experiments (Weeks 5–8)

### If Path B (latents are diffusable):

**Week 5–6: Build and train the latent diffusion model**

The three-component architecture:
```
encoder (JEPA + SIGReg) → denoiser (latent diffusion) → decoder (latent → tokens)
```

Step by step:
1. Train the full JEPA encoder (124M, full seq_len=1024) for 50k steps to convergence.
2. Train a decoder (2-layer MLP or small transformer, frozen encoder): encoder latents → token logits. Loss: cross-entropy. This is your reconstruction bridge.
3. Train a denoiser: add Gaussian noise to encoder latents at level t; train a small transformer to predict the clean latent. Loss: MSE(denoiser_output, clean_latent). This is a standard continuous-space diffusion model operating in the encoder's latent space.
4. Generation: sample noise in latent space → iteratively denoise → decode to tokens (once, at the end).

The denoiser is what makes this a generator. The JEPA encoder shapes the space the denoiser operates in. Do NOT try to use the JEPA predictor directly as a generator — it was trained to predict in-distribution latents, not to denoise noised latents.

**Week 7–8: Path B ablations**

| Ablation | What it isolates |
|---|---|
| JEPA encoder + SIGReg (full model) | baseline for Path B |
| Random encoder + SIGReg-shaped latents | isolates geometry from representation quality |
| JEPA encoder without SIGReg | isolates representation quality from geometry |
| Prior latent diffusion (Diffusion-LM style, no JEPA) | the graveyard baseline |

If the random encoder + SIGReg performs close to full JEPA + SIGReg: geometry is sufficient. If not: representation quality also matters. Either result is interesting — it clarifies the mechanism.

---

### If Path A (latents are not diffusable):

**Week 5–6: From-scratch JEPA init for masked diffusion**

1. Train JEPA encoder for 50k steps (Phase 1).
2. Add a token LM head (d_model → vocab_size) to the encoder.
3. Fine-tune with the masked diffusion objective (LLaDA-style): mask tokens at rate t, predict them with cross-entropy, weight by 1/t. This is Phase 2.
4. Compare compute-matched against: diffusion-from-scratch, data2vec-init → diffusion, random-init → diffusion.

The key claim: JEPA init gives you a better starting geometry for Phase 2, so the Phase 2 training converges faster or to a better solution.

**Week 7–8: Path A ablations**

| Ablation | What it tests |
|---|---|
| JEPA (SIGReg, no EMA) → masked diffusion | main result |
| data2vec (EMA) → masked diffusion | EMA baseline |
| Random init → masked diffusion | lower bound |
| JEPA (no SIGReg, collapsed) → masked diffusion | SIGReg contribution |

Plot: task accuracy (GSM8K or similar) vs. total FLOPs (Phase 1 + Phase 2 combined). If JEPA init reaches the same accuracy with fewer Phase 2 FLOPs, the claim holds.

---

## Month 3 — Analysis and writing (Weeks 9–12)

### Week 9: Mechanism study

Regardless of which path you're on, this week produces the "why it works" section — which is often what separates a strong paper from a weak one.

**Geometry analysis:**
- Plot the SIGReg residual (isotropy deviation) of the encoder throughout training. Does it decrease monotonically? Does it correlate with downstream task performance?
- Compare the embedding geometry of your encoder vs. data2vec vs. a vanilla BERT: isotropy, effective rank, local smoothness.
- Layer-wise geometry: does isotropy increase with depth? Does it transfer across layers?

**For Path B specifically:**
- Measure which latent-space property (isotropy vs. smoothness vs. rank) best predicts diffusion quality across the ablation conditions. This is the causal analysis that makes the thesis defensible.

**For Path A specifically:**
- Plot layer-wise representation drift between Phase 1 (JEPA) and Phase 2 (diffusion fine-tune). Does the good geometry from Phase 1 survive Phase 2? If it does, that's the mechanism.

### Week 10–11: Paper writing

Write in this order (not introduction-first):

1. **Method section first** — you know exactly what you did; write the precise spec.
2. **Experiments section** — present the results tables/figures; write the captions before the prose.
3. **Analysis section** — the mechanism story; this is what justifies the contribution.
4. **Related work** — use the positioning paragraph from the research plan as the skeleton.
5. **Introduction last** — now that you know what the paper actually shows, you can write the intro honestly.
6. **Abstract very last** — 4 sentences: problem, method, result, significance.

Do not write the introduction first. It leads to overclaiming.

### Week 12: Polish and submit

- Read every claim and check it against the actual results. If a claim isn't supported by a figure or table, cut it or run the experiment.
- Have one person unfamiliar with the project read the paper and mark every sentence they don't understand.
- Check the related work table is complete: data2vec, LeJEPA, LLM-JEPA, DLLM-JEPA, I-JEPA, Diffusion-LM, CDCD, LD4LG, LCM.

---

## Baseline checklist (must run before claiming any result)

- [ ] data2vec (EMA, same scale, same data, same compute)
- [ ] JEPA without SIGReg (collapses — confirms SIGReg is load-bearing)
- [ ] SIGReg without JEPA (random encoder + SIGReg — isolates geometry contribution)
- [ ] Path B: prior latent diffusion baseline (Diffusion-LM style, no JEPA, no SIGReg)
- [ ] Path A: masked diffusion from scratch (no JEPA init)
- [ ] Path A: masked diffusion from data2vec init (EMA baseline)
- [ ] All main results at ≥ 3 seeds, report variance

---

## Failure modes and what to do

| Failure | Likely cause | Action |
|---|---|---|
| Rank collapses to ≤ 2 by step 5k | SIGReg weight too low | Increase `--lam` from 0.05 → 0.2; check gradient reaches encoder |
| MSE loss plateaus immediately | Mean-collapse: predictor outputs the batch mean | Switch to span masking, check mask ratio (try 0.30) |
| SIGReg loss doesn't decrease | SIGReg gradient not reaching encoder | Verify `clean_embs.requires_grad=True`; run smoke test check [4] |
| Path B: decoder accuracy < 40% | Latent space too entangled for diffusion | Fall back to Path A |
| Path A: JEPA init = diffusion-from-scratch | Phase 1 geometry doesn't survive Phase 2 | Study layer-wise drift; consider frozen-encoder Phase 2 as diagnostic |
| data2vec beats SIGReg on all metrics | EMA provides better targets than SIGReg alone | Honest result — pivot to "when does SIGReg match EMA?" as the contribution |

---

## Quick-reference command sequence

```bash
# Week 1-2: Verify your self-implemented code
python smoke_test.py

# Week 3: De-risk experiment (run all 6, compare)
python train.py --shakespeare --steps 10000 --mask-strategy span  --mask-ratio 0.15 --tiny
python train.py --shakespeare --steps 10000 --mask-strategy span  --mask-ratio 0.30 --tiny
python train.py --shakespeare --steps 10000 --mask-strategy span  --mask-ratio 0.50 --tiny
python train.py --shakespeare --steps 10000 --mask-strategy random --mask-ratio 0.15 --tiny
python train.py --shakespeare --steps 10000 --mask-strategy block  --mask-ratio 0.30 --tiny

# Week 4: Stability check across seeds
python train.py --shakespeare --steps 10000 --mask-strategy span --mask-ratio 0.15 --tiny
# (run 3x with different random seeds, compare rank/cos curves)

# Month 2: Full-scale run (on L20 or cloud)
python train.py --steps 50000 --wandb [--mask-strategy span --mask-ratio 0.30]
```
