# Results Ledger — everything tested, in order

Companion to [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). One row per run.
Raw artifacts: W&B `austinmyc/lejepa` (train curves, geometry, MTEB summaries),
`mteb_results/<run>/scores.json` + per-task JSONs on the server.

**Eval:** frozen encoder, mean-pool, MTEB 4-task slice — STSBenchmark (Spearman),
SICK-R (Spearman), Banking77 (accuracy), TwentyNewsgroups (V-measure).
All runs: 30k steps, batch 128, seq 128, OWT, 768d/12L unless noted.
⚠ Single seed everywhere so far — deltas < ~0.03 mean not yet claimable
(ctrl repeat pending, `--seed` flag exists since 2026-07-05).

## Reference baselines

| Model | STS-B | SICK-R | B77 | 20NG | Mean | Note |
|---|---|---|---|---|---|---|
| BERT-base no-FT (mean-pool) | .4729 | .5865 | .6345 | .2362 | **.4825** | eval_baselines.py |
| GPT-2 mean-pool | — | — | — | — | — | never run |

## RQ1 — pure from-scratch JEPA (no anchor). Verdict: FAILS, ceiling ≈ 0.16

| Run | lam | mask | norm_tgt | EMA | STS | SICK | B77 | 20NG | Mean | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| owt 120k **encoder** readout | .006 | .15 rand | T | – | .1453 | .2051 | .0251 | .0514 | .1067 | 120k steps, ~2B tokens |
| owt 120k **proj** readout | .006 | .15 rand | T | – | .1106 | .2045 | .0281 | .0591 | .1006 | same ckpt; readout doesn't matter |
| sweep C | .001 | .15 rand | **F** | – | .1222 | .3479 | .1256 | .0594 | **.1638** | best pure-JEPA config |
| sweep A | 0 | .15 rand | F | – | .1211 | .3514 | .0376 | .0591 | .1423 | diverged (loss 138) |
| sweep B | 1e-4 | .15 rand | F | – | .0798 | .2170 | .0278 | .0580 | .0956 | diverged (loss 9243) |
| sweep D | .006 | .15 rand | F | – | .0081 | .1630 | .0214 | .0501 | .0606 | over-regularized |
| sweep E (α=0 shield) | .001 | .15 rand | T | – | .0525 | .1025 | .0130 | .0405 | .0521 | worst — SIGReg→encoder is load-bearing |
| sweep F | .001 | .15 rand | T | – | .1289 | .1866 | .0239 | .0502 | .0974 | C's exact pair: norm costs −.066 |
| ema_pure | 0 | .15 rand | T | ✓ | .0584 | .2241 | .0213 | .0565 | .0901 | collapsed (loss → 0) |
| ema_sigreg | .001 | .15 rand | T | ✓ | .1451 | .3691 | .0374 | .0544 | .1515 | EMA adds nothing over SIGReg |
| maskratio r30 rand | .001 | .30 rand | T | – | .0856 | .1899 | .0248 | .0485 | .0872 | ratio sweep: monotonic ↓ |
| maskratio r30 block | .001 | .30 blk | T | – | .0715 | .1948 | .0229 | .0521 | .0853 | |
| maskratio r50 rand | .001 | .50 rand | T | – | .0545 | .1823 | .0246 | .0550 | .0791 | |
| maskratio r50 block | .001 | .50 blk | T | – | .0747 | .2306 | .0264 | .0528 | .0961 | |
| maskratio r75 block | .001 | .75 blk | T | – | .0757 | .1558 | .0248 | .0533 | .0774 | I-JEPA "mask harder" does NOT transfer |

**Diagnosis (diagnose.py on 120k ckpt):** training MSE (0.0025) = chance floor
2/P for orthogonal unit vectors; R² over mean-predictor = 0.007;
Var(pred)/Var(target) = 0.011; eff_rank(pred) = 5.7/768 vs target 340.9;
cos(pred, random-context pred) = 0.90. Predictor collapsed to near-constant;
only SIGReg was being optimized. Sweep C ckpt (partial record): context cos =
0.15, R² > 0.05 — task non-degenerate without target normalization.
*(Gap: sweep C's full diagnose §1–2 numbers were never captured — rerun if needed.)*

## RQ2 — round 1 anchor (CE added; attach-point comparison). 2026-07-04

All: β=1.0, .15 random, no-normalize-target, CE head Linear(→50257).

| Run | CE attach | mse_w | lam | EMA | STS | SICK | B77 | 20NG | Mean | enc_rank / cos |
|---|---|---|---|---|---|---|---|---|---|---|
| mlm_encoder_ctrl | encoder | 0 | 0 | – | .5248 | .5589 | .5585 | .1517 | **.4485** | 118 / .058 |
| mlm_only | pred | 0 | 0 | – | .4575 | .4984 | .4328 | .1337 | .3806 | 59 / .181 |
| mlm_jepa | pred | 1.0 | 0 | – | .4288 | .4776 | .4447 | .1394 | .3726 | 68 / .224 |
| mlm_jepa_ema | pred | 1.0 | 0 | ✓ | .3129 | .4500 | .3610 | .1300 | .3136 | 36 / .142 |
| mlm_jepa_sigreg | pred | 1.0 | .001 | – | .3308 | .4341 | .3795 | .0727 | .3043 | 109 / .083 |

**Findings:** (1) CE anchor works — 0.30–0.45 vs 0.164 pure-JEPA ceiling.
(2) Attach point dominates: encoder-CE beats pred-CE by +.07 and nearly matches
BERT-base at ¼ its token budget. (3) The pred-attach damage was specifically
SIGReg isotropizing the decode space (mlm_jepa ≈ mlm_only shows plain MSE
neutral). (4) Mechanism table: no mechanism → mse drifts (peak ~10, MTEB
survives); SIGReg → mse contained (0.73) but decode-space conflict; EMA →
target-scale explosion (mse 29) + encoder rank collapse (118→36).

## RQ3 → superseded by the designed loss (2026-07-05)

The planned round-2 per-token arms (enc_jepa_sigreg mse_w 1.0/0.1, enc_jepa_ema)
were NOT run: round 1 already showed per-token latent MSE ≈ neutral next to CE,
and the design principle (latent term must supervise what CE cannot express)
says it's redundant by construction. GPU budget went to the pooled design
instead. Deferred to wave 2 if needed: enc_jepa_ema (SIGReg-vs-EMA at encoder
attach), encoder_ctrl seed repeat (error bar), enc_sigreg_nopred.

## Wave 1 — designed loss @ L=8 spans (run_design.sh). 2026-07-05. Verdict: NEGATIVE

All: 15% budget, span masking L=8, enc-CE β=1, mse_weight=0, seed 1337, 10k ckpts.

| Run | w_span | w_glob | lam | STS | SICK | B77 | 20NG | Mean | enc_rank/cos | CE@30k | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| design L8_ctrl | 0 | 0 | 0 | .4970 | .5451 | .5087 | .1737 | **.4311** | 59/.138 | 6.28 | 7.9 |
| design L8_jepa_span | 1.0 | 0 | .001 | .1815 | .4194 | .1814 | .1066 | .2222 | 64/.098 | 7.21 | 8.0 |
| design L8_jepa_full | 1.0 | 0.5 | .001 | .1383 | .3760 | .1592 | .1047 | .1945 | 54/.120 | 7.29 | 8.0 |
| design L8_jepa_glob | 0 | 0.5 | .001 | ⏳ (running; CE elevated ~7.27 like the others) | | | | | | | |

**Findings:** (1) Latent terms + SIGReg cost −.21 to −.24 mean; STS hit hardest
(.50→.14–.18). (2) Span-CE premise weakened: L8_ctrl .4311 ≈ random ctrl .4485
(−.017, within plausible noise) despite CE 6.3 vs 3.9 nats — token-CE difficulty
≠ embedding damage. (3) Suspect = SIGReg-into-encoder: every SIGReg(α=1)+anchor
arm craters across both rounds, latent-MSE-without-SIGReg was neutral (round-1
mlm_jepa .373), and all three SIGReg wave-1 arms show the same ~1-nat CE
elevation. Ranks fine → semantic scrambling, not rank collapse.
(4) **Pooled chance floor (trajectory forensics):** l_span flatlined at ~0.125
= 1/8 = variance of an 8-token mean-pool of SIGReg-whitened (unit-var) latents;
l_glob flatlined at ~0.011 ≈ 1/128 — both terms learned only the mean, within
2k steps, then contributed pure noise gradient for 28k steps while CE stalled
~1 nat above ctrl (7.2–7.3 vs 6.07). Same disease as RQ1's 2/P floor, pooled
form: SIGReg-whitened target space leaves ~no context-predictable variance.
→ Wave-3 candidate: pool latent targets in ENCODER space (h_clean), keep
SIGReg out of the target path.

## Wave 2 — culprit isolation @ L=8 (launched 2026-07-05)

| Run | w_span | lam | α | Tests | Result |
|---|---|---|---|---|---|
| design2_L8_span_nosig | 1.0 | 0 | – | latent term without SIGReg — innocent? | superseded by wave 3 |
| design2_L8_sigreg_only | 0 | .001 | 1.0 | SIGReg without latent — guilty alone? | ⏳ (queue on GPU 0) |
| design2_L8_span_sig_a0 | 1.0 | .001 | 0.0 | SIGReg confined to proj (grad-scale rescue) | deferred |

## Wave 3 — encoder-space span-JEPA (--latent-space encoder). THE "JEPA WINS" BET

Design from accumulated constraints: targets = pooled h_clean over masked spans
(semantic, CE-grounded, NOT whitened — the pooled-floor fix); predictor at
d_model; stop-grad targets, no EMA, no SIGReg in the loss (CE prevents collapse;
SIGReg's encoder gradient shown toxic). data2vec-style targets, EMA-free.
Baselines: L8_ctrl .4311 (same masking), random ctrl .4485 (absolute).
Watch: l_span must keep DESCENDING — a flatline = floor again, kills the bet.

| Run | w_span | latent_space | STS | SICK | B77 | 20NG | Mean | CE@30k | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| design3_L8_encspan_w01 | 0.1 | encoder | .5047 | .5421 | .5102 | .1659 | .4307 | 6.28 | exact tie w/ ctrl |
| design3_L8_encspan_w1 | 1.0 | encoder | .4740 | .5336 | .4989 | .1756 | .4205 | 6.30 | −.011 (noise-range) |

(Also: w1 jepa_glob finished at .3064 — least-bad wave-1 SIGReg arm, pattern confirmed.)

**Wave-3 verdict (2026-07-06): NEUTRAL.** Encoder-space targets eliminated the
harm (zero CE tax, 6.28 = ctrl) but added nothing (w01 within .0004 of ctrl).
l_span never descends (drifts up 0.015→0.075 as the target cloud grows with CE
training). Combined with waves 1–2 and round 1: **latent prediction on top of a
CE anchor is harmful when SIGReg-coupled, inert otherwise** — across two target
spaces, three granularities, three doses, and three collapse mechanisms. This
is the paper's central empirical claim.

## Closing runs (2026-07-06) — error bar, split-space, post-hoc geometry

| Run | Config | STS | SICK | B77 | 20NG | Mean | CE | Verdict |
|---|---|---|---|---|---|---|---|---|
| design3_L8_ctrl_seed2024 | = L8_ctrl, seed 2024 | .4773 | .5413 | .4990 | .1705 | .4220 | 6.23 | **seed σ(mean) ≈ ±.01**; deltas < ~.015–.02 = noise |
| design3_L8_splitspace | w_span .1 in PROJ space, lam .001, α=0 | .3553 | .4474 | .2827 | .1485 | .3084 | 7.18 | see below — mechanism refinement |

**Seed-repeat implications:** wave-3 neutrality confirmed (w01 and w1 both
inside ctrl's seed range); SIGReg-arm damages (−.14…−.24) are 10–20× noise —
real; span-vs-random masking cost ≈ .02, marginal.

**Split-space accident → sharper mechanism:** the arm (mis)configured the span
term in proj space, so span-MSE pulled the encoder toward SIGReg-whitened
pooled targets while α=0 blocked SIGReg's DIRECT encoder gradient. Damage
replicated anyway (CE 7.18, mean .308 ≈ wave-1) → **the toxin is the whitened
TARGET geometry traveling through the MSE path, not the regularizer gradient.**
Also: dual-readout via SIGReg-only proj head is degenerate (isotropy without a
data term preserves no semantics; adding one re-opens the target channel).

**Post-hoc manifold fitting (eval_manifold.py, ctrl ckpt, W&B manifold_*):**

| σ | raw mean | whiten mean | manfit mean | notes |
|---|---|---|---|---|
| 0.15 | .4485 | .4044 | .3922 | whiten: STS +.047, kills 20NG (.036) & B77 (−.093); manfit: only transform to HELP 20NG (+.016), hurts similarity |
| 0.05 | .4485 | .4044 | .3949 | σ-insensitive; STS partial recovery (.4355→.4472), still ≪ raw |

**No-free-geometric-lunch (final):** whitening helps only STS, manfit helps
only clustering, nothing beats raw on the mean — post-hoc or in-training,
geometry interventions trade task families. Axis closed.

## Wave 4 — beat-BERT push (launched 2026-07-06)

| Run | Config | Tests | Result |
|---|---|---|---|
| ctrl_random_120k | best recipe (enc-CE, random .15) × 4 budget | budget-matched vs BERT-base .4825 | ⏳ |
| contract_w1 | + manifold contraction w=1.0, σ=.15, bank 4096 | training-time manifold fitting (Austin's idea) | ⏳ |
| contract_w01 | + contraction w=0.1 | dose insurance | ⏳ |

Contraction = MSE(pool(h_clean), manfit(pool, FIFO-bank).detach()); no gradient
through fitting; activates at step 2k. Post-hoc evidence says expect clustering
gains at best (+.016 was the post-hoc effect, seed σ .01 — needs to beat that).

## Wave 5 — distributional prediction (diffusion head). QUEUED

LatentLM/MAR mechanism mapped onto our failures: the chance floors are
mode-averaging (point-MSE → conditional mean → no variance); pooled targets
were variance-collapsed (their σ-VAE finding). Fix: noise-prediction head
ε_θ(z_t, t, c) on per-batch-standardized pooled span targets, c = pooled
predictor output (gradient path to encoder verified). Tests: was the latent
term inert because of point regression, or is the residual truly redundant
with CE? Beat bar: L8 ctrl seed range .422–.431 (+ noise .01).

| Run | w_diff | STS | SICK | B77 | 20NG | Mean | CE | Verdict |
|---|---|---|---|---|---|---|---|---|
| diff_w1 | 1.0 | .4740 | .5346 | .5129 | .1802 | .4254 | 6.12 | **NEUTRAL** — mid ctrl seed range |

l_diff: 1.0 → ~0.20 by step 1k, then flat 29k steps — head learns the easy
(≈unconditional) denoising instantly, extracts nothing more from the condition.
**Wave-5 conclusion: mode-averaging eliminated as the explanation. Six loss
families (point-MSE, pooled, global, EMA-target, SIGReg-coupled, diffusion) —
one verdict: after token CE, the residual latent signal is redundant. The
anatomy paper's central claim is closed.**

## Beyond-mean-pool probes (eval_probes.py, 2026-07-08)

Frozen-checkpoint probes on the MTEB-tied trio + ctrl seed repeat (noise floor):

| metric | ctrl s1337 | ctrl s2024 | encspan_w01 | diff_w1 | ctrl spread |
|---|---|---|---|---|---|
| B77 4-shot | .3997 | .3825 | .3982 | .3948 | .017 |
| B77 16-shot | .6133 | .5936 | .6056 | .6126 | .020 |
| B77 64-shot | .7464 | .7324 | .7453 | .7508 | .014 |
| NER acc | .9173 | .9155 | .9182 | .9169 | .002 |
| NER macro-F1 | .6131 | .6081 | .6167 | .6129 | .005 |
| B77 trained mean-pool head | .7071 | .6879 | .7058 | .7175 | .019 |
| B77 trained attn-pool head | .7702 | .7428 | .7666 | .7672 | .027 |

**Verdict: every JEPA number sits inside the ctrl seed spread, on every rung —
few-shot (I-JEPA's headline eval), token-level NER, and learned pooling. The
redundancy thesis is now readout-independent, not an MTEB artifact.**

**Standalone finding: the mean-pool bottleneck is real and model-independent —
a trained attention pooler beats a trained mean-pool head by +5–6 pts on B77
for every checkpoint (≈.70 → ≈.77). An evaluation-convention observation worth
a paragraph, but not a JEPA differentiator.**

## Status: evidence table complete for the anatomy paper (2026-07-06)

Paper = anatomy/reference: two chance floors (2/P per-token; 1/L pooled),
whitened-target toxicity (α ablation isolates pathway), latent inertness
across all benign designs, EMA/SIGReg/none mechanism table, no-free-lunch
post-hoc comparison, BERT-quality MLM recipe at ¼ budget (.4485, seed σ .01).
Optional flourishes: proj-readout eval of splitspace ckpt; 120k ctrl scale-up
(BERT-budget-matched baseline row).

## RQ4 — span-length sweep (pooled JEPA terms). SCRIPTED, not launched

`run_span.sh`: ctrl vs jepa (w_span=1, lam=.001, per-token MSE off) at
L ∈ {1,4,8,16}, fixed 15% budget. Headline hypothesis: Δ grows with L.
w_glob term implemented but deliberately excluded from the sweep (attribution);
test separately at one L afterwards.

## Bookkeeping rules

- Every new run gets a row here **when its MTEB lands** (config + all 4 task
  scores + mean + one-line verdict). Update the ⏳ tables in place.
- Negative/instability observations go in the row's Note — they are data.
- Keep W&B run names ↔ ledger rows 1:1 so the paper tables can be regenerated.
