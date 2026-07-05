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
| design2_L8_span_nosig | 1.0 | 0 | – | latent term without SIGReg — innocent? | ⏳ |
| design2_L8_sigreg_only | 0 | .001 | 1.0 | SIGReg without latent — guilty alone? | ⏳ |
| design2_L8_span_sig_a0 | 1.0 | .001 | 0.0 | SIGReg confined to proj (grad-scale rescue) | queued (GPU 0 after glob) |

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
