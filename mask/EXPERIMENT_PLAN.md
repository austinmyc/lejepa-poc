# Anchored Text-JEPA — Experiment Plan

> All completed-run numbers live in [RESULTS.md](RESULTS.md) — one row per run,
> updated when each run's MTEB lands.

---

# Part II — The condition map (2026-07 reframe)

The thesis (PAPER_STORY.md) is stated as **two conditions** for latent prediction
to carry signal in language: a **view gap** (real information asymmetry between the
two views — cf. LLM-JEPA's text↔code) and an **abstraction gap** (representations
more abstract than inputs — restored by a pretrained backbone). That defines a 2×2.
The ~40 runs of Part I exhaustively characterize the *interior* of **one corner**
— same-text masked views, from scratch — but never move along either axis. So the
paper currently **asserts** the other three cells (cites LLM-JEPA / DLLM-JEPA) but
never turns the knob in our own harness. Filling the map converts the reconciliation
from citation into controlled demonstration and is the strongest version of the paper.

|                              | **abstraction gap: NO** (from scratch)        | **abstraction gap: YES** (pretrained backbone) |
|------------------------------|-----------------------------------------------|------------------------------------------------|
| **view gap: NO** (same-text masked) | ✅ **DONE** — the dead cell (Part I, ~40 runs) | ⬜ cell 2 — same-text JEPA on a pretrained encoder (isolates abstraction gap) |
| **view gap: YES** (paired texts)    | ⬜ **cell 1 — this scaffold** (isolates view gap) | ⬜ cell 3 = LLM-JEPA regime (both on → the cure); CoT-as-action rides here |

**Why cell 1 is the next run (not cell 3):**
- It **discriminates** the two competing explanations. The thesis ("tokens ARE
  abstractions; text has no gap from scratch") predicts a view gap *alone* may NOT
  be enough — you also need the abstraction gap. LLM-JEPA has *both*, so it cannot
  tell you which condition did the work. Cell 1 can.
- **Both outcomes publish.** Rescued → view gap is sufficient (constructive story).
  Still at the floor → even a real cross-modal view gap can't save from-scratch;
  you *also* need pretrained abstraction — a sharper negative than "masking fails".
- **Most novel cell in the literature** — LLM-JEPA and DLLM-JEPA both start from a
  pretrained backbone; nobody has run from-scratch JEPA on genuine pairs.
- **Reuses the Part I stack** — no HF/LoRA module needed yet (that is cells 2–3).

## Cell 1 — cross-view paired JEPA, from scratch  ⏳ SCAFFOLDED (train_paired.py)

Architecture (view A and view B are DIFFERENT texts forming a semantic pair):

```
x_a → Encoder → Proj → Predictor → mean-pool → pred ─┐
                                                     ├─ MSE (pred, target)   [cross-view]
x_b → Encoder → Proj → mean-pool → target (stop-grad)┘
x_b → Proj(per-token) ─ SIGReg                        [target-view geometry]
x_a(masked) → Encoder → CE decode   (optional --mlm-beta)   [anchor]
```

Data — run **two corpora for robustness** so a positive result must replicate
across pairing types, not ride on one:
- `--pair-source code` — docstring↔code (CROSS-MODAL view gap; LLM-JEPA's regime).
- `--pair-source summary` — document↔summary (XSum; abstractive, MONOMODAL, strong
  abstraction asymmetry). Alternates: `cnndm` (CNN/DM), `simplify` (altlex).
`run_paired.sh` sweeps `$PAIRS` (default "code summary") as sequential 3-arm waves.
Eval: unchanged — frozen encoder mean-pool → MTEB slice, vs the Part I baselines
(`mlm_encoder_ctrl` 0.4485; BERT-base 0.483) and the RQ1 chance floor (0.164).

**Arms** (`run_paired.sh`, 5 arms, `ARMS=` selects a subset; 30k screening tier):

| Arm | Objective flags | Tests |
|---|---|---|
| `pure` | `--mlm-beta 0 --lam L` | does a view gap **alone** escape the 2/P floor? |
| `anchored` | `--mlm-beta 1 --mlm-head encoder --lam L` | LeJEPA cross-view + CE — beat CE-only? |
| `shuffled` | `… --shuffle-pairs` | permute view B in-batch. real≈shuffled → pairing not load-bearing. |
| `llmjepa` | `--mlm-beta 1 --mlm-head encoder --lam 0` | **faithful LLM-JEPA from scratch**: NTP + predictor-MSE, NO SIGReg. Rules out "our SIGReg sterilised it" as the reason for a null. |
| `mlmonly` | `--mlm-beta 1 --mse-weight 0 --lam 0` | CE-only in the paired pipeline (batch-96 matched) — the apples-to-apples baseline for "does cross-view help/hurt". |

**Reads.** anchored/llmjepa > mlmonly **and** shuffled ≈ mlmonly → view gap
load-bearing. pure ≫ 0.164 → view gap alone suffices; pure ≈/< 0.164 → needs the
abstraction gap → cells 2–3. llmjepa ≈ shuffled → the null survives even without
SIGReg (it's from-scratch, not the regulariser).

**Results so far — code, 30k, 1 seed** (W&B `austinmyc/lejepa`, MTEB mean):
pure **0.086** (below floor 0.164) · anchored **0.338** · shuffled **0.344**
(≈ anchored) · vs Part I ctrl 0.4485 @ batch128. So on `code`: view gap alone
fails, and the pairing is not load-bearing (anchored≈shuffled). Pending: `llmjepa`
+ `mlmonly` (confound controls), and the `summary` corpus (robustness replicate).

**Downstream / cross-view retrieval** (`eval_retrieval.py`, `eval_retrieval_all.sh`):
LLM-JEPA's own tasks (GSM8K, NL-Regex, Spider) are GENERATIVE — impossible on a
frozen from-scratch encoder. The encoder-appropriate analogue that directly tests
the paired hypothesis is bitext retrieval: freeze encoder, embed held-out
(A,B) pairs, measure recall@1/@10 + MRR both ways. **anchored > shuffled on
retrieval → the pairing taught the A↔B mapping even if MTEB looked flat**; anchored
≈ shuffled → it didn't. Caveat: `code` (codesearchnet) ships only a `train` split,
so code-retrieval overlaps training — trust the anchored−shuffled *delta*, not the
absolute; `summary` (XSum) has a clean test split.

## REMAINING RUNS — the path to submission (2026-07-27 consolidation)

Theory track (no GPU): floor Props 1–2 + linear-CCA analysis + registered
predictions P1–P4 live in **THEORY.md** — predictions written BEFORE these runs.

| # | Wave | Runs | Command | Closes |
|---|---|---|---|---|
| 1 | **Confounds** (critical) | mlmonly, llmjepa × {code, summary} = 4 | `ARMS="mlmonly llmjepa" PAIRS="code summary" GPUS="0 2 3" bash mask/run_paired.sh` | batch-96 MLM baseline (P2); SIGReg-free faithful LLM-JEPA (P1) |
| 2 | **Positive control** (headline-deciding) | contrastive, con_shuffled × {code, summary} = 4 | `ARMS="contrastive con_shuffled" PAIRS="code summary" GPUS="0 2 3" bash mask/run_paired.sh` | can ANY mechanism extract the pairing from scratch? (P3/P4) |
| 3 | **Seeds** (rigor for the null) | anchored, shuffled, mlmonly × 2 more seeds, one corpus = 6 | wave-1/cell-1 commands + `--seed 2024` / `--seed 7` (append to run name) | error bars measured where the claim lives |
| 4 | **Retrieval calibration** (cheap, minutes) | BERT-base + MiniLM on both corpora + retrieval on every new ckpt | `python mask/eval_retrieval.py --hf-model bert-base-uncased --pair-source code --wandb` (×4 combos); `bash mask/eval_retrieval_all.sh` after each wave | absolute calibration + cell-2 teaser (pretrained beats from-scratch?) |

Decision point after wave 2: contrastive ≫ its shuffle → headline = "pairs are
extractable from scratch, but only by contrast — prediction cannot" (ICLR/EMNLP
framing, THEORY.md conjecture confirmed). Contrastive ≈ shuffle → broader null,
TMLR anatomy framing. Both publishable; the controls make either interpretable.
NOT needed: RQ4 span sweep (superseded by waves 3+5), more corpora, more Part-I
loss families, full cells 2–3 (next paper; BERT retrieval baseline is the teaser).

**Then:** cell 2 (same-text masked JEPA on a pretrained HF encoder + LoRA — cheap,
no paired corpus, isolates the abstraction gap), then cell 3 (both on) with
**CoT-as-action** as the novelty spike (condition the predictor on pooled CoT via
the AdaLN path already in jepa_diffusion_idea.md; controls = shuffled-CoT /
random-vector action, mirroring `--shuffle-pairs`). Cells 2–3 need a new `ft/`
module (HF backbone + LoRA + generation-accuracy eval) — scaffold after cell 1 reads.

---

# Part I — Anchored from-scratch study  (✅ closed — the dead cell)

**Thesis for the paper:** JEPA-style latent prediction (predictor + projection +
SIGReg, no EMA teacher) improves text encoder pretraining over pure MLM — from
scratch — with the predictor earning its contribution where token-space
supervision is weakest (long masked spans).

Architecture under test (fixed since round 2):

```
x_masked → Encoder → h_masked ──→ CE head (decode masked tokens)     [anchor]
                        └→ Proj → Predictor → pred_M ─ MSE → z_clean[mask] (stop-grad)
x_clean  → Encoder → h_clean → grad_scale(α) → Proj → z_clean ─ SIGReg [geometry]
```

Eval: frozen-encoder mean-pool → MTEB slice (STSB, SICK-R, Banking77, 20NG).
Reference points: BERT-base no-FT = 0.483 mean; our `mlm_encoder_ctrl` = 0.4485.

---

## RQ1 — Can pure latent JEPA bootstrap from scratch in text?  ✅ ANSWERED: NO

13-run ablation (lam × normalize_target × EMA × mask ratio/strategy, 30k steps
each). Ceiling 0.164 vs BERT 0.483. Mechanism identified (diagnose.py):
normalized latent MSE sits at its chance floor 2/P for the entire run; the
predictor collapses to a near-constant (R²=0.007, eff_rank 5.7/768). Random
text encoders give zero-information targets — the chicken-and-egg failure.
**Paper role: motivation + analysis section. Done.**

## RQ2 — Does a CE anchor fix it, and where must CE attach?  ✅ ANSWERED

Round 1 (anchor_*): CE anchor works (0.38–0.45 vs 0.164). Attach point matters
more than expected:
- CE on predictor output (proj space): 0.3806; +SIGReg → 0.3043. SIGReg
  isotropizes the space CE must decode from — geometric conflict.
- CE on encoder output (BERT-style): 0.4485. MSE alone was neutral (0.3726 vs
  0.3806); the damage was specifically SIGReg-on-the-decode-space.
- EMA arm: encoder eff_rank collapsed to ~19, mse exploded to ~29.
**Paper role: the attach-point/geometry-conflict finding + mechanism table
(none → drift, SIGReg → contained, EMA → collapse). Done.**

## RQ3 — With CE in encoder space, do the JEPA terms help?  🔄 RUNNING (round 2)

All arms: `--mlm-head encoder --mlm-beta 1.0 --no-normalize-target`, 30k steps.
Baseline: `mlm_encoder_ctrl` (0.4485).

| Arm | mse_w | lam | EMA | Tests |
|---|---|---|---|---|
| enc_jepa_sigreg | 1.0 | 0.001 | – | headline: JEPA+SIGReg > MLM? |
| enc_jepa_sigreg_msew01 | 0.1 | 0.001 | – | gentler latent dose |
| enc_jepa_ema | 1.0 | 0 | ✓ | SIGReg-vs-EMA head-to-head |

Success criteria: `enc_jepa_sigreg ≥ ctrl` on MTEB **and** better encoder
geometry (eff_rank > 117.7, mean_cos < 0.058) → "geometry gains at accuracy
parity or better" (strictly stronger than Boukhari 2606.05173, who got geometry
gains at accuracy-neutral with an EMA teacher). `sigreg > ema` → EMA-free claim.

## RQ4 — Does the PREDICTOR contribute where CE breaks down?  ⏳ ROUND 3 (the JEPA-defining test)

Rationale: 15% random single-token masking is the regime where CE supervision
is strongest and latent prediction most redundant. JEPA's raison d'être
(I-JEPA, LeCun position paper) is abstract prediction of large missing regions
— where exact-token CE entropy explodes and its signal degrades.

**Span-length sweep** at fixed total mask budget (15% of tokens), varying span
length L_span ∈ {1, 4, 8, 16} (needs a small `--span-len` addition to
data.py's span strategy). At each L_span, two arms:
- `mlm_only(L_span)`  — CE only
- `mlm_jepa(L_span)`  — CE + MSE + SIGReg (round-2 winner config)

**Hypothesis: Δ(jepa − only) grows with L_span.** If confirmed → headline JEPA
claim: *latent prediction contributes precisely where token-space supervision
breaks down*. Also retroactively explains RQ1's mask-ratio null (no anchor to
bootstrap from). 8 runs; 2 GPUs × ~4 days at 30k steps, or drop to
{1, 8, 16} = 6 runs.

## RQ5 — Credit attribution + rigor (after RQ3/RQ4 pick a winner)

1. **Predictor-free ablation**: `mse_weight=0, lam=0.001, mlm_head=encoder` —
   CE + SIGReg, no latent prediction. If this matches the winner, the paper is
   about SIGReg, not JEPA; if it's worse, the predictor is load-bearing.
   (One run. The single most important control for reviewer defense.)
2. **α (sigreg_grad_scale) sweep** {0, 0.3, 1.0} on the winner — in the
   anchored regime α is a geometry-dose dial, not collapse insurance; find
   whether encoder isotropy has an optimum.
3. **Seeds**: add `--seed`; ×3 on winner + ctrl. Single-seed deltas < ~0.03
   MTEB mean are not claimable.
4. **Scale check (staged protocol)**: 30k = screening tier — explore widely,
   trust rankings not absolute numbers. 120k (~35 h/run, ≈2B tokens ≈ BERT
   budget) = confirmation tier — only for survivors: best JEPA arm +
   encoder_ctrl + EMA counterpart, ×3 seeds ≈ 9 runs ≈ 3–4 days on 4 GPUs.
   This 120k×3-seed grid IS the paper's main table; the claim is "the gap
   survives/grows with 4× data", and ctrl at 120k doubles as a
   BERT-budget-matched baseline. Everything else stays at 30k permanently.
5. **Wider eval**: add held-out MTEB tasks (e.g. STS12-16, EmotionClassification,
   RedditClustering) to whatever wins before believing tuned numbers.

---

## Paper skeleton (outcome-dependent)

- §3 Why from-scratch text JEPA fails: chance floor, degenerate predictor (RQ1)
- §4 Anchored JEPA: attach-point geometry conflict, mechanism table (RQ2)
- §5 Main result: one of
  - (a) JEPA+SIGReg > MLM at parity geometry-and-accuracy (RQ3 win)
  - (b) predictor earns its place on long spans (RQ4 win — strongest version)
  - (c) honest null: latent term inert in text even anchored; SIGReg as
        drop-in geometry regularizer for MLM; why text ≠ vision for JEPA
- §6 SIGReg vs EMA: EMA-free JEPA in text (either way — EMA collapses)
- Every outcome above is publishable; (b) > (a) > (c) in venue ambition.
