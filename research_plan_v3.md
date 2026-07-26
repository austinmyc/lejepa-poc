# Research Plan v3 (2026-07-27) — from "JEPA fails from scratch" to CoT-as-action

Supersedes research_plan.md / research_plan_v2.md. Working docs:
[mask/EXPERIMENT_PLAN.md](mask/EXPERIMENT_PLAN.md) (runs + commands),
[mask/RESULTS.md](mask/RESULTS.md) (ledger), [mask/THEORY.md](mask/THEORY.md)
(proofs + registered predictions), [ft/PLAN.md](ft/PLAN.md) (Part III).

## The map (2×2: view gap × abstraction gap)

|                      | from scratch                         | pretrained backbone |
|----------------------|--------------------------------------|---------------------|
| same-text masked     | ✅ **Part I, closed** — dead (floors) | cell 2 — next paper |
| genuine paired views | ✅ **Part II cell 1** — pairing not load-bearing; waves 1–4 remaining | cell 3 — **Part III: + CoT as the predictor's action** |

## State of evidence

- **Part I (~40 runs):** pure JEPA ceiling .164 vs BERT .4825; chance floors
  2/P and 1/L **now proved** (THEORY.md Props 1–2, predictions match to 3
  decimals); whitened-target toxicity pathway isolated; latent term redundant
  on CE across 6 loss families, 4 readouts, 4× scale; anchor chain
  .16 → .22 (data2vec implicit) → .45 (explicit CE).
- **Part II cell 1 (6 runs + retrieval):** view gap alone lands BELOW the floor
  (.086/.070); anchored ≈ shuffled on 2 corpora × 2 metrics (MTEB Δ ±.006 with
  sign flip; retrieval .1175 vs .1155, .2805 vs .2690) — the A↔B pairing
  carries no signal that prediction can extract from scratch.
- **Theory:** linear JEPA + whitening = CCA ⇒ the failure is a DYNAMICS story,
  not information-theoretic. Central conjecture: predictive SSL's anti-collapse
  and signal terms are antagonistic; contrastive's are the same term.
  Predictions P1–P4 registered before the deciding runs.

## Paper 1 — "When does latent prediction have signal in language?" (target: ICLR/EMNLP; TMLR floor)

Remaining evidence (all commands in mask/EXPERIMENT_PLAN.md "REMAINING RUNS"):

1. **Wave 1 — confounds (4 runs):** `mlmonly` (batch-matched CE baseline),
   `llmjepa` (λ=0, faithful LLM-JEPA) × {code, summary}. Tests P1/P2.
2. **Wave 2 — positive control (4 runs):** `contrastive` / `con_shuffled`
   (in-batch InfoNCE, the non-JEPA mechanism on the same pairs). Tests P3/P4.
   **Decision point:** contrastive ≫ its shuffle → headline = "pairs are
   extractable from scratch, but only by contrast — prediction cannot"
   (conjecture confirmed, ICLR framing). Contrastive ≈ shuffle → broader null
   (TMLR anatomy framing). Either way publishable.
3. **Wave 3 — seeds (6 runs):** ×3 on anchored/shuffled/mlmonly, one corpus —
   error bars where the null lives.
4. **Wave 4 — calibration (cheap):** BERT-base / MiniLM retrieval on the same
   pairs (also the cell-2 teaser: does a pretrained encoder beat all
   from-scratch arms with zero pair training?).

Writing tracks in parallel: floor propositions into the paper; timeboxed
(1 wk) linear init-dynamics lemma; anisotropy grounding (Ethayarajh, Gao).

## Paper 2 / Part III — CoT as the predictor's action ([ft/PLAN.md](ft/PLAN.md))

LeWM (LeCun/Balestriero 2026) structure ported to language finetuning:
Pred(Enc(question), a=CoT) → Enc(answer). LLM-JEPA/DLLM-JEPA dropped the
action; CoT restores it. CoT conditions the predictor ONLY (never the token
loss) → test-time inference needs no CoT. Staged: Llama-3.2-1B/Qwen-1.5B AR
(Gate 1 = reproduce LLM-JEPA gain) → LLaDA-8B diffusion. Ladder A0–A5 incl.
shuffled-CoT, random-action, and CoT-SFT comparator. GSM8K primary.
Prerequisites: `nvidia-smi` VRAM check; JEPA-Reasoner + Coconut overlap read.

## Execution order (3 GPUs: 0, 2, 3)

1. NOW → Wave 1, then Wave 2 (each ≈ 1 day: 4 runs over 3 GPUs).
2. Retrieval eval after each wave (`bash mask/eval_retrieval_all.sh`) + wave-4
   baselines (minutes).
3. Wave 3 seeds while drafting paper 1.
4. Part III Stage A scaffold in parallel (no GPU); launches when wave 2 frees GPUs.

## Known caveats / risks

- codesearchnet has only a train split — code retrieval overlaps training;
  trust deltas, and lean on XSum (clean test split) for absolutes.
- Cell-1 batch is 96 vs Part I ctrl's 128 — cross-pipeline MTEB comparisons
  carry a token-budget caveat until `mlmonly` lands (P2 resolves this).
- Part III novelty risks: JEPA-Reasoner, Coconut — read before building.
- Never claim "JEPA cannot work in text": all claims scoped to "across all
  tested configurations"; the CCA result forbids the information-theoretic
  phrasing.
