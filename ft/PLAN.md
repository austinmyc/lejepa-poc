# Part III plan — CoT as the predictor's ACTION (action-conditioned JEPA finetuning)

Status: PLANNING NOTE (2026-07-27). No code yet. Gated on Part II waves 1–2
(mask/EXPERIMENT_PLAN.md) — paper 1 closes first; this is paper 2.

## The idea, in the world-model frame

LeWorldModel (LeWM, arXiv 2603.19312 — Maes, Le Lidec, Scieur, LeCun,
Balestriero 2026) is a stable end-to-end **action-conditioned** JEPA:
ẑ_{t+1} = Pred(z_t, a_t), trained with next-embedding prediction + a Gaussian
latent regularizer. Port that structure to language finetuning:

    state z_t      = Enc(question)
    action a_t     = CoT  (the reasoning trace — the "plan" that transports
                     question-state to answer-state in latent space)
    next state     = Enc(answer)
    JEPA loss      = d( Pred(Enc(question), embed(CoT)),  sg(Enc(answer)) )

LLM-JEPA (2509.14252) and DLLM-JEPA (2606.00091) both drop the action from
LeCun's original JEPA formulation — their predictor is view→view with no
conditioning. CoT is the natural action variable for reasoning tasks. That gap
is the contribution.

**The selling point (test-time story):** the CoT conditions the *predictor
only* — it never enters the generation loss context. If A2 > A1 (below), the
reasoning got internalized into the representations during training, and at
test time the model answers WITHOUT generating a CoT. "Train-time reasoning
supervision, test-time short inference" — distillation of CoT into the encoder
via action-conditioned JEPA, not into the token stream.

## Positioning

- Paper 1 (mask/) establishes: from-scratch latent prediction is dead (no
  abstraction gap); pairing alone doesn't fix it. The published gains all live
  on pretrained backbones. → Part III moves to the regime where JEPA works
  (pretrained + finetune) and adds the missing JEPA ingredient (the action).
- vs LLM-JEPA / DLLM-JEPA: they = A1 (view→view, no action). We add
  action-conditioning + the control ladder they lack.
- vs LeWM: same architecture pattern, pixels→language, actions→CoT.
- Related-work overlap TO CHECK BEFORE COMMITTING (novelty risks):
  JEPA-Reasoner (OpenReview — generative latent-space reasoning; read
  carefully), Coconut (latent CoT, Meta 2024), implicit-CoT distillation line,
  VLA-JEPA (latent action tokens), Causal-JEPA (aux variables condition the
  predictor), V-JEPA 2-AC (action-conditioned predictor at scale).

## Base model — staged (answer to "LLaDA or Llama?")

| Stage | Model | Regime | Why | Compute |
|---|---|---|---|---|
| A | Llama-3.2-1B or Qwen2.5-1.5B + LoRA | AR (LLM-JEPA setting) | cheap iteration; reproduce LLM-JEPA finetune gain first (GO/NO-GO gate); full A0–A5 ladder | any ≥24GB GPU |
| B | LLaDA-8B (+LoRA/QLoRA) | masked diffusion (DLLM-JEPA setting) | the flagship: diffusion's two-mask-rate views + denoising trajectory is the natural "state sequence" for an action; reproduces DLLM-JEPA then adds CoT-action | ~40GB+ bf16 LoRA; QLoRA if 24GB. CHECK `nvidia-smi` FIRST |

Stage A first, always: if we cannot reproduce the published A1 gain on a small
AR model, everything downstream is uninterpretable. Stage B only after A2>A1
shows life at 1B scale.

## Action-injection designs (pick default in first week)

1. **AdaLN conditioning** (default): a = mean-pool of Enc(CoT) (frozen or
   trained pass); predictor blocks get (scale, shift) from a — exactly our
   existing DiffusionHead/AdaLN machinery (mask/model.py, jepa_diffusion_idea.md).
2. Cross-attention: predictor attends to per-token CoT states (richer, costlier).
3. Prompt-style: prepend projected CoT embedding as prefix tokens to the
   predictor input (LLM-JEPA implements its predictor via special tokens, so
   this is the most faithful extension of their mechanism).

Constraint for ALL designs: CoT tokens NEVER appear in the NTP/diffusion loss
context — otherwise the arm silently becomes CoT-SFT (see A5).

## The ablation ladder (the paper's spine — same control discipline as Part II)

| Arm | Predictor conditioning | Isolates |
|---|---|---|
| A0 | no JEPA term (plain SFT on Q→A) | baseline |
| A1 | JEPA, no action | LLM-JEPA / DLLM-JEPA reproduction (the gate) |
| A2 | **JEPA + real CoT action** | the contribution |
| A3 | JEPA + shuffled CoT (another example's) | faithfulness: example-specific signal vs generic conditioning capacity |
| A4 | JEPA + random vector (dim-matched) | any-conditioning control |
| A5 | no JEPA; CoT-SFT (CoT in generation target) | the standard use of the same data — the honest comparator for "was action-conditioning the right way to spend the CoT?" |

Registered readings: A2 > A3 ≈ A4 ≈ A1 → CoT carries real example-specific
transport signal = clean win. A2 ≈ A3 > A1 → conditioning helps but
faithfulness doesn't = weaker claim, report honestly. A2 ≈ A1 → action inert
in language finetuning = null, still informative next to LeWM's pixel result.
A2 vs A5 at matched data: if A2 ≈ A5 accuracy with NO test-time CoT generation,
the efficiency claim alone carries the paper.
Extension (faithfulness gradient): gold CoT vs model-generated CoT vs corrupted
CoT — does action quality dose-respond?

## Data & eval

- GSM8K (7.5k train, gold CoT) — primary; MetaMathQA (395k) for scale;
  MATH stretch. NL-RX/Spider have no CoT — keep as A1-reproduction tasks only.
- Eval: task accuracy (GSM8K/MATH test, greedy, no CoT at inference for
  A0–A4), plus our probe suite on representations (eff_rank, retrieval,
  linear probes) to tie gains to representation change.
- Log everything to W&B `austinmyc/lejepa` (or a new `lejepa-ft` project).

## GO/NO-GO gates

1. Compute check: `nvidia-smi` — VRAM per GPU decides Stage-B feasibility.
2. Gate 1 (wk 1–2): A1 reproduces a nonzero finetune gain on Stage A. If not,
   debug or stop — do NOT proceed to A2 on a dead A1.
3. Gate 2: A2 vs A1/A3/A4 on GSM8K, 3 seeds (finetuning is cheap enough).
4. Gate 3: port winner to LLaDA-8B (Stage B) = DLLM-JEPA + action.

## Sequencing vs current work

1. NOW: Part II waves 1–2 on the 3 GPUs (mask/EXPERIMENT_PLAN.md) — paper 1.
2. Parallel (no GPU): read JEPA-Reasoner + Coconut for overlap; lock the
   action-injection default; scaffold ft/ (HF + LoRA + GSM8K loader + A0/A1).
3. After wave 2's decision point: Stage A launches on freed GPUs.
